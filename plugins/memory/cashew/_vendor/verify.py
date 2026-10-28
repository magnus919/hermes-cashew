"""Offline provenance verification; never import or execute the engine."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, cast

COMMIT = "fcb4919ac37144bfbeb822eaafc668a4bdceb791"
TREE = "29d97fb93c7998c97be0da8f19b90c6523f9d1e9"
ARCHIVE_SHA256 = "23765a473ab550db86856fd4b8f0a011d6046196f2e87ebb263a752e8d114db3"

PRISTINE_SHA256 = "017a81f3a23c6d3195c3d6b1096d35bb2e83f2ebc069a7853d49a465cf0f568c"
LICENSE_SHA256 = "8f713dc86b956eb075e2e1cb50aebbb166594d89cd26ac2b6f9c1c9737dfca26"


def verify(root: Path | None = None) -> dict[str, Any]:
    """Reject changed/incomplete snapshots and non-import runtime modifications."""
    root = root or Path(__file__).resolve().parent
    manifest = cast(dict[str, Any], json.loads((root / "provenance.json").read_text()))
    if (manifest["commit"], manifest["tree"], manifest["archive_sha256"]) != (
        COMMIT,
        TREE,
        ARCHIVE_SHA256,
    ):
        raise ValueError("unexpected Cashew source identity")
    snapshot = (root / "pristine-core.tar.gz").read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != PRISTINE_SHA256:
        raise ValueError("pristine snapshot hash mismatch")
    license_text = (root / "LICENSE").read_bytes()
    if hashlib.sha256(license_text).hexdigest() != LICENSE_SHA256:
        raise ValueError("upstream license hash mismatch")
    actual = {
        str(path.relative_to(root))
        for folder in ("core", "scripts")
        for path in (root / folder).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if actual != set(manifest["files"]):
        raise ValueError("bundled file inventory mismatch")
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(snapshot), mode="r:gz") as archive:
        for member in archive.getmembers():
            name = member.name
            path = PurePosixPath(name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] not in {"core", "scripts"}
                or member.size > 2_000_000
                or name in seen
            ):
                raise ValueError("invalid pristine snapshot member")
            seen.add(name)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("missing pristine source")
            original = source.read()
            expected = manifest["files"][name]
            if hashlib.sha256(original).hexdigest() != expected["upstream_sha256"]:
                raise ValueError(f"pristine source hash mismatch: {name}")
            relocated = original
            if name.endswith(".py"):
                prefix = (
                    rb"\1from ..core." if name.startswith("scripts/") else rb"\1from ."
                )
                relocated = re.sub(rb"(?m)^([ \t]*)from core\.", prefix, original)
            if name == "scripts/migrate_embeddings.py":
                relocated = relocated.replace(
                    b"ROOT = Path(__file__).resolve().parent.parent\nif str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\n",
                    b"",
                )
            installed = root / name
            if installed.is_symlink():
                raise ValueError(f"bundled source is a symlink: {name}")
            runtime = installed.read_bytes()
            if (
                runtime != relocated
                or hashlib.sha256(runtime).hexdigest() != expected["runtime_sha256"]
            ):
                raise ValueError(
                    f"bundled source differs from import-only relocation: {name}"
                )
    if seen != actual:
        raise ValueError("pristine file inventory mismatch")
    return manifest
