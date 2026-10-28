#!/usr/bin/env python3
"""Reproduce the private Cashew snapshot from the approved upstream archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

COMMIT = "fcb4919ac37144bfbeb822eaafc668a4bdceb791"
TREE = "29d97fb93c7998c97be0da8f19b90c6523f9d1e9"
ARCHIVE_SHA256 = "23765a473ab550db86856fd4b8f0a011d6046196f2e87ebb263a752e8d114db3"
VENDOR = Path(__file__).resolve().parents[1] / "plugins/memory/cashew/_vendor"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    data = args.archive.read_bytes()
    if hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256:
        raise SystemExit("upstream archive SHA-256 mismatch")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            relative = PurePosixPath(*path.parts[1:])
            if (
                member.isfile()
                and relative.parts
                and (
                    relative.parts[0] == "core"
                    or str(relative)
                    in {"scripts/__init__.py", "scripts/migrate_embeddings.py"}
                )
            ):
                if ".." in relative.parts or relative.is_absolute():
                    raise SystemExit("invalid upstream path")
                source = archive.extractfile(member)
                assert source is not None
                files[str(relative)] = source.read()
            elif str(relative) == "LICENSE":
                source = archive.extractfile(member)
                assert source is not None
                license_text = source.read()
    VENDOR.mkdir(parents=True, exist_ok=True)
    (VENDOR / "__init__.py").write_text(
        '"""Private, provenance-verified upstream Cashew engine."""\n'
    )
    (VENDOR / "LICENSE").write_bytes(license_text)
    manifest = {
        "commit": COMMIT,
        "tree": TREE,
        "archive_url": f"https://github.com/magnus919/true/archive/{COMMIT}.tar.gz",
        "archive_sha256": ARCHIVE_SHA256,
        "relocation": "relative intra-engine imports; remove migration CLI sys.path bootstrap",
        "files": {},
        "license_sha256": hashlib.sha256(license_text).hexdigest(),
    }
    pristine = io.BytesIO()
    with tarfile.open(fileobj=pristine, mode="w") as archive:
        for name, original in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(original)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(original))
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
            destination = VENDOR / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(relocated)
            manifest["files"][name] = {
                "upstream_sha256": hashlib.sha256(original).hexdigest(),
                "runtime_sha256": hashlib.sha256(relocated).hexdigest(),
            }
    snapshot = gzip.compress(pristine.getvalue(), mtime=0)
    (VENDOR / "pristine-core.tar.gz").write_bytes(snapshot)
    manifest["pristine_sha256"] = hashlib.sha256(snapshot).hexdigest()
    (VENDOR / "provenance.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Bundled {len(files)} upstream files at {COMMIT}; imports relocated only")


if __name__ == "__main__":
    main()
