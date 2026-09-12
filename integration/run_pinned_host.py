#!/usr/bin/env python3
"""Run hermes-cashew against an immutable Hermes source tree.

This lane is intentionally outside ``tests/``.  The repository test fixture
injects ``agent.memory_provider`` and ``agent.memory_manager`` stubs, which is
useful for fast unit tests but would make a real-host contract test vacuous.
The command runs each loader mode in a fresh process and imports the host
modules only from the pinned source tree supplied by the caller.

Setup is separate from execution.  See ``integration/README.md`` for the
public archive URL, revision, and SHA256 required before running this script.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

HERMES_REVISION = "990473a79c6b0396b0a648fdd85ee8f7a5c267d3"
HERMES_ARCHIVE_SHA256 = "6c8585bfcb3807b7f0c1038be080429e0465137634745b9e48e11b65a9653bda"
REQUIRED_HOST_FILES = (
    "agent/memory_provider.py",
    "agent/memory_manager.py",
    "agent/auxiliary_client.py",
    "plugins/memory/__init__.py",
    "plugins/plugin_loader.py",
    "cron/jobs.py",
)


def _error(message: str) -> NoReturn:
    raise SystemExit(f"SETUP ERROR: {message}")


def _git_revision(source: Path) -> str | None:
    if not (source / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        _error(f"git revision check exceeded the 30-second bound: {exc}")
    return result.stdout.strip() if result.returncode == 0 else None


def _file_hashes(source: Path) -> dict[str, str]:
    return {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
        and path.name != ".hermes-source.json"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }


def _archive_file_hashes(archive: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with tarfile.open(archive, "r:gz") as handle:
        members = [member for member in handle.getmembers() if member.isfile()]
        for member in members:
            parts = PurePosixPath(member.name).parts
            if len(parts) < 2:
                continue
            payload = handle.extractfile(member)
            if payload is None:
                raise ValueError(f"archive member has no payload: {member.name}")
            result[str(Path(*parts[1:]))] = hashlib.sha256(payload.read()).hexdigest()
    return result


def verify_hermes_source(source: Path, archive: Path | None) -> None:
    """Require either a git checkout or setup metadata for the pinned archive."""
    if not source.is_dir():
        _error(f"Hermes source directory does not exist: {source}")
    missing = [name for name in REQUIRED_HOST_FILES if not (source / name).is_file()]
    if missing:
        _error(f"Hermes source is incomplete; missing: {', '.join(missing)}")

    revision = _git_revision(source)
    if revision is not None:
        if revision != HERMES_REVISION:
            _error(f"Hermes git HEAD is {revision}, expected {HERMES_REVISION}")
        try:
            status = subprocess.run(
                ["git", "-C", str(source), "status", "--porcelain"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            _error(f"git cleanliness check exceeded the 30-second bound: {exc}")
        if status.returncode != 0 or status.stdout.strip():
            _error("Hermes git checkout is dirty; use a clean checkout of the pinned revision")
        return

    archive = archive or source.parent / f"{source.name}.tar.gz"
    if not archive.is_file():
        _error(f"pinned Hermes archive is missing: {archive}")
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive_hash != HERMES_ARCHIVE_SHA256:
        _error(f"Hermes archive SHA256 is {archive_hash}, expected {HERMES_ARCHIVE_SHA256}")
    metadata_path = source / ".hermes-source.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        _error(
            f"{source} is not a pinned git checkout; extract the documented archive "
            f"and create {metadata_path} before running the offline lane ({exc})"
        )
    if metadata.get("revision") != HERMES_REVISION:
        _error(f"archive metadata revision is {metadata.get('revision')!r}, expected {HERMES_REVISION}")
    if metadata.get("archive_sha256") != HERMES_ARCHIVE_SHA256:
        _error("archive metadata SHA256 does not match the pinned public Hermes archive")
    try:
        if _file_hashes(source) != _archive_file_hashes(archive):
            _error("extracted Hermes source contents do not match the verified pinned archive")
    except (OSError, tarfile.TarError, ValueError) as exc:
        _error(f"cannot verify extracted Hermes source against the pinned archive: {exc}")


def _copy_plugin(source: Path, destination: Path) -> None:
    ignored = shutil.ignore_patterns(".git", ".venv", "__pycache__", "graphify-out")
    shutil.copytree(source, destination, ignore=ignored)


def _make_dev_overlay(hermes_source: Path, plugin_source: Path, destination: Path) -> None:
    """Expose the plugin as a bundled/dev provider without mutating Hermes source."""
    (destination / "plugins" / "memory").mkdir(parents=True)
    for relative in ("plugins/__init__.py", "plugins/plugin_loader.py"):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(hermes_source / relative)
    memory_init = destination / "plugins" / "memory" / "__init__.py"
    shutil.copy2(hermes_source / "plugins/memory/__init__.py", memory_init)
    shutil.copytree(
        plugin_source / "plugins" / "memory" / "cashew",
        destination / "plugins" / "memory" / "cashew",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _write_profile(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "memory:\n"
        "  provider: cashew\n"
        "auxiliary:\n"
        "  memory:\n"
        "    provider: custom\n"
        "    model: integration-test-model\n"
        "    base_url: http://127.0.0.1:9/v1\n"
        "    api_key: integration-test-key\n",
        encoding="utf-8",
    )


class _FakeEmbeddingModel:
    def encode(self, texts: Any, **_: Any):
        import numpy as np

        values = [texts] if isinstance(texts, str) else list(texts)
        result = np.ones((len(values), 384), dtype="float32")
        return result[0] if isinstance(texts, str) else result


def _assert_host_provenance(hermes_source: Path) -> None:
    memory_provider_module = importlib.import_module("agent.memory_provider")
    memory_manager_module = importlib.import_module("agent.memory_manager")
    for module in (memory_provider_module, memory_manager_module):
        origin = Path(inspect.getfile(module)).resolve()
        if not origin.is_relative_to(hermes_source.resolve()):
            raise AssertionError(f"host module escaped pinned source: {origin}")
    if "tests._memory_manager_stub" in sys.modules:
        raise AssertionError("test MemoryManager stub was imported into the real-host lane")
    if not hasattr(memory_provider_module.MemoryProvider, "recall_status"):
        raise AssertionError("pinned Hermes MemoryProvider contract was not loaded")


def _prioritize_import_roots(import_root: Path, hermes_source: Path) -> None:
    """Restore declared import precedence after host imports mutate ``sys.path``."""
    roots = {str(import_root), str(hermes_source)}
    sys.path[:] = [entry for entry in sys.path if entry not in roots]
    sys.path.insert(0, str(hermes_source))
    sys.path.insert(0, str(import_root))


def _require_runtime_dependencies() -> None:
    """Turn absent host/runtime packages into actionable setup failures."""
    required = (
        "agent.memory_provider",
        "agent.memory_manager",
        "agent.auxiliary_client",
        "cron.jobs",
        "core.context",
        "numpy",
        "yaml",
    )
    for module_name in required:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            _error(
                f"missing runtime dependency {module_name!r}: {exc}; install the pinned "
                "Hermes/Cashew host environment before running this offline lane"
            )


def _exercise_auxiliary_api() -> None:
    auxiliary = importlib.import_module("agent.auxiliary_client")
    captured: dict[str, Any] = {}

    class FakeExternalClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.api_key = kwargs.get("api_key")
            self.base_url = kwargs.get("base_url")

    original = auxiliary.OpenAI
    auxiliary.OpenAI = FakeExternalClient
    try:
        client, model = auxiliary.get_text_auxiliary_client("memory")
    finally:
        auxiliary.OpenAI = original
    assert client is not None
    assert model == "integration-test-model"
    assert captured["api_key"] == "integration-test-key"
    assert str(captured["base_url"]) == "http://127.0.0.1:9/v1"


def _exercise_real_host(mode: str, hermes_source: Path, plugin_source: Path) -> None:
    with tempfile.TemporaryDirectory(prefix=f"hermes-cashew-199-{mode}-") as raw_tmp:
        temp_root = Path(raw_tmp)
        home = temp_root / "hermes-home"
        user_home = temp_root / "user-home"
        _write_profile(home)
        user_home.mkdir()

        if mode == "flat":
            _copy_plugin(plugin_source, home / "plugins" / "cashew")
            import_root = hermes_source
        elif mode == "dev":
            overlay = home / "hermes-agent"
            _make_dev_overlay(hermes_source, plugin_source, overlay)
            import_root = overlay
        else:
            raise AssertionError(f"unknown loader mode: {mode}")

        for key in list(os.environ):
            if (
                key.startswith("CASHEW_")
                or key in {"HERMES_HOME", "HERMES_PROFILE", "HERMES_CONFIG", "HERMES_ENV"}
                or key.endswith(("_API_KEY", "_ACCESS_TOKEN", "_SECRET", "_PASSWORD"))
            ):
                os.environ.pop(key, None)
        os.environ.update(
            {
                "HERMES_HOME": str(home),
                "HOME": str(user_home),
                "USERPROFILE": str(user_home),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            }
        )
        os.environ["PYTHONPATH"] = os.pathsep.join((str(import_root), str(hermes_source)))
        os.chdir(temp_root)
        # The interpreter may have an editable Hermes checkout in site-packages.
        # Remove only that source path, retaining installed third-party
        # dependencies, then put the declared immutable tree first.
        normalized_host = hermes_source.resolve()
        sys.path[:] = [
            entry for entry in sys.path
            if not entry or Path(entry).resolve() != normalized_host
        ]
        sys.path.insert(0, str(hermes_source))
        sys.path.insert(0, str(import_root))
        _require_runtime_dependencies()
        _assert_host_provenance(hermes_source)
        _prioritize_import_roots(import_root, hermes_source)
        from plugins.memory import discover_memory_providers, load_memory_provider

        discovered = {name: available for name, _, available in discover_memory_providers()}
        assert discovered.get("cashew") is True, discovered
        provider = load_memory_provider("cashew", register_skills=False)
        assert provider is not None
        provider_module = sys.modules[type(provider).__module__]
        provider_origin = Path(inspect.getfile(provider_module)).resolve()
        allowed_plugin_roots = [plugin_source.resolve()]
        if mode == "flat":
            allowed_plugin_roots.append((home / "plugins" / "cashew").resolve())
        else:
            allowed_plugin_roots.append((home / "hermes-agent" / "plugins" / "memory" / "cashew").resolve())
        assert any(provider_origin.is_relative_to(root) for root in allowed_plugin_roots), provider_origin

        provider.save_config(
            {
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_device": "cpu",
                "llm_aux_role": None,
                "sleep_cycles": True,
                "think_cycles": False,
                "auto_extraction": False,
                "sleep_schedule": "every 12h",
            },
            str(home),
        )
        provider_module.load_sentence_transformer = lambda *_args, **_kwargs: _FakeEmbeddingModel()

        from agent.memory_manager import MemoryManager

        manager = MemoryManager(external_prefetch_timeout=2.0)
        manager.add_provider(provider)
        manager.initialize_all(
            "integration-session",
            hermes_home=str(home),
            platform="integration",
            agent_context="primary",
        )
        assert provider.is_available() is True
        assert manager.get_provider("cashew") is provider

        import sqlite3

        assert provider._retriever is not None
        assert provider._db_path is not None
        with sqlite3.connect(str(provider._db_path)) as connection:
            connection.execute(
                "INSERT INTO thought_nodes (id, content, node_type, domain, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                ("integration-node", "integration host contract", "fact", "integration", "2026-09-12T00:00:00"),
            )
            connection.commit()
        recall = manager.prefetch_all("integration", session_id="integration-session")
        assert "integration" in recall.lower(), recall
        manager.sync_all(
            "integration user",
            "integration assistant",
            session_id="integration-session",
            messages=[
                {"role": "user", "content": "integration user"},
                {"role": "assistant", "content": "integration assistant"},
            ],
        )
        assert manager.flush_pending(timeout=10.0) is True
        tool_names = manager.get_all_tool_names()
        assert {"cashew_query", "cashew_extract"}.issubset(tool_names), tool_names
        tool_result = manager.handle_tool_call(
            "cashew_query", {"query": "integration"}, session_id="integration-session"
        )
        assert isinstance(tool_result, str)
        tool_payload = json.loads(tool_result)
        assert tool_payload.get("ok") is True, tool_payload
        assert tool_payload.get("node_count", 0) >= 1, tool_payload
        assert tool_payload.get("context"), tool_payload

        # A controlled legacy-signature mutation must be rejected by the
        # success assertion above; otherwise a host TypeError could be hidden
        # behind MemoryManager's neutral error envelope.
        real_handler = provider.handle_tool_call

        def legacy_handler(name: str, args: dict[str, Any]) -> str:
            return real_handler(name, args)

        provider.handle_tool_call = legacy_handler  # type: ignore[method-assign]
        mutant_payload = json.loads(
            manager.handle_tool_call("cashew_query", {"query": "integration"}, session_id="integration-session")
        )
        provider.handle_tool_call = real_handler  # type: ignore[method-assign]
        assert mutant_payload.get("ok") is not True, mutant_payload
        manager.on_session_switch("integration-session-2", parent_session_id="integration-session", reset=True)
        assert isinstance(manager.on_pre_compress([{"role": "user", "content": "checkpoint"}]), str)
        manager.on_session_end([])

        from cron.jobs import list_jobs

        jobs = [job for job in list_jobs() if job.get("name") == "cashew-sleep-cycle"]
        assert len(jobs) == 1, jobs
        assert jobs[0].get("no_agent") is True
        script_path = home / "scripts" / "cashew-sleep-cycle.py"
        assert script_path.is_file()

        stubs = temp_root / "stubs" / "sentence_transformers"
        stubs.mkdir(parents=True)
        (stubs / "__init__.py").write_text(
            "import numpy as np\n\n"
            "class SentenceTransformer:\n"
            "    def __init__(self, *args, **kwargs): pass\n"
            "    def encode(self, texts, **kwargs):\n"
            "        values = [texts] if isinstance(texts, str) else list(texts)\n"
            "        result = np.ones((len(values), 384), dtype='float32')\n"
            "        return result[0] if isinstance(texts, str) else result\n",
            encoding="utf-8",
        )
        cron_env = os.environ.copy()
        cron_env["PYTHONPATH"] = os.pathsep.join(
            (str(stubs.parent), str(import_root), str(hermes_source))
        )
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=temp_root,
            env=cron_env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode:
            raise AssertionError(
                "real cron script failed; issue #186 must be incorporated before "
                f"#199 can pass this boundary:\n{result.stdout}\n{result.stderr}"
            )
        assert result.stdout.strip(), "cron script produced no JSON output"

        manager.shutdown_all()
        _exercise_auxiliary_api()
        # Cron jobs intentionally persist across provider/session shutdown so
        # a 12-hour schedule is not reset at every session boundary. The next
        # initialization must reconcile this same profile-owned record.
        assert len([job for job in list_jobs() if job.get("name") == "cashew-sleep-cycle"]) == 1


def _run_child(mode: str, hermes_source: Path, plugin_source: Path, hermes_archive: Path | None) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--scenario",
        mode,
        "--hermes-source",
        str(hermes_source),
        "--plugin-source",
        str(plugin_source),
    ]
    if hermes_archive is not None:
        command.extend(("--hermes-archive", str(hermes_archive)))
    try:
        result = subprocess.run(command, env=env, text=True, timeout=180)
    except subprocess.TimeoutExpired as exc:
        _error(f"real Hermes {mode} scenario exceeded the 180-second bound: {exc}")
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-source", type=Path, help="pinned Hermes checkout or prepared archive extraction")
    parser.add_argument("--plugin-source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--hermes-archive", type=Path, help="verified archive matching an extracted Hermes source")
    parser.add_argument("--scenario", choices=("flat", "dev"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    hermes_source = (args.hermes_source or Path(os.environ.get("HERMES_PINNED_SOURCE", ""))).resolve()
    plugin_source = args.plugin_source.resolve()
    verify_hermes_source(hermes_source, args.hermes_archive.resolve() if args.hermes_archive else None)
    if not (plugin_source / "plugins" / "memory" / "cashew" / "__init__.py").is_file():
        _error(f"plugin source is not a hermes-cashew checkout: {plugin_source}")

    if args.scenario:
        _exercise_real_host(args.scenario, hermes_source, plugin_source)
        print(f"PASS real Hermes {args.scenario} loader/lifecycle/auxiliary contract")
        return

    hermes_archive = args.hermes_archive.resolve() if args.hermes_archive else None
    for mode in ("flat", "dev"):
        _run_child(mode, hermes_source, plugin_source, hermes_archive)
    print(f"PASS pinned Hermes {HERMES_REVISION}: flat and dev lifecycle contracts")


if __name__ == "__main__":
    main()
