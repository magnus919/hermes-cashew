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
import base64
import csv
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
import time
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

HERMES_REVISION = "990473a79c6b0396b0a648fdd85ee8f7a5c267d3"
HERMES_ARCHIVE_SHA256 = (
    "6c8585bfcb3807b7f0c1038be080429e0465137634745b9e48e11b65a9653bda"
)
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


def _parse_cron_result(stdout: str) -> dict[str, Any]:
    """Validate the minimum durable-result contract of one cron invocation."""
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"cron script did not produce JSON: {stdout!r}") from exc
    if not isinstance(payload, dict) or not payload:
        raise AssertionError(f"cron script produced an empty/non-object result: {payload!r}")
    if payload.get("status") not in {"completed", "partial", "unavailable"}:
        raise AssertionError(f"cron script returned an unknown status: {payload!r}")
    if payload.get("nodes_selected", 0) < 1:
        raise AssertionError(f"cron script selected no eligible nodes: {payload!r}")
    return payload


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
            _error(
                "Hermes git checkout is dirty; use a clean checkout of the pinned revision"
            )
        return

    archive = archive or source.parent / f"{source.name}.tar.gz"
    if not archive.is_file():
        _error(f"pinned Hermes archive is missing: {archive}")
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive_hash != HERMES_ARCHIVE_SHA256:
        _error(
            f"Hermes archive SHA256 is {archive_hash}, expected {HERMES_ARCHIVE_SHA256}"
        )
    metadata_path = source / ".hermes-source.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        _error(
            f"{source} is not a pinned git checkout; extract the documented archive "
            f"and create {metadata_path} before running the offline lane ({exc})"
        )
    if metadata.get("revision") != HERMES_REVISION:
        _error(
            f"archive metadata revision is {metadata.get('revision')!r}, expected {HERMES_REVISION}"
        )
    if metadata.get("archive_sha256") != HERMES_ARCHIVE_SHA256:
        _error(
            "archive metadata SHA256 does not match the pinned public Hermes archive"
        )
    try:
        if _file_hashes(source) != _archive_file_hashes(archive):
            _error(
                "extracted Hermes source contents do not match the verified pinned archive"
            )
    except (OSError, tarfile.TarError, ValueError) as exc:
        _error(
            f"cannot verify extracted Hermes source against the pinned archive: {exc}"
        )


def _copy_plugin(source: Path, destination: Path) -> None:
    ignored = shutil.ignore_patterns(".git", ".venv", "__pycache__", "graphify-out")
    shutil.copytree(source, destination, ignore=ignored)


def _installed_wheel_package(repository: Path) -> Path:
    """Locate the provider package from the installed distribution only."""
    try:
        from importlib.metadata import distribution, entry_points

        installed = distribution("hermes-cashew")
        entry = next(
            (item for item in entry_points(group="hermes_agent.plugins")
             if item.name == "cashew"),
            None,
        )
        if entry is None or entry.value != "plugins.memory.cashew":
            _error("installed wheel has no cashew Hermes entry point")
        package = Path(installed.locate_file("plugins/memory/cashew")).resolve()
    except Exception as exc:
        _error(f"hermes-cashew wheel is not installed in this interpreter: {exc}")
    if not package.is_dir():
        _error(f"installed hermes-cashew wheel has no provider package: {package}")
    if package.is_relative_to(repository.resolve()):
        _error(f"wheel scenario resolved the repository provider: {package}")
    _verify_wheel_package(
        package,
        (repository / "plugins" / "memory" / "cashew").resolve(),
        installed.read_text("RECORD"),
    )
    return package


def _verify_wheel_package(package: Path, candidate: Path, record_text: str) -> None:
    """Require installed provider bytes and RECORD hashes to match candidate."""
    record_hashes = {
        Path(row[0]): row[1]
        for row in csv.reader(record_text.splitlines())
        if len(row) == 3 and row[0].startswith("plugins/memory/cashew/") and row[1]
    }
    candidate_files = {
        Path("plugins/memory/cashew") / path.relative_to(candidate)
        for path in candidate.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    installed_files = {
        Path("plugins/memory/cashew") / path.relative_to(package)
        for path in package.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if candidate_files != installed_files:
        _error("installed wheel provider files do not match the candidate tree")
    for relative in sorted(candidate_files):
        relative_package = relative.relative_to("plugins/memory/cashew")
        candidate_path = candidate / relative_package
        installed_path = package / relative_package
        if candidate_path.read_bytes() != installed_path.read_bytes():
            _error(f"installed wheel differs from candidate source: {relative}")
        encoded = record_hashes.get(relative)
        if encoded:
            algorithm, expected = encoded.split("=", 1)
            actual = base64.urlsafe_b64encode(
                hashlib.new(algorithm, candidate_path.read_bytes()).digest()
            ).rstrip(b"=").decode("ascii")
            if actual != expected:
                _error(f"wheel RECORD hash mismatch: {relative}")


def _make_dev_overlay(
    hermes_source: Path, plugin_source: Path, destination: Path,
    *, installed_package: Path | None = None,
) -> None:
    """Expose the plugin as a bundled/dev provider without mutating Hermes source."""
    (destination / "plugins" / "memory").mkdir(parents=True)
    for relative in ("plugins/__init__.py", "plugins/plugin_loader.py"):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(hermes_source / relative)
    memory_init = destination / "plugins" / "memory" / "__init__.py"
    shutil.copy2(hermes_source / "plugins/memory/__init__.py", memory_init)
    if installed_package is None:
        shutil.copytree(
            plugin_source / "plugins" / "memory" / "cashew",
            destination / "plugins" / "memory" / "cashew",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    else:
        (destination / "plugins" / "memory" / "cashew").symlink_to(
            installed_package, target_is_directory=True
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


def _assert_host_provenance(hermes_source: Path) -> None:
    memory_provider_module = importlib.import_module("agent.memory_provider")
    memory_manager_module = importlib.import_module("agent.memory_manager")
    for module in (memory_provider_module, memory_manager_module):
        origin = Path(inspect.getfile(module)).resolve()
        if not origin.is_relative_to(hermes_source.resolve()):
            raise AssertionError(f"host module escaped pinned source: {origin}")
    if "tests._memory_manager_stub" in sys.modules:
        raise AssertionError(
            "test MemoryManager stub was imported into the real-host lane"
        )
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

        installed_package: Path | None = None
        if mode == "flat":
            _copy_plugin(plugin_source, home / "plugins" / "cashew")
            import_root = hermes_source
        elif mode == "wheel":
            installed_package = _installed_wheel_package(plugin_source)
            overlay = home / "hermes-agent"
            _make_dev_overlay(
                hermes_source,
                plugin_source,
                overlay,
                installed_package=installed_package,
            )
            import_root = overlay
        elif mode == "dev":
            overlay = home / "hermes-agent"
            _make_dev_overlay(hermes_source, plugin_source, overlay)
            import_root = overlay
        else:
            raise AssertionError(f"unknown loader mode: {mode}")

        for key in list(os.environ):
            if (
                key.startswith("CASHEW_")
                or key
                in {"HERMES_HOME", "HERMES_PROFILE", "HERMES_CONFIG", "HERMES_ENV"}
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
        os.environ["PYTHONPATH"] = os.pathsep.join(
            (str(import_root), str(hermes_source))
        )
        os.chdir(temp_root)
        # The interpreter may have an editable Hermes checkout in site-packages.
        # Remove only that source path, retaining installed third-party
        # dependencies, then put the declared immutable tree first.
        normalized_host = hermes_source.resolve()
        sys.path[:] = [
            entry
            for entry in sys.path
            if not entry or Path(entry).resolve() != normalized_host
        ]
        sys.path.insert(0, str(hermes_source))
        sys.path.insert(0, str(import_root))
        _require_runtime_dependencies()
        _assert_host_provenance(hermes_source)
        _prioritize_import_roots(import_root, hermes_source)
        from plugins.memory import discover_memory_providers, load_memory_provider

        discovered = {
            name: available for name, _, available in discover_memory_providers()
        }
        assert discovered.get("cashew") is True, discovered
        provider = load_memory_provider("cashew", register_skills=False)
        assert provider is not None
        provider_module = sys.modules[type(provider).__module__]
        provider_origin = Path(inspect.getfile(provider_module)).resolve()
        if mode == "flat":
            allowed_plugin_roots = [(home / "plugins" / "cashew").resolve()]
        elif mode == "wheel":
            assert installed_package is not None
            allowed_plugin_roots = [installed_package.resolve()]
        else:
            allowed_plugin_roots = [
                (home / "hermes-agent" / "plugins" / "memory" / "cashew").resolve(),
            ]
        assert any(
            provider_origin.is_relative_to(root) for root in allowed_plugin_roots
        ), provider_origin

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

        stubs = temp_root / "stubs" / "sentence_transformers"
        stubs.mkdir(parents=True)
        first_request = temp_root / "first-embedding-request"
        failure = "exit" if mode == "flat" else "hang"
        (stubs / "__init__.py").write_text(
            "import os\n"
            "import time\n"
            "from pathlib import Path\n"
            "import numpy as np\n\n"
            f"MARKER = Path({str(first_request)!r})\n"
            f"FAILURE = {failure!r}\n\n"
            "class SentenceTransformer:\n"
            "    def __init__(self, *args, **kwargs): pass\n"
            "    def get_embedding_dimension(self): return 384\n"
            "    def encode(self, texts, **kwargs):\n"
            "        values = [texts] if isinstance(texts, str) else list(texts)\n"
            "        if not MARKER.exists():\n"
            "            MARKER.write_text(FAILURE)\n"
            "            if FAILURE == 'exit': os._exit(23)\n"
            "            time.sleep(60)\n"
            "        result = np.ones((len(values), 384), dtype='float32')\n"
            "        return result[0] if isinstance(texts, str) else result\n",
            encoding="utf-8",
        )
        child_python = temp_root / "embedding-child-python"
        child_python.write_text(
            f"#!{sys.executable}\n"
            "import runpy, sys\n"
            f"sys.path.insert(0, {str(stubs.parent)!r})\n"
            "sys.executable = __file__\n"
            "script = sys.argv.pop(1)\n"
            "runpy.run_path(script, run_name='__main__')\n",
            encoding="utf-8",
        )
        child_python.chmod(0o700)
        process_module = sys.modules[provider_module.EmbeddingSupervisor.__module__]
        process_module.sys.executable = str(child_python)
        assert "sentence_transformers" not in sys.modules

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
        assert provider._model_fn is None

        import sqlite3

        assert provider._retriever is not None
        assert provider._db_path is not None
        assert provider._embedding_supervisor is not None
        supervisor = provider._embedding_supervisor
        supervisor.active_timeout = 0.1
        supervisor.backoff_base = 0.0
        with sqlite3.connect(str(provider._db_path)) as connection:
            assert connection.execute(
                "SELECT 1 FROM embeddings WHERE node_id = ?", ("integration-node",)
            ).fetchone() is None
            connection.execute(
                "INSERT INTO thought_nodes (id, content, node_type, domain, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    "integration-node",
                    "integration host contract",
                    "fact",
                    "integration",
                    "2026-09-12T00:00:00",
                ),
            )
            connection.commit()
        started = time.monotonic()
        recall = manager.prefetch_all("integration", session_id="integration-session")
        assert time.monotonic() - started < 2.0
        assert "integration" in recall.lower(), recall
        assert first_request.read_text() == failure
        assert supervisor.encode(["recovery"]).shape == (1, 384)
        assert "sentence_transformers" not in sys.modules
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
            manager.handle_tool_call(
                "cashew_query",
                {"query": "integration"},
                session_id="integration-session",
            )
        )
        provider.handle_tool_call = real_handler  # type: ignore[method-assign]
        assert mutant_payload.get("ok") is not True, mutant_payload
        manager.on_session_switch(
            "integration-session-2", parent_session_id="integration-session", reset=True
        )
        assert isinstance(
            manager.on_pre_compress([{"role": "user", "content": "checkpoint"}]), str
        )
        manager.on_session_end([])

        from cron.jobs import list_jobs

        jobs = [job for job in list_jobs() if job.get("name") == "cashew-sleep-cycle"]
        assert len(jobs) == 1, jobs
        assert jobs[0].get("no_agent") is True
        script_path = home / "scripts" / "cashew-sleep-cycle.py"
        assert script_path.is_file()
        cron_env = os.environ.copy()
        cron_env["PYTHONPATH"] = os.pathsep.join(
            (str(stubs.parent), str(import_root), str(hermes_source))
        )
        with sqlite3.connect(str(provider._db_path)) as connection:
            metadata = dict(
                connection.execute(
                    "SELECT key, value FROM hermes_provider_meta"
                ).fetchall()
            )
        assert metadata.get("embedding_model") == "all-MiniLM-L6-v2", metadata
        assert metadata.get("embedding_dim") == "384", metadata
        assert metadata.get("vec_dim") == "384", metadata
        result = subprocess.run(
            [str(child_python), str(script_path)],
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
        if result.stdout.strip() == "{}":
            raise AssertionError(f"cron script skipped its cycle: {result.stderr}")
        cron_payload = _parse_cron_result(result.stdout)
        assert cron_payload.get("orphans_embedded", 0) >= 1, cron_payload
        assert cron_payload["status"] in {"completed", "partial"}, cron_payload
        assert cron_payload["dream_generation"] == "skipped", cron_payload
        with sqlite3.connect(str(provider._db_path)) as connection:
            assert connection.execute(
                "SELECT LENGTH(vector) FROM embeddings WHERE node_id = ?",
                ("integration-node",),
            ).fetchone() == (384 * 4,)

        manager.shutdown_all()
        _exercise_auxiliary_api()
        # Cron jobs intentionally persist across provider/session shutdown so
        # a 12-hour schedule is not reset at every session boundary. The next
        # initialization must reconcile this same profile-owned record.
        assert (
            len([job for job in list_jobs() if job.get("name") == "cashew-sleep-cycle"])
            == 1
        )


def _run_child(
    mode: str, hermes_source: Path, plugin_source: Path, hermes_archive: Path | None
) -> None:
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
    parser.add_argument(
        "--hermes-source",
        type=Path,
        help="pinned Hermes checkout or prepared archive extraction",
    )
    parser.add_argument(
        "--plugin-source", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--hermes-archive",
        type=Path,
        help="verified archive matching an extracted Hermes source",
    )
    parser.add_argument(
        "--scenario", choices=("flat", "wheel", "dev"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    hermes_source = (
        args.hermes_source or Path(os.environ.get("HERMES_PINNED_SOURCE", ""))
    ).resolve()
    plugin_source = args.plugin_source.resolve()
    verify_hermes_source(
        hermes_source, args.hermes_archive.resolve() if args.hermes_archive else None
    )
    if not (plugin_source / "plugins" / "memory" / "cashew" / "__init__.py").is_file():
        _error(f"plugin source is not a hermes-cashew checkout: {plugin_source}")

    if args.scenario:
        _exercise_real_host(args.scenario, hermes_source, plugin_source)
        print(f"PASS real Hermes {args.scenario} loader/lifecycle/auxiliary contract")
        return

    hermes_archive = args.hermes_archive.resolve() if args.hermes_archive else None
    for mode in ("flat", "dev", "wheel"):
        _run_child(mode, hermes_source, plugin_source, hermes_archive)
    print(
        f"PASS pinned Hermes {HERMES_REVISION}: flat, dev, and wheel lifecycle contracts"
    )


if __name__ == "__main__":
    main()
