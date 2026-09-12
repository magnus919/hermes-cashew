"""Unit contracts for persistent sleep-cron desired-state reconciliation."""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

import plugins.memory.cashew as provider_module
from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import CashewConfig
from plugins.memory.cashew.cron_reconcile import (
    CRON_JOB_NAME,
    CRON_SCRIPT_NAME,
    cron_prompt,
    profile_identity,
    stage_script,
)


def _install_fake_cron(monkeypatch, jobs: list[dict]):
    removed: list[str] = []
    created: list[dict] = []
    updated: list[tuple[str, dict]] = []
    stores: list[Path] = []
    package = types.ModuleType("cron")
    package.__path__ = []
    module = types.ModuleType("cron.jobs")
    list_calls: list[bool] = []

    def list_jobs(*, include_disabled: bool = False):
        list_calls.append(include_disabled)
        return [
            job
            for job in jobs
            if include_disabled or job.get("enabled", True) is not False
        ]

    module.list_jobs = list_jobs
    module.list_calls = list_calls
    module.remove_job = lambda job_id: removed.append(job_id)
    module.parse_schedule = lambda schedule: schedule

    @contextmanager
    def use_cron_store(home):
        stores.append(Path(home))
        yield

    def update_job(job_id, updates):
        updated.append((job_id, updates))
        for job in jobs:
            if job.get("id") == job_id:
                job.update(updates)
                return job
        return None

    def create_job(**kwargs):
        created.append(kwargs)
        return {"id": "replacement"}

    module.create_job = create_job
    module.update_job = update_job
    module.use_cron_store = use_cron_store
    monkeypatch.setitem(sys.modules, "cron", package)
    monkeypatch.setitem(sys.modules, "cron.jobs", module)
    return removed, created, updated, stores


def _provider(tmp_path: Path, **changes) -> CashewMemoryProvider:
    source = Path(__file__).parents[1] / "plugins" / "memory" / "cashew"
    anchor = tmp_path / "hermes-agent" / "plugins" / "memory" / "cashew"
    if not anchor.exists():
        anchor.parent.mkdir(parents=True, exist_ok=True)
        anchor.symlink_to(source, target_is_directory=True)
    provider = CashewMemoryProvider()
    provider._hermes_home = tmp_path
    provider._config = replace(CashewConfig(), **changes)
    provider._embedding_identity_ready = True
    return provider


def _owned_job(home: Path, job_id: str, schedule: str) -> dict:
    return {
        "id": job_id,
        "name": CRON_JOB_NAME,
        "script": CRON_SCRIPT_NAME,
        "prompt": cron_prompt(profile_identity(home)),
        "schedule": schedule,
        "no_agent": True,
        "repeat": {"times": None, "completed": 0},
    }


def test_disabling_sleep_removes_persisted_job_before_host_uninstall(
    tmp_path, monkeypatch
):
    """Disabling the provider is the supported pre-uninstall cleanup path."""
    removed, created, _updated, stores = _install_fake_cron(
        monkeypatch,
        [
            _owned_job(tmp_path, "old", "every 12h"),
            {"id": "other", "name": CRON_JOB_NAME, "schedule": "every 12h"},
        ],
    )
    provider = _provider(tmp_path, sleep_cycles=False)

    provider._register_sleep_cron()

    assert removed == ["old"]  # unproven legacy/other-profile jobs survive
    assert created == []
    assert provider._sleep_cron_job_id is None
    assert stores == [tmp_path]
    assert sys.modules["cron.jobs"].list_calls == [True]


def test_disabled_owned_job_is_adopted_without_creating_a_duplicate(
    tmp_path, monkeypatch
):
    job = _owned_job(tmp_path, "paused", "every 12h")
    job["enabled"] = False
    removed, created, _updated, _stores = _install_fake_cron(monkeypatch, [job])
    provider = _provider(tmp_path)

    provider._register_sleep_cron()

    assert provider._sleep_cron_job_id == "paused"
    assert removed == []
    assert created == []
    assert sys.modules["cron.jobs"].list_calls == [True]


def test_suspend_removes_disabled_owned_job(tmp_path, monkeypatch):
    job = _owned_job(tmp_path, "paused", "every 12h")
    job["enabled"] = False
    removed, _created, _updated, _stores = _install_fake_cron(monkeypatch, [job])
    provider = _provider(tmp_path)

    provider._suspend_sleep_cron()

    assert removed == ["paused"]
    assert sys.modules["cron.jobs"].list_calls == [True]


def test_schedule_change_updates_owned_job_and_refreshes_script(tmp_path, monkeypatch):
    job = _owned_job(tmp_path, "old", "every 12h")
    job["next_run_at"] = "2030-01-01T00:00:00+00:00"
    removed, created, updated, stores = _install_fake_cron(
        monkeypatch,
        [job],
    )
    script = tmp_path / "scripts" / "cashew-sleep-cycle.py"
    script.parent.mkdir()
    script.write_text("stale plugin code")
    provider = _provider(tmp_path, sleep_schedule="every 6h")

    provider._register_sleep_cron()

    assert removed == []
    assert created == []
    assert updated == [("old", {"schedule": "every 6h"})]
    assert provider._sleep_cron_job_id == "old"
    assert job["next_run_at"] == "2030-01-01T00:00:00+00:00"
    assert stores == [tmp_path]
    assert "_INSTALLATION_MARKER = {" in script.read_text()


def test_matching_job_is_adopted_without_reset(tmp_path, monkeypatch):
    removed, created, _updated, stores = _install_fake_cron(monkeypatch, [])
    provider = _provider(tmp_path)
    # First registration supplies the real generated marker. A fresh provider
    # can then adopt only the matching profile-tagged scheduler record.
    provider._register_sleep_cron()
    job = _owned_job(tmp_path, "current", "every 12h")
    sys.modules["cron.jobs"].list_jobs = lambda *, include_disabled=False: [job]
    created.clear()
    provider = _provider(tmp_path)

    provider._register_sleep_cron()

    assert removed == []
    assert created == []
    assert provider._sleep_cron_job_id == "current"
    assert stores == [tmp_path, tmp_path]


def test_concurrent_reconciliation_adopts_one_profile_job(tmp_path, monkeypatch):
    """The profile lock turns simultaneous initialize calls into one job."""
    jobs: list[dict] = []
    removed, created, _updated, _stores = _install_fake_cron(monkeypatch, jobs)

    def create_job(**kwargs):
        created.append(kwargs)
        job = {"id": f"job-{len(jobs)}", **kwargs}
        jobs.append(job)
        return job

    sys.modules["cron.jobs"].create_job = create_job
    first = _provider(tmp_path)
    second = _provider(tmp_path)
    import threading

    threads = [
        threading.Thread(target=provider._register_sleep_cron)
        for provider in (first, second)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert [job["id"] for job in jobs] == ["job-0"]
    assert removed == []
    assert len(created) == 1


def test_script_staging_uses_unique_cleanup_on_replace_failure(tmp_path, monkeypatch):
    """A failed refresh cannot leave the old script or a fixed shared stage."""
    destination = tmp_path / "scripts" / CRON_SCRIPT_NAME
    destination.parent.mkdir()
    destination.write_text("old")

    def fail_replace(_source, _target):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr("plugins.memory.cashew.cron_reconcile.os.replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        stage_script(destination, "new")

    assert destination.read_text() == "old"
    assert list(destination.parent.glob(f".{CRON_SCRIPT_NAME}.*.tmp")) == []


def test_reconciliation_uses_real_profile_cron_store(tmp_path):
    """The real Hermes API scopes, adopts, and updates even paused owned jobs."""
    cron_jobs = pytest.importorskip("cron.jobs")
    provider = _provider(tmp_path)
    other_home = tmp_path / "other-profile"

    try:
        provider._register_sleep_cron()
        job_id = provider._sleep_cron_job_id
        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.update_job(job_id, {"enabled": False})

        # Hermes omits disabled records unless callers ask for them.  Reconcile
        # must retain this owned paused job rather than create a duplicate.
        provider._config = replace(provider._config, sleep_schedule="every 6h")
        provider._register_sleep_cron()
        with cron_jobs.use_cron_store(tmp_path):
            owned = [
                job
                for job in cron_jobs.list_jobs(include_disabled=True)
                if job.get("id") == provider._sleep_cron_job_id
            ]
        with cron_jobs.use_cron_store(other_home):
            other_jobs = cron_jobs.list_jobs()

        assert len(owned) == 1
        assert provider._sleep_cron_job_id == job_id
        assert owned[0]["schedule"] == cron_jobs.parse_schedule("every 6h")
        assert owned[0]["enabled"] is False
        assert other_jobs == []
    finally:
        if provider._sleep_cron_job_id:
            with cron_jobs.use_cron_store(tmp_path):
                cron_jobs.remove_job(provider._sleep_cron_job_id)


def test_tampered_cron_template_does_not_install_script_or_job(
    tmp_path, monkeypatch, caplog
):
    _removed, created, _updated, _stores = _install_fake_cron(monkeypatch, [])
    provider = _provider(tmp_path)
    template = Path(provider_module.__file__).parent / "sleep_cron_script.py"
    original_read_text = Path.read_text

    def read_tampered_template(path, *args, **kwargs):
        if path == template:
            return "# marker removed by a damaged installation\n"
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_tampered_template)

    provider._register_sleep_cron()

    assert created == []
    assert provider._sleep_cron_job_id is None
    assert not (tmp_path / "scripts" / "cashew-sleep-cycle.py").exists()
    assert "cron script template is invalid" in caplog.text
