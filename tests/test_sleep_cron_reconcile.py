"""Unit contracts for persistent sleep-cron desired-state reconciliation."""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from pathlib import Path

from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import CashewConfig


def _install_fake_cron(monkeypatch, jobs: list[dict]):
    removed: list[str] = []
    created: list[dict] = []
    package = types.ModuleType("cron")
    package.__path__ = []
    module = types.ModuleType("cron.jobs")
    module.list_jobs = lambda: list(jobs)
    module.remove_job = lambda job_id: removed.append(job_id)

    def create_job(**kwargs):
        created.append(kwargs)
        return {"id": "replacement"}

    module.create_job = create_job
    monkeypatch.setitem(sys.modules, "cron", package)
    monkeypatch.setitem(sys.modules, "cron.jobs", module)
    return removed, created


def _provider(tmp_path: Path, **changes) -> CashewMemoryProvider:
    provider = CashewMemoryProvider()
    provider._hermes_home = tmp_path
    provider._config = replace(CashewConfig(), **changes)
    return provider


def test_disabled_sleep_removes_persisted_job(tmp_path, monkeypatch):
    removed, created = _install_fake_cron(
        monkeypatch,
        [{"id": "old", "name": "cashew-sleep-cycle", "schedule": "every 12h"}],
    )
    provider = _provider(tmp_path, sleep_cycles=False)

    provider._register_sleep_cron()

    assert removed == ["old"]
    assert created == []
    assert provider._sleep_cron_job_id is None


def test_schedule_change_replaces_job_and_refreshes_script(tmp_path, monkeypatch):
    removed, created = _install_fake_cron(
        monkeypatch,
        [{"id": "old", "name": "cashew-sleep-cycle", "schedule": "every 12h"}],
    )
    script = tmp_path / "scripts" / "cashew-sleep-cycle.py"
    script.parent.mkdir()
    script.write_text("stale plugin code")
    provider = _provider(tmp_path, sleep_schedule="every 6h")

    provider._register_sleep_cron()

    assert removed == ["old"]
    assert created[0]["schedule"] == "every 6h"
    assert provider._sleep_cron_job_id == "replacement"
    packaged = Path(__file__).parents[1] / "plugins/memory/cashew/sleep_cron_script.py"
    assert script.read_text() == packaged.read_text()


def test_matching_job_is_adopted_without_reset(tmp_path, monkeypatch):
    removed, created = _install_fake_cron(
        monkeypatch,
        [{"id": "current", "name": "cashew-sleep-cycle", "schedule": "every 12h"}],
    )
    provider = _provider(tmp_path)

    provider._register_sleep_cron()

    assert removed == []
    assert created == []
    assert provider._sleep_cron_job_id == "current"
