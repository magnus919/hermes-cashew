# tests/test_e2e_install_lifecycle.py
# Phase 6 Plan 01: INSTALL-04 E2E test
# Tests: hermes plugins install -> memory status lifecycle (INSTALL-04)
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time

from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import get_config_schema, resolve_db_path, DEFAULTS


def test_e2e_install_lifecycle_saves_config_and_reports_available(tmp_path):
    r'''INSTALL-04 Success Criterion #1 + #2:
    E2E test spawns a temporary hermes_home and installs cashew provider
    config into it. E2E test instantiates CashewMemoryProvider and verifies
    is_available() returns True after config exists.
    
    Pattern: directly save config to tmp_path/hermes_home, then instantiate
    provider and check is_available(). No hermes CLI involved (CLI integration
    tested separately in hermes-agent itself); this tests the provider's
    install->available lifecycle.
    '''
    t0 = time.monotonic()
    
    # Step 1: Spawn temporary hermes_home (already provided by tmp_path fixture)
    hermes_home = tmp_path
    
    # Step 2: Install cashew provider config into it
    provider = CashewMemoryProvider()
    provider.save_config({}, str(hermes_home))  # empty config = all defaults
    
    # Step 3: Verify is_available() returns True after config exists
    # We must set _hermes_home so is_available doesn't probe the real ~/.hermes
    provider._hermes_home = hermes_home
    assert provider.is_available() is True, (
        'is_available() must return True after cashew.json exists in hermes_home'
    )
    
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0, f'E2E lifecycle exceeded 5s budget: {elapsed:.2f}s'


def test_e2e_install_lifecycle_schema_returns_field_descriptors(tmp_path):
    r'''INSTALL-04 Success Criterion #3:
    E2E test verifies get_config_schema() returns list-of-field-descriptors format.
    Phase 10: schema expanded to 31 keys.
    '''
    t0 = time.monotonic()

    schema = get_config_schema()

    # Must return a list
    assert isinstance(schema, list), (
        f'get_config_schema() must return list, got {type(schema).__name__}'
    )

    from plugins.memory.cashew.config import DEFAULTS
    expected = len(DEFAULTS)
    assert len(schema) == expected, f'Expected {expected} field descriptors, got {len(schema)}'

    # Each entry must be a dict with required keys (Phase 10 adds env_var)
    required_keys = {'key', 'description', 'default', 'env_var'}
    for i, field in enumerate(schema):
        assert isinstance(field, dict), f'Field #{i} must be dict, got {type(field).__name__}'
        missing = required_keys - field.keys()
        assert not missing, f'Field #{i} missing keys: {missing}'

    # Verify the four original expected keys are present with correct defaults
    keys = {f['key'] for f in schema}
    expected_keys = {'cashew_db_path', 'embedding_model', 'recall_k', 'sync_queue_timeout'}
    assert expected_keys.issubset(keys), f'Original 4 keys not found in schema'

    # Verify default values match config.DEFAULTS for original keys
    defaults_by_key = {f['key']: f['default'] for f in schema}
    assert defaults_by_key['cashew_db_path'] == 'cashew/brain.db'
    assert defaults_by_key['embedding_model'] == 'all-MiniLM-L6-v2'
    assert defaults_by_key['recall_k'] == 5
    assert defaults_by_key['sync_queue_timeout'] == 30.0

    elapsed = time.monotonic() - t0
    assert elapsed < 5.0, f'E2E schema check exceeded 5s budget: {elapsed:.2f}s'


def test_e2e_install_lifecycle_full_provider_init_and_shutdown(tmp_path):
    r'''INSTALL-04 Success Criteria #1 + #2 + #4 (combined):
    Full lifecycle: install config -> initialize provider -> check available ->
    shutdown. Verifies is_available() after initialize() + within 5s budget.
    
    Uses fake_embedder fixture from conftest.py (autouse) to block embedding downloads.
    '''
    t0 = time.monotonic()
    
    hermes_home = tmp_path
    
    # Install config
    provider = CashewMemoryProvider()
    provider.save_config({'recall_k': 3}, str(hermes_home))
    
    # Initialize provider
    provider.initialize(session_id='test-e2e-01', hermes_home=str(hermes_home))
    
    # After initialize, provider must report available
    assert provider.is_available() is True
    
    # Verify config was loaded correctly
    assert provider._config is not None
    assert provider._config.recall_k == 3
    assert provider._config.cashew_db_path == 'cashew/brain.db'
    
    # Verify DB path resolved correctly
    assert provider._db_path is not None
    expected_db = hermes_home / 'cashew' / 'brain.db'
    assert provider._db_path == expected_db
    
    # Shutdown must be safe
    provider.shutdown()
    
    # Post-shutdown: _retriever and _config cleared, but _hermes_home persists
    # (is_available should still return True since config file still exists)
    assert provider._retriever is None
    assert provider._config is None
    # _hermes_home persists per shutdown() contract
    # is_available probes file system directly, not internal state
    provider._hermes_home = hermes_home  # re-set since we tested the cleared state
    assert provider.is_available() is True
    
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0, f'Full E2E lifecycle exceeded 5s budget: {elapsed:.2f}s'


def test_e2e_full_lifecycle_with_retrieval_and_sync(tmp_path):
    """INSTALL-04 Criterion 1 + 3: Full E2E lifecycle exercising prefetch,
    sync_turn, and proper teardown — verifying the complete pipeline works
    end-to-end with real retrieval via keyword fallback."""
    t0 = time.monotonic()
    hermes_home = tmp_path

    provider = CashewMemoryProvider()
    provider.save_config({"recall_k": 5}, str(hermes_home))
    provider.initialize("e2e-full", hermes_home=str(hermes_home))
    try:
        assert provider._config is not None
        assert provider._retriever is not None

        db_path = resolve_db_path(hermes_home, DEFAULTS["cashew_db_path"])

        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO thought_nodes (id, content, node_type, domain, timestamp) "
            "VALUES ('n1', 'E2E test content for retrieval', 'test', 'e2e', '2026-01-01T00:00:00')"
        )
        conn.commit()
        conn.close()

        context = provider.prefetch("E2E test")
        assert "E2E test content" in context, f"prefetch should find seeded node, got: {context!r}"
        assert "=== RELEVANT CONTEXT ===" in context

        provider.sync_turn("hello from test", "hello from assistant")
        assert provider._sync_queue is not None
        assert provider._sync_queue.unfinished_tasks >= 1

        provider.on_session_end([])

    finally:
        provider.shutdown()

    elapsed = time.monotonic() - t0
    assert elapsed < 15.0, f"E2E full lifecycle exceeded 15s budget: {elapsed:.2f}s"


def test_concurrent_sync_stress_no_db_locked(tmp_path):
    """INSTALL-04 Criterion 4: Multiple threads calling sync_turn concurrently
    must not produce 'database is locked' errors. Exercises the write path
    under concurrent load that simulates bursty multi-session patterns."""
    hermes_home = tmp_path
    provider = CashewMemoryProvider()
    provider.save_config({"sync_queue_timeout": 10.0}, str(hermes_home))
    provider.initialize("stress", hermes_home=str(hermes_home))
    thread_count = 4
    turns_per_thread = 10
    barrier = threading.Barrier(thread_count)
    errors: list[Exception] = []
    lock = threading.Lock()

    def _worker(tid: int):
        try:
            barrier.wait()
            for i in range(turns_per_thread):
                provider.sync_turn(f"stress_u{tid}", f"stress_a{tid}")
        except Exception as e:
            with lock:
                errors.append(e)

    threads = [
        threading.Thread(target=_worker, args=(tid,), daemon=True)
        for tid in range(thread_count)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    provider.shutdown()

    assert not errors, f"Concurrent sync_turn raised errors: {errors}"
    # Queue must have drained or be empty after shutdown
    assert provider._sync_queue is None or provider._sync_queue.unfinished_tasks == 0
