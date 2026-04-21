# tests/test_e2e_install_lifecycle.py
# Phase 6 Plan 01: INSTALL-04 E2E test
# Tests: hermes plugins install -> memory status lifecycle (INSTALL-04)
from __future__ import annotations

import json
import time

from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import get_config_schema


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
    '''
    t0 = time.monotonic()
    
    schema = get_config_schema()
    
    # Must return a list
    assert isinstance(schema, list), (
        f'get_config_schema() must return list, got {type(schema).__name__}'
    )
    
    # Must have exactly 4 entries (one per config key)
    assert len(schema) == 4, f'Expected 4 field descriptors, got {len(schema)}'
    
    # Each entry must be a dict with required keys
    required_keys = {'key', 'description', 'default'}
    for i, field in enumerate(schema):
        assert isinstance(field, dict), f'Field #{i} must be dict, got {type(field).__name__}'
        missing = required_keys - field.keys()
        assert not missing, f'Field #{i} missing keys: {missing}'
    
    # Verify the four expected keys are present
    keys = {f['key'] for f in schema}
    expected_keys = {'cashew_db_path', 'embedding_model', 'recall_k', 'sync_queue_timeout'}
    assert keys == expected_keys, f'Expected keys {expected_keys}, got {keys}'
    
    # Verify default values match config.DEFAULTS
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
