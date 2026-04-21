# tests/test_config_roundtrip.py
# Phase 2 Plan 02-03: pure helper-module tests for plugins/memory/cashew/config.py.
# Exercises CashewConfig dataclass, DEFAULTS, CONFIG_FILENAME, get_config_schema,
# resolve_config_path, resolve_db_path, load_config, save_config — all in isolation
# from CashewMemoryProvider. Provider-level tests live in test_initialize_lifecycle.py.

from __future__ import annotations

import dataclasses
import json
import pathlib

import pytest

from plugins.memory.cashew.config import (
    CashewConfig,
    CONFIG_FILENAME,
    DEFAULTS,
    get_config_schema,
    load_config,
    resolve_config_path,
    resolve_db_path,
    save_config,
)


def test_config_filename_is_cashew_json():
    """CONF-02: config file is `cashew.json` directly under hermes_home (flat)."""
    assert CONFIG_FILENAME == "cashew.json"


def test_defaults_contain_exactly_four_keys_with_documented_values():
    """CONF-01: schema declares cashew_db_path, embedding_model, recall_k, sync_queue_timeout."""
    assert set(DEFAULTS.keys()) == {
        "cashew_db_path",
        "embedding_model",
        "recall_k",
        "sync_queue_timeout",
    }
    assert DEFAULTS["cashew_db_path"] == "cashew/brain.db"
    assert DEFAULTS["embedding_model"] == "all-MiniLM-L6-v2"
    assert DEFAULTS["recall_k"] == 5
    assert DEFAULTS["sync_queue_timeout"] == 30.0


def test_cashew_config_dataclass_field_set_matches_defaults():
    """The dataclass and DEFAULTS dict must stay in lockstep — drift here breaks load_config."""
    assert dataclasses.is_dataclass(CashewConfig)
    fields = {f.name for f in dataclasses.fields(CashewConfig)}
    assert fields == set(DEFAULTS.keys())


def test_get_config_schema_shape_is_list_of_field_descriptors():
    """CONF-01 + Decision Point 3: the schema is the source of truth for hermes memory setup.

    Hermes' memory_setup.py iterates `for f in schema` expecting dicts with keys
    like `key`, `description`, `default`, `secret`, `env_var`. We emit a list of
    field descriptors matching that contract (see
    https://hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin#config-schema).
    """
    schema = get_config_schema()
    assert isinstance(schema, list)
    keys = {f["key"] for f in schema}
    assert keys == set(DEFAULTS.keys())
    for field in schema:
        assert "key" in field
        assert "description" in field and len(field["description"]) >= 20, (
            f"field {field['key']!r} description too short for hermes memory setup UX"
        )
        assert "default" in field and field["default"] == DEFAULTS[field["key"]], (
            f"field {field['key']!r} default {field['default']!r} != DEFAULTS {DEFAULTS[field['key']]!r}"
        )
        # Phase 2 has no secret fields (no API keys / credentials).
        assert not field.get("secret", False), f"field {field['key']!r} unexpectedly marked secret"


def test_resolve_config_path_under_hermes_home(tmp_path):
    """CONF-02: config file lives at $HERMES_HOME/cashew.json (no subdirectory)."""
    assert resolve_config_path(tmp_path) == tmp_path / "cashew.json"


def test_resolve_db_path_under_hermes_home(tmp_path):
    """CONF-03: DB lives at $HERMES_HOME/cashew/brain.db (nested) — separate from config."""
    assert resolve_db_path(tmp_path, "cashew/brain.db") == tmp_path / "cashew" / "brain.db"


def test_resolve_db_path_rejects_absolute_paths(tmp_path):
    """CONF-04: absolute db paths bypass profile isolation; reject at config-load time."""
    with pytest.raises(ValueError) as exc:
        resolve_db_path(tmp_path, "/etc/passwd")
    assert "/etc/passwd" in str(exc.value)


def test_load_config_returns_defaults_when_file_absent(tmp_path):
    """No cashew.json → defaults. No exception."""
    cfg = load_config(tmp_path)
    assert cfg == CashewConfig(**DEFAULTS)


def test_load_config_merges_partial_file_over_defaults(tmp_path):
    """Partial cashew.json (1 of 4 keys) → that key honored, others default."""
    (tmp_path / "cashew.json").write_text(json.dumps({"recall_k": 9}))
    cfg = load_config(tmp_path)
    assert cfg.recall_k == 9
    assert cfg.cashew_db_path == DEFAULTS["cashew_db_path"]
    assert cfg.embedding_model == DEFAULTS["embedding_model"]
    assert cfg.sync_queue_timeout == DEFAULTS["sync_queue_timeout"]


def test_load_config_drops_unknown_keys(tmp_path):
    """Future-compat: a `verify_on_startup: true` from Phase 5 won't break Phase 2 load."""
    (tmp_path / "cashew.json").write_text(json.dumps({
        "recall_k": 9,
        "verify_on_startup": True,  # not in Phase 2 schema
    }))
    cfg = load_config(tmp_path)
    assert cfg.recall_k == 9
    # Did not raise; unknown key dropped silently.


def test_load_config_raises_on_corrupt_json(tmp_path):
    """Honest I/O: load_config does not swallow JSONDecodeError (caller decides)."""
    (tmp_path / "cashew.json").write_text("not json {")
    with pytest.raises(json.JSONDecodeError):
        load_config(tmp_path)


def test_load_config_raises_on_non_object_json(tmp_path):
    """JSON list at the top level is invalid — caller gets ValueError, not silent garbage."""
    (tmp_path / "cashew.json").write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_save_config_roundtrip(tmp_path):
    """Goal-level: save_config + load_config returns identical CashewConfig."""
    save_config({"recall_k": 7, "embedding_model": "BAAI/bge-small-en"}, tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.recall_k == 7
    assert cfg.embedding_model == "BAAI/bge-small-en"
    assert cfg.cashew_db_path == DEFAULTS["cashew_db_path"]
    assert cfg.sync_queue_timeout == DEFAULTS["sync_queue_timeout"]


def test_save_config_writes_sorted_keys_with_trailing_newline(tmp_path):
    """Stable diffs: file is sorted-keys + 2-space indent + trailing newline."""
    save_config({}, tmp_path)
    text = (tmp_path / "cashew.json").read_text(encoding="utf-8")
    assert text.endswith("\n"), "config file missing trailing newline"
    parsed = json.loads(text)
    assert list(parsed.keys()) == sorted(parsed.keys()), "keys not sorted"
    # 2-space indent: lines after the opening brace start with two spaces.
    lines = text.splitlines()
    assert lines[0] == "{"
    assert lines[1].startswith("  "), "keys not 2-space indented"


def test_save_config_creates_parent_directories(tmp_path):
    """Real users may have a fresh $HERMES_HOME; save_config must mkdir parents."""
    nested = tmp_path / "fresh-hermes-home"
    # Note: the parent (tmp_path) exists; `nested` does not. save_config must create it.
    save_config({"recall_k": 11}, nested)
    assert (nested / "cashew.json").exists()


def test_save_config_drops_unknown_keys(tmp_path):
    """Defensive: a typo or future-key in `values` doesn't end up in the file."""
    save_config({"recall_k": 11, "telemetry_endpoint": "https://attacker.example"}, tmp_path)
    on_disk = json.loads((tmp_path / "cashew.json").read_text())
    assert "telemetry_endpoint" not in on_disk
    assert on_disk["recall_k"] == 11
