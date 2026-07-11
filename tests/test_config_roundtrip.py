# tests/test_config_roundtrip.py
# Phase 2 Plan 02-03: pure helper-module tests for plugins/memory/cashew/config.py.
# Phase 10 expansion: ~30 keys aligned with upstream Cashew.
# Exercises CashewConfig dataclass, DEFAULTS, CONFIG_FILENAME, get_config_schema,
# resolve_config_path, resolve_db_path, load_config, save_config — all in isolation
# from CashewMemoryProvider. Provider-level tests live in test_initialize_lifecycle.py.

from __future__ import annotations

import dataclasses
import json
import os

import pytest

from plugins.memory.cashew.config import (
    _PROVIDER_BASE_URLS,
    _PROVIDER_ENV_MAP,
    DEFAULTS,
    ENV_VAR_MAP,
    UNSUPPORTED_TUNING_KEYS,
    CashewConfig,
    _env_var_name,
    get_ai_domain,
    get_config_schema,
    get_user_domain,
    load_config,
    resolve_config_path,
    resolve_db_path,
    resolve_model_fn,
    save_config,
)

EXPECTED_KEY_COUNT = 37


def test_defaults_contains_exactly_32_keys_with_documented_values():
    """CONF-01: schema declares all ~32 keys."""
    assert len(DEFAULTS) == EXPECTED_KEY_COUNT
    assert DEFAULTS["cashew_db_path"] == "cashew/brain.db"
    assert DEFAULTS["embedding_model"] == "thenlper/gte-large"
    assert DEFAULTS["recall_k"] == 5
    assert DEFAULTS["sync_queue_timeout"] == 30.0
    assert DEFAULTS["user_domain"] == "user"
    assert DEFAULTS["ai_domain"] == "ai"
    assert DEFAULTS["default_domain"] == "general"
    assert DEFAULTS["auto_classify"] is True
    assert DEFAULTS["domain_classifications"] == [
        "personal",
        "work",
        "projects",
        "learning",
        "system",
    ]
    assert DEFAULTS["domain_separation_enabled"] is True
    assert DEFAULTS["token_budget"] == 2000
    assert DEFAULTS["walk_depth"] == 2
    assert DEFAULTS["similarity_threshold"] == 0.3
    assert DEFAULTS["access_weight"] == 0.2
    assert DEFAULTS["temporal_weight"] == 0.1
    assert DEFAULTS["clustering_eps"] == 0.35
    assert DEFAULTS["clustering_min_samples"] == 3
    assert DEFAULTS["novelty_threshold"] == 0.82
    assert DEFAULTS["max_think_iterations"] == 3
    assert DEFAULTS["think_cycle_nodes"] == 5
    assert DEFAULTS["gc_mode"] == "soft"
    assert DEFAULTS["gc_threshold"] == 0.05
    assert DEFAULTS["gc_grace_days"] == 7
    assert DEFAULTS["gc_protect_types"] == ["seed", "core_memory"]
    assert DEFAULTS["gc_think_cycle_penalty"] == 1.5
    assert DEFAULTS["auto_extraction"] is True
    assert DEFAULTS["think_cycles"] is True
    assert DEFAULTS["sleep_cycles"] is True
    assert DEFAULTS["decay_pruning"] is True
    assert DEFAULTS["pattern_detection"] is True
    assert DEFAULTS["llm_aux_role"] == "memory"
    assert DEFAULTS["think_interval"] == 10


def test_cashew_config_dataclass_field_set_matches_defaults():
    """The dataclass and DEFAULTS dict must stay in lockstep — drift here breaks load_config."""
    assert dataclasses.is_dataclass(CashewConfig)
    fields = {f.name for f in dataclasses.fields(CashewConfig)}
    assert fields == set(DEFAULTS.keys())
    assert len(fields) == EXPECTED_KEY_COUNT


def test_get_config_schema_shape_is_list_of_field_descriptors():
    """CONF-01 + CONFIG-07: schema returns ~30 field descriptors for hermes memory setup."""
    schema = get_config_schema()
    assert isinstance(schema, list)
    assert len(schema) == EXPECTED_KEY_COUNT - len(UNSUPPORTED_TUNING_KEYS)
    keys = {f["key"] for f in schema}
    assert keys == set(DEFAULTS) - UNSUPPORTED_TUNING_KEYS
    for field in schema:
        assert "key" in field
        assert "description" in field and len(field["description"]) >= 20, (
            f"field {field['key']!r} description too short for hermes memory setup UX"
        )
        assert "default" in field and field["default"] == DEFAULTS[field["key"]], (
            f"field {field['key']!r} default {field['default']!r} != DEFAULTS {DEFAULTS[field['key']]!r}"
        )
        assert "env_var" in field
        assert not field.get("secret", False), (
            f"field {field['key']!r} unexpectedly marked secret"
        )


def test_env_var_map_has_30_entries():
    """ENV_VAR_MAP covers all 30 config keys."""
    assert len(ENV_VAR_MAP) == EXPECTED_KEY_COUNT
    for key in DEFAULTS:
        assert key in ENV_VAR_MAP
        assert ENV_VAR_MAP[key] == _env_var_name(key)


def test_env_var_name_derivation():
    """_env_var_name correctly derives CASHEW_* names."""
    assert _env_var_name("cashew_db_path") == "CASHEW_DB_PATH"
    assert _env_var_name("user_domain") == "CASHEW_USER_DOMAIN"
    assert _env_var_name("recall_k") == "CASHEW_RECALL_K"
    assert _env_var_name("ai_domain") == "CASHEW_AI_DOMAIN"
    assert _env_var_name("gc_mode") == "CASHEW_GC_MODE"


def test_domain_helpers():
    """get_user_domain and get_ai_domain return configured values."""
    cfg = CashewConfig()
    assert get_user_domain(cfg) == "user"
    assert get_ai_domain(cfg) == "ai"
    cfg_custom = CashewConfig(user_domain="custom_user", ai_domain="custom_ai")
    assert get_user_domain(cfg_custom) == "custom_user"
    assert get_ai_domain(cfg_custom) == "custom_ai"


def test_resolve_config_path_under_hermes_home(tmp_path):
    """CONF-02: config file lives at $HERMES_HOME/cashew.json (no subdirectory)."""
    assert resolve_config_path(tmp_path) == tmp_path / "cashew.json"


def test_resolve_db_path_under_hermes_home(tmp_path):
    """CONF-03: DB lives at $HERMES_HOME/<db_path_value> — separate from config."""
    assert (
        resolve_db_path(tmp_path, "cashew/brain.db") == tmp_path / "cashew" / "brain.db"
    )


def test_resolve_db_path_rejects_absolute_paths(tmp_path):
    """CONF-04: absolute db paths bypass profile isolation; reject at config-load time."""
    with pytest.raises(ValueError) as exc:
        resolve_db_path(tmp_path, "/etc/passwd")
    assert "/etc/passwd" in str(exc.value)


def test_resolve_db_path_rejects_parent_traversal(tmp_path):
    """CONF-04: relative paths must not traverse above hermes_home."""
    with pytest.raises(ValueError, match="must stay within hermes_home"):
        resolve_db_path(tmp_path, "../outside.db")


def test_resolve_db_path_rejects_symlink_escape(tmp_path):
    """CONF-04: an in-profile symlink must not redirect the DB outside."""
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="must stay within hermes_home"):
        resolve_db_path(tmp_path, "linked/brain.db")


def test_resolve_db_path_allows_nested_profile_path(tmp_path):
    """CONF-03: ordinary nested paths still resolve below hermes_home."""
    expected = tmp_path / "nested" / "cashew" / "brain.db"

    assert resolve_db_path(tmp_path, "nested/cashew/brain.db") == expected


def test_cron_script_db_path_uses_profile_isolation_guard(tmp_path):
    """The standalone cron path must reject the same escapes as the provider."""
    from plugins.memory.cashew.sleep_cron_script import _resolve_db_path

    with pytest.raises(ValueError, match="must stay within hermes_home"):
        _resolve_db_path(tmp_path, {"cashew_db_path": "../outside.db"})

    assert _resolve_db_path(tmp_path, {}) == str(tmp_path / "cashew" / "brain.db")


def test_load_config_returns_defaults_when_file_absent(tmp_path):
    """No cashew.json → defaults for all 30 keys. No exception."""
    cfg = load_config(tmp_path)
    assert cfg == CashewConfig(**DEFAULTS)


def test_load_config_merges_partial_file_over_defaults(tmp_path):
    """Partial cashew.json (1 of 30 keys) → that key honored, others default."""
    (tmp_path / "cashew.json").write_text(
        json.dumps({"recall_k": 9, "user_domain": "ganesh"})
    )
    cfg = load_config(tmp_path)
    assert cfg.recall_k == 9
    assert cfg.user_domain == "ganesh"
    assert cfg.cashew_db_path == DEFAULTS["cashew_db_path"]
    assert cfg.embedding_model == DEFAULTS["embedding_model"]
    assert cfg.gc_mode == DEFAULTS["gc_mode"]
    assert cfg.auto_extraction == DEFAULTS["auto_extraction"]


def test_load_config_drops_unknown_keys(tmp_path):
    """Future-compat: unknown keys are dropped at dataclass construction."""
    (tmp_path / "cashew.json").write_text(
        json.dumps(
            {
                "recall_k": 9,
                "future_key": True,
            }
        )
    )
    cfg = load_config(tmp_path)
    assert cfg.recall_k == 9


def test_load_config_raises_on_corrupt_json(tmp_path):
    """Honest I/O: load_config does not swallow JSONDecodeError (caller decides)."""
    (tmp_path / "cashew.json").write_text("not json {")
    with pytest.raises(json.JSONDecodeError):
        load_config(tmp_path)


def test_load_config_raises_on_non_object_json(tmp_path):
    """JSON list at the top level is invalid — caller gets ValueError."""
    (tmp_path / "cashew.json").write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_load_config_env_override_int(monkeypatch, tmp_path):
    """CONFIG-03: CASHEW_RECALL_K=12 overrides int field."""
    monkeypatch.setenv("CASHEW_RECALL_K", "12")
    cfg = load_config(tmp_path)
    assert cfg.recall_k == 12


def test_load_config_env_override_float(monkeypatch, tmp_path):
    """CONFIG-03: CASHEW_SIMILARITY_THRESHOLD=0.5 overrides float field."""
    monkeypatch.setenv("CASHEW_SIMILARITY_THRESHOLD", "0.5")
    cfg = load_config(tmp_path)
    assert cfg.similarity_threshold == 0.5


def test_load_config_env_override_bool(monkeypatch, tmp_path):
    """CONFIG-03: CASHEW_AUTO_CLASSIFY=false overrides bool field to False."""
    monkeypatch.setenv("CASHEW_AUTO_CLASSIFY", "false")
    cfg = load_config(tmp_path)
    assert cfg.auto_classify is False


def test_load_config_env_override_bool_yes(monkeypatch, tmp_path):
    """CONFIG-03: CASHEW_THINK_CYCLES=yes sets bool to True."""
    monkeypatch.setenv("CASHEW_THINK_CYCLES", "yes")
    cfg = load_config(tmp_path)
    assert cfg.think_cycles is True


def test_load_config_env_override_list(monkeypatch, tmp_path):
    """CONFIG-03: CASHEW_GC_PROTECT_TYPES=seed,custom overrides list field."""
    monkeypatch.setenv("CASHEW_GC_PROTECT_TYPES", "seed, custom_type")
    cfg = load_config(tmp_path)
    assert cfg.gc_protect_types == ["seed", "custom_type"]


def test_load_config_env_override_string(monkeypatch, tmp_path):
    """CONFIG-03: CASHEW_USER_DOMAIN=custom_domain overrides string field."""
    monkeypatch.setenv("CASHEW_USER_DOMAIN", "custom_domain")
    cfg = load_config(tmp_path)
    assert cfg.user_domain == "custom_domain"


def test_load_config_env_override_invalid_int_skips(monkeypatch, tmp_path, caplog):
    """CONFIG-03: invalid env var (CASHEW_RECALL_K=notanint) logs warning and is skipped."""
    monkeypatch.setenv("CASHEW_RECALL_K", "notanint")
    with caplog.at_level("WARNING", logger="plugins.memory.cashew.config"):
        cfg = load_config(tmp_path)
    assert cfg.recall_k == DEFAULTS["recall_k"]
    assert "CASHEW_RECALL_K" in caplog.text or "recall_k" in caplog.text


def test_load_config_env_override_invalid_float_skips(monkeypatch, tmp_path, caplog):
    """CONFIG-03: invalid float env var logs warning and is skipped."""
    monkeypatch.setenv("CASHEW_SIMILARITY_THRESHOLD", "not-a-float")
    with caplog.at_level("WARNING", logger="plugins.memory.cashew.config"):
        cfg = load_config(tmp_path)
    assert cfg.similarity_threshold == DEFAULTS["similarity_threshold"]


def test_load_config_env_overrides_file(tmp_path):
    """CONFIG-05 / D-05: priority is env var > JSON file > hardcoded defaults."""
    os.environ["CASHEW_RECALL_K"] = "99"
    try:
        (tmp_path / "cashew.json").write_text(json.dumps({"recall_k": 7}))
        cfg = load_config(tmp_path)
        assert cfg.recall_k == 99  # env wins over file
    finally:
        del os.environ["CASHEW_RECALL_K"]


def test_save_config_roundtrip(tmp_path):
    """Goal-level: save_config + load_config returns identical CashewConfig for all types."""
    save_config(
        {
            "recall_k": 7,
            "embedding_model": "BAAI/bge-small-en",
            "similarity_threshold": 0.45,
            "auto_classify": False,
            "gc_protect_types": ["seed", "custom"],
        },
        tmp_path,
    )
    cfg = load_config(tmp_path)
    assert cfg.recall_k == 7
    assert cfg.embedding_model == "BAAI/bge-small-en"
    assert cfg.similarity_threshold == 0.45
    assert cfg.auto_classify is False
    assert cfg.gc_protect_types == ["seed", "custom"]
    assert cfg.cashew_db_path == DEFAULTS["cashew_db_path"]
    assert cfg.sync_queue_timeout == DEFAULTS["sync_queue_timeout"]
    assert cfg.user_domain == DEFAULTS["user_domain"]
    assert cfg.gc_mode == DEFAULTS["gc_mode"]


def test_save_config_writes_sorted_keys_with_trailing_newline(tmp_path):
    """Stable diffs: file is sorted-keys + 2-space indent + trailing newline."""
    save_config({}, tmp_path)
    text = (tmp_path / "cashew.json").read_text(encoding="utf-8")
    assert text.endswith("\n"), "config file missing trailing newline"
    parsed = json.loads(text)
    assert list(parsed.keys()) == sorted(parsed.keys()), "keys not sorted"
    lines = text.splitlines()
    assert lines[0] == "{"
    assert lines[1].startswith("  "), "keys not 2-space indented"


def test_save_config_creates_parent_directories(tmp_path):
    """Real users may have a fresh $HERMES_HOME; save_config must mkdir parents."""
    nested = tmp_path / "fresh-hermes-home"
    save_config({"recall_k": 11}, nested)
    assert (nested / "cashew.json").exists()


def test_save_config_drops_unknown_keys_from_values(tmp_path):
    """CONFIG-06: unknown keys in values dict are dropped."""
    save_config({"recall_k": 11, "telemetry_url": "https://attacker.example"}, tmp_path)
    on_disk = json.loads((tmp_path / "cashew.json").read_text())
    assert "telemetry_url" not in on_disk
    assert on_disk["recall_k"] == 11


def test_save_config_preserves_unknown_keys_from_existing_file(tmp_path):
    """CONFIG-06: save_config preserves unknown keys from existing JSON file."""
    (tmp_path / "cashew.json").write_text(
        json.dumps({"recall_k": 7, "custom_user_setting": "preserved"})
    )
    save_config({"user_domain": "ganesh"}, tmp_path)
    on_disk = json.loads((tmp_path / "cashew.json").read_text())
    assert on_disk["recall_k"] == 7
    assert on_disk["user_domain"] == "ganesh"
    assert on_disk["custom_user_setting"] == "preserved"
    assert "ai_domain" in on_disk


def test_save_config_v0_1_0_roundtrip_preserves_original_keys(tmp_path):
    """CONFIG-06: saving a v0.1.0-style partial dict preserves the original keys and fills defaults."""
    v0_1_0 = {"recall_k": 11}
    save_config(v0_1_0, tmp_path)
    on_disk = json.loads((tmp_path / "cashew.json").read_text())
    assert on_disk["recall_k"] == 11
    assert on_disk["embedding_model"] == DEFAULTS["embedding_model"]
    assert on_disk["user_domain"] == DEFAULTS["user_domain"]
    assert on_disk["think_cycles"] == DEFAULTS["think_cycles"]


def test_load_config_v0_1_0_four_key_file_gets_defaults_for_new_keys(tmp_path):
    """CONFIG-02: existing 4-key JSON from v0.1.0 loads transparently."""
    v0_1_0 = {
        "cashew_db_path": "custom/brain.db",
        "embedding_model": "BAAI/bge-small-en",
        "recall_k": 9,
        "sync_queue_timeout": 45.0,
    }
    (tmp_path / "cashew.json").write_text(json.dumps(v0_1_0))
    cfg = load_config(tmp_path)

    assert cfg.cashew_db_path == "custom/brain.db"
    assert cfg.embedding_model == "BAAI/bge-small-en"
    assert cfg.recall_k == 9
    assert cfg.sync_queue_timeout == 45.0

    assert cfg.user_domain == DEFAULTS["user_domain"]
    assert cfg.ai_domain == DEFAULTS["ai_domain"]
    assert cfg.token_budget == DEFAULTS["token_budget"]
    assert cfg.walk_depth == DEFAULTS["walk_depth"]
    assert cfg.gc_mode == DEFAULTS["gc_mode"]
    assert cfg.auto_extraction == DEFAULTS["auto_extraction"]
    assert cfg.domain_classifications == list(DEFAULTS["domain_classifications"])
    assert cfg.gc_protect_types == list(DEFAULTS["gc_protect_types"])


# ── resolve_model_fn tests ───────────────────────────────────────────────────


def test_provider_env_map_has_all_entries():
    """_PROVIDER_ENV_MAP covers all well-known providers."""
    assert len(_PROVIDER_ENV_MAP) == 15
    assert _PROVIDER_ENV_MAP["deepseek"] == "DEEPSEEK_API_KEY"
    assert _PROVIDER_ENV_MAP["openai"] == "OPENAI_API_KEY"


def test_provider_base_urls_covers_openai_compatible_providers():
    """_PROVIDER_BASE_URLS has known endpoints for all OpenAI-compatible providers."""
    assert _PROVIDER_BASE_URLS["deepseek"] == "https://api.deepseek.com/v1"
    assert _PROVIDER_BASE_URLS["opencode-zen"] == "https://opencode.ai/zen/v1"
    assert _PROVIDER_BASE_URLS["opencode-go"] == "https://opencode.ai/zen/go/v1"
    assert _PROVIDER_BASE_URLS["openrouter"] == "https://openrouter.ai/api/v1"
    # Anthropic and Google are intentionally NOT in the map.
    assert "anthropic" not in _PROVIDER_BASE_URLS
    assert "google" not in _PROVIDER_BASE_URLS


def test_resolve_model_fn_returns_none_when_no_cashew_json(tmp_path):
    """No cashew.json → resolve_model_fn returns None."""
    result = resolve_model_fn(tmp_path)
    assert result is None


def test_resolve_model_fn_returns_none_when_llm_role_empty(tmp_path):
    """cashew.json exists but llm_aux_role is empty → None."""
    hermes_home = tmp_path / "h1"
    hermes_home.mkdir()
    (hermes_home / "cashew.json").write_text(json.dumps({"llm_aux_role": ""}))
    result = resolve_model_fn(hermes_home)
    assert result is None


def test_resolve_model_fn_returns_none_when_config_yaml_missing(tmp_path):
    """cashew.json with llm_aux_role set, but no config.yaml → None."""
    hermes_home = tmp_path / "h2"
    hermes_home.mkdir()
    (hermes_home / "cashew.json").write_text(json.dumps({"llm_aux_role": "memory"}))
    result = resolve_model_fn(hermes_home)
    assert result is None


def test_resolve_model_fn_returns_none_when_aux_section_missing(tmp_path):
    """config.yaml exists but has no auxiliary.memory section → None."""
    hermes_home = tmp_path / "h3"
    hermes_home.mkdir()
    (hermes_home / "cashew.json").write_text(json.dumps({"llm_aux_role": "memory"}))
    (hermes_home / "config.yaml").write_text("model:\n  provider: test\n")
    result = resolve_model_fn(hermes_home)
    assert result is None


def test_resolve_model_fn_returns_none_when_no_model(tmp_path):
    """auxiliary.memory exists but has no 'model' key → None."""
    hermes_home = tmp_path / "h4"
    hermes_home.mkdir()
    (hermes_home / "cashew.json").write_text(json.dumps({"llm_aux_role": "memory"}))
    (hermes_home / "config.yaml").write_text(
        "auxiliary:\n  memory:\n    provider: deepseek\n"
    )
    result = resolve_model_fn(hermes_home)
    assert result is None


def test_resolve_model_fn_returns_none_when_no_api_key(tmp_path, monkeypatch):
    """auxiliary.memory fully configured but no API key → None."""
    hermes_home = tmp_path / "h5"
    hermes_home.mkdir()
    (hermes_home / "cashew.json").write_text(json.dumps({"llm_aux_role": "memory"}))
    (hermes_home / "config.yaml").write_text(
        "auxiliary:\n  memory:\n    provider: deepseek\n    model: deepseek-v4-flash\n"
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = resolve_model_fn(hermes_home)
    assert result is None


def test_resolve_model_fn_returns_callable_with_env_var(tmp_path, monkeypatch):
    """Valid config + DEEPSEEK_API_KEY → returns callable."""
    hermes_home = tmp_path / "h6"
    hermes_home.mkdir()
    (hermes_home / "cashew.json").write_text(json.dumps({"llm_aux_role": "memory"}))
    (hermes_home / "config.yaml").write_text(
        "auxiliary:\n  memory:\n"
        "    provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
        "    base_url: https://api.deepseek.com/v1\n"
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-123")
    result = resolve_model_fn(hermes_home)
    assert result is not None
    assert callable(result)


def test_resolve_model_fn_returns_callable_with_explicit_api_key(tmp_path):
    """API key in config.yaml (not env var) → returns callable."""
    hermes_home = tmp_path / "h7"
    hermes_home.mkdir()
    (hermes_home / "cashew.json").write_text(json.dumps({"llm_aux_role": "memory"}))
    (hermes_home / "config.yaml").write_text(
        "auxiliary:\n  memory:\n"
        "    provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
        "    base_url: https://api.deepseek.com/v1\n"
        "    api_key: inline-key-456\n"
    )
    result = resolve_model_fn(hermes_home)
    assert result is not None
    assert callable(result)


def test_resolve_model_fn_accepts_passed_config(tmp_path, monkeypatch):
    """Passing a CashewConfig avoids re-reading cashew.json."""
    hermes_home = tmp_path / "h8"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "auxiliary:\n  memory:\n"
        "    provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
        "    base_url: https://api.deepseek.com/v1\n"
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-789")
    cfg = CashewConfig(llm_aux_role="memory")
    result = resolve_model_fn(hermes_home, config=cfg)
    assert result is not None
    assert callable(result)


def test_resolve_model_fn_graceful_on_corrupt_config_yaml(tmp_path):
    """Corrupt config.yaml → returns None, does not raise."""
    hermes_home = tmp_path / "h9"
    hermes_home.mkdir()
    (hermes_home / "cashew.json").write_text(json.dumps({"llm_aux_role": "memory"}))
    (hermes_home / "config.yaml").write_text("::: not yaml :::")
    result = resolve_model_fn(hermes_home)
    assert result is None


def test_resolve_model_fn_uses_default_base_url(tmp_path):
    """No base_url in config → defaults to provider's well-known URL."""
    hermes_home = tmp_path / "h10"
    hermes_home.mkdir()
    (hermes_home / "cashew.json").write_text(json.dumps({"llm_aux_role": "memory"}))
    (hermes_home / "config.yaml").write_text(
        "auxiliary:\n  memory:\n"
        "    provider: openai\n"
        "    model: gpt-4\n"
        "    api_key: sk-test\n"
    )
    result = resolve_model_fn(hermes_home)
    assert result is not None


def test_resolve_model_fn_empty_base_url_resolves_to_provider_default(
    tmp_path, monkeypatch
):
    """base_url: '' (Hermes convention for 'use default') → provider's well-known URL."""
    hermes_home = tmp_path / "h11"
    hermes_home.mkdir()
    (hermes_home / "cashew.json").write_text(json.dumps({"llm_aux_role": "memory"}))
    (hermes_home / "config.yaml").write_text(
        "auxiliary:\n  memory:\n"
        "    provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
        "    base_url: ''\n"
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    result = resolve_model_fn(hermes_home)
    assert result is not None
    # The base_url is captured as a string in the closure
    captured = [
        v.cell_contents for v in result.__closure__ if isinstance(v.cell_contents, str)
    ]
    assert "https://api.deepseek.com/v1" in captured, (
        f"Expected deepseek base_url in closure, got: {captured}"
    )


def test_resolve_model_fn_missing_base_url_uses_provider_map(tmp_path, monkeypatch):
    """No base_url key at all for non-OpenAI provider → resolves from _PROVIDER_BASE_URLS."""
    hermes_home = tmp_path / "h12"
    hermes_home.mkdir()
    (hermes_home / "cashew.json").write_text(json.dumps({"llm_aux_role": "memory"}))
    (hermes_home / "config.yaml").write_text(
        "auxiliary:\n  memory:\n    provider: opencode-zen\n    model: mimo-v2.5-free\n"
    )
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "test-key")
    result = resolve_model_fn(hermes_home)
    assert result is not None
    captured = [
        v.cell_contents for v in result.__closure__ if isinstance(v.cell_contents, str)
    ]
    assert "https://opencode.ai/zen/v1" in captured, (
        f"Expected opencode-zen base_url in closure, got: {captured}"
    )


def test_setup_schema_excludes_unsupported_legacy_tuning_keys():
    from plugins.memory.cashew.config import UNSUPPORTED_TUNING_KEYS, get_config_schema

    advertised = {field["key"] for field in get_config_schema()}
    assert advertised.isdisjoint(UNSUPPORTED_TUNING_KEYS)


def test_non_default_legacy_tuning_value_warns(tmp_path, caplog):
    import json
    import logging

    from plugins.memory.cashew.config import load_config

    (tmp_path / "cashew.json").write_text(json.dumps({"walk_depth": 99}))
    with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew.config"):
        config = load_config(tmp_path)

    assert config.walk_depth == 99
    assert "Ignoring unsupported legacy Cashew settings: walk_depth" in caplog.text
