# Graph Report - .  (2026-04-29)

## Corpus Check
- Corpus is ~22,038 words - fits in a single context window. You may not need a graph.

## Summary
- 577 nodes · 813 edges · 87 communities detected
- Extraction: 72% EXTRACTED · 28% INFERRED · 0% AMBIGUOUS · INFERRED: 226 edges (avg confidence: 0.73)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Configuration System|Configuration System]]
- [[_COMMUNITY_CashewMemoryProvider Core|CashewMemoryProvider Core]]
- [[_COMMUNITY_Test Fixtures & E2E Install|Test Fixtures & E2E Install]]
- [[_COMMUNITY_Tool Schema Validation|Tool Schema Validation]]
- [[_COMMUNITY_Tool Call Handling|Tool Call Handling]]
- [[_COMMUNITY_Sync Turn Queue|Sync Turn Queue]]
- [[_COMMUNITY_Sync Worker Thread|Sync Worker Thread]]
- [[_COMMUNITY_Memory Manager Stub|Memory Manager Stub]]
- [[_COMMUNITY_Database Migration|Database Migration]]
- [[_COMMUNITY_Recall & Prefetch|Recall & Prefetch]]
- [[_COMMUNITY_Sync Burst & Threading|Sync Burst & Threading]]
- [[_COMMUNITY_Retrieval & Context|Retrieval & Context]]
- [[_COMMUNITY_Cashew Extract|Cashew Extract]]
- [[_COMMUNITY_Test Helper Cross-Refs|Test Helper Cross-Refs]]
- [[_COMMUNITY_Plugin Registration|Plugin Registration]]
- [[_COMMUNITY_Quality Metrics|Quality Metrics]]
- [[_COMMUNITY_Plugin Discovery|Plugin Discovery]]
- [[_COMMUNITY_Verify Script|Verify Script]]
- [[_COMMUNITY_Profile Isolation|Profile Isolation]]
- [[_COMMUNITY_Config Migration Bridge|Config Migration Bridge]]
- [[_COMMUNITY_Sync Provider Helpers|Sync Provider Helpers]]
- [[_COMMUNITY_Flat Entrypoint Tests|Flat Entrypoint Tests]]
- [[_COMMUNITY_Home Leak & Profile Isolation|Home Leak & Profile Isolation]]
- [[_COMMUNITY_Corrupt Config Handling|Corrupt Config Handling]]
- [[_COMMUNITY_Half-State Error Handling|Half-State Error Handling]]
- [[_COMMUNITY_Dual Layout & Path Loading|Dual Layout & Path Loading]]
- [[_COMMUNITY_Availability Tests|Availability Tests]]
- [[_COMMUNITY_Recency Weighting|Recency Weighting]]
- [[_COMMUNITY_Config Defaults Validation|Config Defaults Validation]]
- [[_COMMUNITY_Vec Embeddings Guard|Vec Embeddings Guard]]
- [[_COMMUNITY_Thread Exit Helpers|Thread Exit Helpers]]
- [[_COMMUNITY_Slow Session Simulations|Slow Session Simulations]]
- [[_COMMUNITY_Queue Bounds Tests|Queue Bounds Tests]]
- [[_COMMUNITY_Worker Resilience|Worker Resilience]]
- [[_COMMUNITY_Lazy Import Tests|Lazy Import Tests]]
- [[_COMMUNITY_Initialized Provider Helpers|Initialized Provider Helpers]]
- [[_COMMUNITY_Retrieval Exception Handling|Retrieval Exception Handling]]
- [[_COMMUNITY_Drain Queue Helpers|Drain Queue Helpers]]
- [[_COMMUNITY_OK Session Simulations|OK Session Simulations]]
- [[_COMMUNITY_Worker Lifecycle|Worker Lifecycle]]
- [[_COMMUNITY_Pre-Initialize Safety|Pre-Initialize Safety]]
- [[_COMMUNITY_Unknown Tool Routing|Unknown Tool Routing]]
- [[_COMMUNITY_Filtering Tests|Filtering Tests]]
- [[_COMMUNITY_Test Flat Entrypoint Reexports|Test Flat Entrypoint Reexports]]
- [[_COMMUNITY_Test Stub Lifecycle Test Name|Test Stub Lifecycle Test Name ]]
- [[_COMMUNITY_Test Stub Lifecycle Test Is Av|Test Stub Lifecycle Test Is Av]]
- [[_COMMUNITY_Test Retrieval Provider|Test Retrieval Provider]]
- [[_COMMUNITY_Test Retrieval Db Path|Test Retrieval Db Path]]
- [[_COMMUNITY_Test Retrieval Test Prefetch E|Test Retrieval Test Prefetch E]]
- [[_COMMUNITY_Test Retrieval Test Prefetch E|Test Retrieval Test Prefetch E]]
- [[_COMMUNITY_Test Retrieval Test Retrieve K|Test Retrieval Test Retrieve K]]
- [[_COMMUNITY_Test Retrieval Test Format Con|Test Retrieval Test Format Con]]
- [[_COMMUNITY_Test Retrieval Test Format Con|Test Retrieval Test Format Con]]
- [[_COMMUNITY_Test No Home Leak Test Save Co|Test No Home Leak Test Save Co]]
- [[_COMMUNITY_Test Config Roundtrip Test Con|Test Config Roundtrip Test Con]]
- [[_COMMUNITY_Test Config Roundtrip Test Get|Test Config Roundtrip Test Get]]
- [[_COMMUNITY_Test Config Roundtrip Test Env|Test Config Roundtrip Test Env]]
- [[_COMMUNITY_Test Config Roundtrip Test Res|Test Config Roundtrip Test Res]]
- [[_COMMUNITY_Test Config Roundtrip Test Loa|Test Config Roundtrip Test Loa]]
- [[_COMMUNITY_Test Config Roundtrip Test Sav|Test Config Roundtrip Test Sav]]
- [[_COMMUNITY_Test Plugin Discovery Test Ent|Test Plugin Discovery Test Ent]]
- [[_COMMUNITY_Test Plugin Discovery Test Ent|Test Plugin Discovery Test Ent]]
- [[_COMMUNITY_Test Plugin Discovery Test Mod|Test Plugin Discovery Test Mod]]
- [[_COMMUNITY_Test Migration Test No Columns|Test Migration Test No Columns]]
- [[_COMMUNITY_Test Migration Test Existing D|Test Migration Test Existing D]]
- [[_COMMUNITY_Test Migration Test Fresh Db H|Test Migration Test Fresh Db H]]
- [[_COMMUNITY_Test Profile Isolation Forbidd|Test Profile Isolation Forbidd]]
- [[_COMMUNITY_Test Sync Turn Test Drop Oldes|Test Sync Turn Test Drop Oldes]]
- [[_COMMUNITY_Test Sync Turn Test Sync Turn|Test Sync Turn Test Sync Turn ]]
- [[_COMMUNITY_Test Initialize Lifecycle Test|Test Initialize Lifecycle Test]]
- [[_COMMUNITY_Test Initialize Lifecycle Test|Test Initialize Lifecycle Test]]
- [[_COMMUNITY_Test Initialize Lifecycle Test|Test Initialize Lifecycle Test]]
- [[_COMMUNITY_Test E2E Install Lifecycle Tes|Test E2E Install Lifecycle Tes]]
- [[_COMMUNITY_Test Recall Test Prefetch Form|Test Recall Test Prefetch Form]]
- [[_COMMUNITY_Test Sync Worker Fake End Sess|Test Sync Worker Fake End Sess]]
- [[_COMMUNITY_Test Sync Worker Test Worker S|Test Sync Worker Test Worker S]]
- [[_COMMUNITY_Test Sync Worker Test Worker P|Test Sync Worker Test Worker P]]
- [[_COMMUNITY_Test Sync Worker Test Shutdown|Test Sync Worker Test Shutdown]]
- [[_COMMUNITY_Test Sync Worker Test Worker N|Test Sync Worker Test Worker N]]
- [[_COMMUNITY_Test Handle Tool Call Test Max|Test Handle Tool Call Test Max]]
- [[_COMMUNITY_Test Handle Tool Call Test Nod|Test Handle Tool Call Test Nod]]
- [[_COMMUNITY_Test Quality Metrics Test Last|Test Quality Metrics Test Last]]
- [[_COMMUNITY_Test Quality Metrics Test Perm|Test Quality Metrics Test Perm]]
- [[_COMMUNITY_Test Quality Metrics Test Rece|Test Quality Metrics Test Rece]]
- [[_COMMUNITY_Test Quality Metrics Test Empt|Test Quality Metrics Test Empt]]
- [[_COMMUNITY_Test Quality Metrics Test Hybr|Test Quality Metrics Test Hybr]]
- [[_COMMUNITY_Test Quality Metrics Test Perm|Test Quality Metrics Test Perm]]

## God Nodes (most connected - your core abstractions)
1. `CashewMemoryProvider` - 86 edges
2. `CashewConfig` - 69 edges
3. `resolve_db_path()` - 22 edges
4. `load_config()` - 22 edges
5. `MemoryProvider` - 13 edges
6. `MemoryManager` - 12 edges
7. `_make_initialized_provider()` - 12 edges
8. `_seed_node()` - 11 edges
9. `test_burst_worker_drains_50_distinct_turns_with_larger_queue()` - 10 edges
10. `test_burst_default_queue_drops_oldest_cleanly()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `resolve_db_path()` --conceptually_related_to--> `Profile Isolation`  [INFERRED]
  plugins/memory/cashew/config.py → AGENTS.md
- `resolve_db_path()` --conceptually_related_to--> `Profile Isolation`  [INFERRED]
  plugins/memory/cashew/config.py → CLAUDE.md
- `CashewMemoryProvider` --implements--> `Plugin Interface (Hermes ABC)`  [INFERRED]
  plugins/memory/cashew/__init__.py → CLAUDE.md
- `test_extract_bypasses_sync_queue()` --conceptually_related_to--> `Threading Rule`  [INFERRED]
  tests/test_cashew_extract.py → AGENTS.md
- `CONF-01 + CONFIG-07: schema returns ~31 field descriptors for hermes memory setu` --uses--> `CashewConfig`  [INFERRED]
  tests/test_config_roundtrip.py → plugins/memory/cashew/config.py

## Hyperedges (group relationships)
- **Cashew sync lifecycle** — cashew_init_cashewmemoryprovider_initialize, cashew_init_cashewmemoryprovider_sync_turn, cashew_init_cashewmemoryprovider_worker_loop, cashew_init_cashewmemoryprovider_on_session_end, cashew_init_cashewmemoryprovider_shutdown [INFERRED 0.80]
- **Three-tier retrieval pipeline** — cashew_init_cashewmemoryprovider_prefetch, cashew_init_cashewmemoryprovider_retrieve_with_vec, cashew_init_cashewmemoryprovider_retrieve_keyword, cashew_init_cashewmemoryprovider_score_nodes, cashew_init_cashewmemoryprovider_format_context [INFERRED 0.85]
- **Tool response envelope builders** — cashew_tools_build_success_envelope, cashew_tools_build_error_envelope, cashew_tools_build_extract_success_envelope, cashew_tools_build_extract_error_envelope [EXTRACTED 1.00]
- **Sync test helper copy-paste idiom** — test_sync_turn_drain_queue, test_sync_turn_wait_for_thread_exit, test_sync_turn_fake_end_session_ok, test_sync_turn_fake_end_session_slow, test_sync_turn_make_initialized_provider, test_sync_worker_drain_queue, test_sync_worker_wait_for_thread_exit, test_sync_worker_fake_end_session_ok, test_sync_worker_fake_end_session_slow, test_sync_worker_make_initialized_provider [INFERRED 0.95]
- **Cross-file seed node helper duplication** — test_retrieval_seed_node, test_memory_manager_e2e_seed_node, test_recall_seed_node, test_handle_tool_call_seed_node, test_quality_metrics_seed_node [INFERRED 0.95]
- **Silent degrade contract tests** — test_sync_turn_test_sync_turn_silent_noop_on_fresh_provider, test_sync_turn_test_sync_turn_silent_noop_on_corrupt_config_provider, test_sync_turn_test_sync_turn_silent_noop_after_shutdown, test_recall_test_prefetch_half_state_returns_empty_no_log, test_recall_test_prefetch_corrupt_config_half_state_returns_empty_no_new_log, test_handle_tool_call_test_half_state_returns_error_envelope_no_log, test_initialize_lifecycle_test_initialize_silent_degrades_on_corrupt_config, test_sync_worker_test_worker_not_started_on_corrupt_config [INFERRED 0.85]

## Communities

### Community 0 - "Configuration System"
Cohesion: 0.04
Nodes (79): CashewConfig, _env_var_name(), get_ai_domain(), get_config_schema(), get_user_domain(), load_config(), Pure config + path-resolution helpers for the Cashew memory provider.  This modu, Derive the CASHEW_* environment variable name for a config key.      Rule: strip (+71 more)

### Community 1 - "CashewMemoryProvider Core"
Cohesion: 0.04
Nodes (52): CashewMemoryProvider, Return the JSON-Schema-shaped dict Hermes uses to drive `hermes memory setup` (C, Background drain loop. Entry point for self._sync_worker.          Sentinel chec, Create or migrate Cashew schema tables.          Cashew's _ensure_schema() only, Migrate derivation_edges.timestamp to NOT NULL DEFAULT ''.          _create_edge, Cashew thought-graph memory provider for Hermes Agent., Persist one turn via Cashew's heuristic extractor.          Lazy-imports core.se, Best-effort bounded drain; does NOT stop the worker (ABC-06).          Polls unf (+44 more)

### Community 2 - "Test Fixtures & E2E Install"
Cohesion: 0.07
Nodes (17): ABC, HF_HUB_OFFLINE constraint, No ~/.hermes writes, fake_embedder(), home_snapshot(), MemoryProvider, Snapshot ~/.hermes mtime + recursive file listing before/after a test.      Used, Minimal stub of the Hermes MemoryProvider ABC for test isolation. (+9 more)

### Community 3 - "Tool Schema Validation"
Cohesion: 0.07
Nodes (25): Return the list of LLM tool schemas this provider exposes (RECALL-02 + SYNC-03)., Negative: max_nodes has an upper bound of 20 to cap retrieval load (T-03-02-03)., RECALL-02 + SYNC-03: get_tool_schemas exposes both tool schemas.      Phase 3 sh, Schema structural contract — OpenAI tool-schema shape., RECALL-02: description >= 50 chars., PHASE_DESIGN_NOTES Decision Point 6: description is the exact 03-RESEARCH.md §3, RECALL-02: `required` is explicitly the one-element list ["query"]., Defensive: unknown parameters are rejected at schema-validation layer (T-03-02-0 (+17 more)

### Community 4 - "Tool Call Handling"
Cohesion: 0.12
Nodes (28): Profile Isolation, Resolve the Cashew DB path under hermes_home.      `db_path_value` is the user-c, resolve_db_path(), Profile Isolation, _make_initialized_provider(), RECALL-03: name != cashew_query -> error envelope + one WARNING (no exc_info)., RECALL-04: pre-initialize -> error envelope, NO log (init-time WARNING is the au, RECALL-04: args={} -> KeyError inside main try block -> error envelope + one WAR (+20 more)

### Community 5 - "Sync Turn Queue"
Cohesion: 0.14
Nodes (25): drain_queue(), fake_end_session_ok(), fake_end_session_slow(), make_initialized_provider(), Enforces the ROADMAP's 10ms success criterion with 5ms headroom.      Under idea, Pitfall 5: sync_turn stays fast even when the queue is full (drop-oldest path)., Overflow produces WARNING with substring 'overflow'., Pitfall 3: drop-oldest overflow must not cause task_done() double-count ValueErr (+17 more)

### Community 6 - "Sync Worker Thread"
Cohesion: 0.16
Nodes (23): drain_queue(), fake_end_session_ok(), fake_end_session_raises(), fake_end_session_slow(), make_initialized_provider(), 0.3s sleep gives the shutdown timeout (0.1s) time to fire while     guaranteeing, Poll unfinished_tasks until 0 or budget expires. Returns True iff drained., Poll threading.active_count until it equals baseline or budget expires. (+15 more)

### Community 7 - "Memory Manager Stub"
Cohesion: 0.11
Nodes (13): inject_into_sys_modules(), MemoryManager, Minimal stand-in for Hermes's agent.memory_manager.MemoryManager.      Implement, Route tool call to the first provider that declares `name` in its schemas., Route a completed turn through each registered provider's sync_turn.          Ph, Populate sys.modules['agent.memory_manager'] so `from agent.memory_manager     i, TEST-01 extended: MemoryManager.sync_all exercises the Phase 4 write path., TEST-01: full MemoryManager lifecycle — add_provider -> initialize_all -> handle (+5 more)

### Community 8 - "Database Migration"
Cohesion: 0.16
Nodes (17): _get_columns(), _make_v0_1_0_db(), SCHEMA-06: Migration is strictly additive — no columns are dropped., D-06 / SCHEMA-03: Existing rows receive DEFAULT backfill for metadata., Create a v0.1.0 schema DB (missing v0.2.0 columns)., SCHEMA-04: When sqlite-vec is unavailable, initialize() succeeds and logs INFO., SCHEMA-06: All existing row data is preserved after migration., Return set of column names for a table. (+9 more)

### Community 9 - "Recall & Prefetch"
Cohesion: 0.16
Nodes (18): _make_initialized_provider(), RECALL-04: retrieval raises -> "" + ONE WARNING with exc_info., RECALL-04: _format_context() raises -> same contract as retrieve() raising., Empty graph (no matching nodes) is not a failure — empty string, no log., Construct a provider, save a minimal config, and initialize it., Insert a row into thought_nodes with sensible defaults., RECALL-01: retrieve + format_context produces a non-empty recalled context strin, RECALL-01: recall_k from config caps the number of returned nodes. (+10 more)

### Community 10 - "Sync Burst & Threading"
Cohesion: 0.18
Nodes (13): sync_turn <10 ms, Threading Rule, Wire the provider to a hermes_home (ABC-04).          Reads kwargs["hermes_home", Launch the non-daemon worker. Called from initialize() only on happy path., Hot-path enqueue of a completed turn (SYNC-01).          Contract: returns in <1, Post sentinel, bounded-join worker, clear references (ABC-05 + SYNC-05)., _drain_queue(), _fake_end_session_ok() (+5 more)

### Community 11 - "Retrieval & Context"
Cohesion: 0.16
Nodes (14): db_path(), provider(), Simulate macOS where sqlite-vec extension loading is unavailable — provider fall, Create an initialized provider with a fresh DB., Return the resolved DB path for direct sqlite3 access., Insert a row into thought_nodes with sensible defaults., _seed_node(), test_format_context_with_domain_and_type() (+6 more)

### Community 12 - "Cashew Extract"
Cohesion: 0.16
Nodes (9): Persist the four-key Cashew config to $HERMES_HOME/cashew.json (CONF-02)., make_initialized_provider(), test_extract_bypasses_sync_queue(), test_extract_calls_end_session_with_canonical_kwargs(), test_extract_cashew_failure_returns_error_envelope_and_logs_once(), test_extract_error_envelope_has_no_stack_trace_substrings(), test_extract_half_state_returns_error_envelope_no_log(), test_extract_happy_path_returns_success_envelope() (+1 more)

### Community 13 - "Test Helper Cross-Refs"
Cohesion: 0.12
Nodes (16): test_e2e_install_lifecycle_full_provider_init_and_shutdown, _seed_node helper, test_error_envelope_has_no_stack_trace_substrings, test_happy_path_returns_valid_success_envelope, test_happy_path_uses_recall_k_when_max_nodes_omitted, _seed_node helper, test_full_lifecycle_with_sync_path, test_memory_manager_e2e_lifecycle (+8 more)

### Community 14 - "Plugin Registration"
Cohesion: 0.16
Nodes (11): Hermes discovery entry point — filesystem loader calls this., register(), build_error_envelope(), build_extract_error_envelope(), build_extract_success_envelope(), build_success_envelope(), Tool-surface helpers for the Cashew memory provider.  This module owns:   - CASH, Return a json.dumps(...) string wrapping a cashew_query error.      Args: (+3 more)

### Community 15 - "Quality Metrics"
Cohesion: 0.25
Nodes (13): db_path(), provider(), _seed_node(), test_access_count_incremented_after_retrieval(), test_domain_filtering(), test_empty_filter_returns_all_nodes(), test_hybrid_scoring_includes_confidence(), test_last_accessed_updated_after_retrieval() (+5 more)

### Community 16 - "Plugin Discovery"
Cohesion: 0.29
Nodes (6): PKG-02: pyproject.toml declares [project.entry-points."hermes_agent.plugins"]., The entry-point target (register function) must be loadable., PKG-01: plugins.memory.cashew is importable from the installed package., test_entry_point_loads(), test_entry_point_registered(), test_module_importable()

### Community 17 - "Verify Script"
Cohesion: 0.4
Nodes (5): _error(), main(), Smoke test for hermes-cashew.  Exercises the full CashewMemoryProvider lifecycle, Print a cashew-prefixed error and exit with code 1., Run the smoke test.      Creates a temp hermes_home, initializes the provider, r

### Community 18 - "Profile Isolation"
Cohesion: 0.4
Nodes (4): CONF-04: every path scopes under hermes_home — no Path.home, expanduser, or HOME, Strip # line comments and standalone string-expression statements (docstrings)., _strip_comments_and_docstrings(), test_no_home_related_patterns_under_plugins()

### Community 19 - "Config Migration Bridge"
Cohesion: 0.4
Nodes (5): test_load_config_v0_1_0_four_key_file_gets_defaults_for_new_keys, _make_v0_1_0_db helper, test_metadata_backfill, test_migration_idempotent, test_v0_1_0_columns_added

### Community 20 - "Sync Provider Helpers"
Cohesion: 0.4
Nodes (5): make_initialized_provider helper, test_sync_turn_empty_queue_strict_15ms, test_sync_turn_fast_on_empty_queue, make_initialized_provider helper, test_worker_processes_single_turn

### Community 21 - "Flat Entrypoint Tests"
Cohesion: 0.67
Nodes (2): Simulates hermes plugins install dropping the repo at $HERMES_HOME/plugins/cashe, test_root_init_reexports_register_and_provider()

### Community 22 - "Home Leak & Profile Isolation"
Cohesion: 0.67
Nodes (3): test_full_lifecycle_does_not_touch_home, _strip_comments_and_docstrings helper, test_no_home_related_patterns_under_plugins

### Community 23 - "Corrupt Config Handling"
Cohesion: 0.67
Nodes (3): test_initialize_silent_degrades_on_corrupt_config, test_prefetch_corrupt_config_half_state_returns_empty_no_new_log, test_sync_turn_silent_noop_on_corrupt_config_provider

### Community 24 - "Half-State Error Handling"
Cohesion: 0.67
Nodes (3): test_half_state_returns_error_envelope_no_log, test_prefetch_half_state_returns_empty_no_log, test_sync_turn_silent_noop_on_fresh_provider

### Community 25 - "Dual Layout & Path Loading"
Cohesion: 1.0
Nodes (2): Dual layout, Dual-path loading strategy

### Community 26 - "Availability Tests"
Cohesion: 1.0
Nodes (2): test_is_available_calls_path_exists_exactly_once, test_is_available_no_filesystem_calls

### Community 27 - "Recency Weighting"
Cohesion: 1.0
Nodes (2): test_recency_weighting_uses_referent_time, test_retrieve_keyword_orders_by_referent_time

### Community 28 - "Config Defaults Validation"
Cohesion: 1.0
Nodes (2): test_cashew_config_dataclass_field_set_matches_defaults, test_defaults_contains_exactly_31_keys_with_documented_values

### Community 29 - "Vec Embeddings Guard"
Cohesion: 1.0
Nodes (2): test_vec_embeddings_guarded_when_unavailable, test_macos_fallback_simulated

### Community 30 - "Thread Exit Helpers"
Cohesion: 1.0
Nodes (2): wait_for_thread_exit helper, wait_for_thread_exit helper

### Community 31 - "Slow Session Simulations"
Cohesion: 1.0
Nodes (2): fake_end_session_slow helper, fake_end_session_slow helper

### Community 32 - "Queue Bounds Tests"
Cohesion: 1.0
Nodes (2): test_initialize_creates_bounded_queue, test_sync_turn_fast_even_on_full_queue

### Community 33 - "Worker Resilience"
Cohesion: 1.0
Nodes (2): test_drop_oldest_logs_warning_and_preserves_newest, test_poisoned_turn_does_not_break_worker

### Community 34 - "Lazy Import Tests"
Cohesion: 1.0
Nodes (2): test_initialize_does_not_import_cashew_runtime, test_lazy_import_preserves_module_loadability

### Community 35 - "Initialized Provider Helpers"
Cohesion: 1.0
Nodes (2): _make_initialized_provider helper, _make_initialized_provider helper

### Community 36 - "Retrieval Exception Handling"
Cohesion: 1.0
Nodes (2): test_retrieval_exception_returns_error_envelope_and_logs_once, test_prefetch_retrieval_exception_logs_once_and_returns_empty

### Community 37 - "Drain Queue Helpers"
Cohesion: 1.0
Nodes (2): drain_queue helper, drain_queue helper

### Community 38 - "OK Session Simulations"
Cohesion: 1.0
Nodes (2): fake_end_session_ok helper, fake_end_session_ok helper

### Community 39 - "Worker Lifecycle"
Cohesion: 1.0
Nodes (2): test_on_session_end_polls_and_returns, test_shutdown_posts_sentinel_and_joins

### Community 40 - "Pre-Initialize Safety"
Cohesion: 1.0
Nodes (2): test_shutdown_pre_initialize_is_safe_noop, test_pre_initialize_shutdown_still_safe_noop

### Community 41 - "Unknown Tool Routing"
Cohesion: 1.0
Nodes (2): test_unknown_tool_returns_error_envelope_and_logs_once, test_memory_manager_routes_unknown_tool_to_none

### Community 42 - "Filtering Tests"
Cohesion: 1.0
Nodes (2): test_domain_filtering, test_tag_filtering

### Community 46 - "Test Flat Entrypoint Reexports"
Cohesion: 1.0
Nodes (1): test_root_init_reexports_register_and_provider

### Community 47 - "Test Stub Lifecycle Test Name "
Cohesion: 1.0
Nodes (1): test_name_is_cashew

### Community 48 - "Test Stub Lifecycle Test Is Av"
Cohesion: 1.0
Nodes (1): test_is_available_false_before_config

### Community 49 - "Test Retrieval Provider"
Cohesion: 1.0
Nodes (1): provider fixture

### Community 50 - "Test Retrieval Db Path"
Cohesion: 1.0
Nodes (1): db_path fixture

### Community 51 - "Test Retrieval Test Prefetch E"
Cohesion: 1.0
Nodes (1): test_prefetch_empty_graph_returns_empty_string

### Community 52 - "Test Retrieval Test Prefetch E"
Cohesion: 1.0
Nodes (1): test_prefetch_empty_graph_no_error

### Community 53 - "Test Retrieval Test Retrieve K"
Cohesion: 1.0
Nodes (1): test_retrieve_keyword_respects_max_nodes

### Community 54 - "Test Retrieval Test Format Con"
Cohesion: 1.0
Nodes (1): test_format_context_with_domain_and_type

### Community 55 - "Test Retrieval Test Format Con"
Cohesion: 1.0
Nodes (1): test_format_context_without_domain_or_type

### Community 56 - "Test No Home Leak Test Save Co"
Cohesion: 1.0
Nodes (1): test_save_config_writes_only_under_tmp_path

### Community 57 - "Test Config Roundtrip Test Con"
Cohesion: 1.0
Nodes (1): test_config_filename_is_cashew_json

### Community 58 - "Test Config Roundtrip Test Get"
Cohesion: 1.0
Nodes (1): test_get_config_schema_shape_is_list_of_field_descriptors

### Community 59 - "Test Config Roundtrip Test Env"
Cohesion: 1.0
Nodes (1): test_env_var_map_has_31_entries

### Community 60 - "Test Config Roundtrip Test Res"
Cohesion: 1.0
Nodes (1): test_resolve_db_path_rejects_absolute_paths

### Community 61 - "Test Config Roundtrip Test Loa"
Cohesion: 1.0
Nodes (1): test_load_config_env_overrides_file

### Community 62 - "Test Config Roundtrip Test Sav"
Cohesion: 1.0
Nodes (1): test_save_config_roundtrip

### Community 63 - "Test Plugin Discovery Test Ent"
Cohesion: 1.0
Nodes (1): test_entry_point_registered

### Community 64 - "Test Plugin Discovery Test Ent"
Cohesion: 1.0
Nodes (1): test_entry_point_loads

### Community 65 - "Test Plugin Discovery Test Mod"
Cohesion: 1.0
Nodes (1): test_module_importable

### Community 66 - "Test Migration Test No Columns"
Cohesion: 1.0
Nodes (1): test_no_columns_dropped

### Community 67 - "Test Migration Test Existing D"
Cohesion: 1.0
Nodes (1): test_existing_data_preserved

### Community 68 - "Test Migration Test Fresh Db H"
Cohesion: 1.0
Nodes (1): test_fresh_db_has_all_columns

### Community 69 - "Test Profile Isolation Forbidd"
Cohesion: 1.0
Nodes (1): FORBIDDEN_PATTERNS constant

### Community 70 - "Test Sync Turn Test Drop Oldes"
Cohesion: 1.0
Nodes (1): test_drop_oldest_balances_task_done

### Community 71 - "Test Sync Turn Test Sync Turn "
Cohesion: 1.0
Nodes (1): test_sync_turn_silent_noop_after_shutdown

### Community 72 - "Test Initialize Lifecycle Test"
Cohesion: 1.0
Nodes (1): test_fresh_provider_is_not_available

### Community 73 - "Test Initialize Lifecycle Test"
Cohesion: 1.0
Nodes (1): test_initialize_then_shutdown_returns_to_baseline_threads

### Community 74 - "Test Initialize Lifecycle Test"
Cohesion: 1.0
Nodes (1): test_initialize_succeeds_on_fresh_hermes_home

### Community 75 - "Test E2E Install Lifecycle Tes"
Cohesion: 1.0
Nodes (1): test_e2e_install_lifecycle_saves_config_and_reports_available

### Community 76 - "Test Recall Test Prefetch Form"
Cohesion: 1.0
Nodes (1): test_prefetch_format_context_exception_logs_once_and_returns_empty

### Community 77 - "Test Sync Worker Fake End Sess"
Cohesion: 1.0
Nodes (1): fake_end_session_raises helper

### Community 78 - "Test Sync Worker Test Worker S"
Cohesion: 1.0
Nodes (1): test_worker_starts_after_initialize_on_happy_path

### Community 79 - "Test Sync Worker Test Worker P"
Cohesion: 1.0
Nodes (1): test_worker_processes_multiple_turns_fifo

### Community 80 - "Test Sync Worker Test Shutdown"
Cohesion: 1.0
Nodes (1): test_shutdown_hung_worker_logs_warning_no_raise

### Community 81 - "Test Sync Worker Test Worker N"
Cohesion: 1.0
Nodes (1): test_worker_not_started_on_corrupt_config

### Community 82 - "Test Handle Tool Call Test Max"
Cohesion: 1.0
Nodes (1): test_max_nodes_override_wins_over_recall_k

### Community 83 - "Test Handle Tool Call Test Nod"
Cohesion: 1.0
Nodes (1): test_node_count_matches_retrieved_length

### Community 84 - "Test Quality Metrics Test Last"
Cohesion: 1.0
Nodes (1): test_last_accessed_updated_after_retrieval

### Community 85 - "Test Quality Metrics Test Perm"
Cohesion: 1.0
Nodes (1): test_permanent_nodes_ranked_higher

### Community 86 - "Test Quality Metrics Test Rece"
Cohesion: 1.0
Nodes (1): test_recency_fallback_to_timestamp

### Community 87 - "Test Quality Metrics Test Empt"
Cohesion: 1.0
Nodes (1): test_empty_filter_returns_all_nodes

### Community 88 - "Test Quality Metrics Test Hybr"
Cohesion: 1.0
Nodes (1): test_hybrid_scoring_includes_confidence

### Community 89 - "Test Quality Metrics Test Perm"
Cohesion: 1.0
Nodes (1): test_permanent_flag_visible_in_context

## Knowledge Gaps
- **214 isolated node(s):** `Pure config + path-resolution helpers for the Cashew memory provider.  This modu`, `Typed view over the Cashew config dict (~31 keys aligned with upstream).`, `Derive the CASHEW_* environment variable name for a config key.      Rule: strip`, `Return the configured user domain label (replaces hardcoded 'user').`, `Return the configured AI domain label (replaces hardcoded 'ai').` (+209 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Flat Entrypoint Tests`** (3 nodes): `test_flat_entrypoint_reexports.py`, `Simulates hermes plugins install dropping the repo at $HERMES_HOME/plugins/cashe`, `test_root_init_reexports_register_and_provider()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Dual Layout & Path Loading`** (2 nodes): `Dual layout`, `Dual-path loading strategy`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Availability Tests`** (2 nodes): `test_is_available_calls_path_exists_exactly_once`, `test_is_available_no_filesystem_calls`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Recency Weighting`** (2 nodes): `test_recency_weighting_uses_referent_time`, `test_retrieve_keyword_orders_by_referent_time`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Config Defaults Validation`** (2 nodes): `test_cashew_config_dataclass_field_set_matches_defaults`, `test_defaults_contains_exactly_31_keys_with_documented_values`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Vec Embeddings Guard`** (2 nodes): `test_vec_embeddings_guarded_when_unavailable`, `test_macos_fallback_simulated`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Thread Exit Helpers`** (2 nodes): `wait_for_thread_exit helper`, `wait_for_thread_exit helper`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Slow Session Simulations`** (2 nodes): `fake_end_session_slow helper`, `fake_end_session_slow helper`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Queue Bounds Tests`** (2 nodes): `test_initialize_creates_bounded_queue`, `test_sync_turn_fast_even_on_full_queue`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Worker Resilience`** (2 nodes): `test_drop_oldest_logs_warning_and_preserves_newest`, `test_poisoned_turn_does_not_break_worker`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Lazy Import Tests`** (2 nodes): `test_initialize_does_not_import_cashew_runtime`, `test_lazy_import_preserves_module_loadability`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Initialized Provider Helpers`** (2 nodes): `_make_initialized_provider helper`, `_make_initialized_provider helper`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Retrieval Exception Handling`** (2 nodes): `test_retrieval_exception_returns_error_envelope_and_logs_once`, `test_prefetch_retrieval_exception_logs_once_and_returns_empty`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Drain Queue Helpers`** (2 nodes): `drain_queue helper`, `drain_queue helper`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `OK Session Simulations`** (2 nodes): `fake_end_session_ok helper`, `fake_end_session_ok helper`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Worker Lifecycle`** (2 nodes): `test_on_session_end_polls_and_returns`, `test_shutdown_posts_sentinel_and_joins`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pre-Initialize Safety`** (2 nodes): `test_shutdown_pre_initialize_is_safe_noop`, `test_pre_initialize_shutdown_still_safe_noop`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Unknown Tool Routing`** (2 nodes): `test_unknown_tool_returns_error_envelope_and_logs_once`, `test_memory_manager_routes_unknown_tool_to_none`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Filtering Tests`** (2 nodes): `test_domain_filtering`, `test_tag_filtering`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Flat Entrypoint Reexports`** (1 nodes): `test_root_init_reexports_register_and_provider`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Stub Lifecycle Test Name `** (1 nodes): `test_name_is_cashew`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Stub Lifecycle Test Is Av`** (1 nodes): `test_is_available_false_before_config`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Retrieval Provider`** (1 nodes): `provider fixture`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Retrieval Db Path`** (1 nodes): `db_path fixture`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Retrieval Test Prefetch E`** (1 nodes): `test_prefetch_empty_graph_returns_empty_string`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Retrieval Test Prefetch E`** (1 nodes): `test_prefetch_empty_graph_no_error`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Retrieval Test Retrieve K`** (1 nodes): `test_retrieve_keyword_respects_max_nodes`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Retrieval Test Format Con`** (1 nodes): `test_format_context_with_domain_and_type`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Retrieval Test Format Con`** (1 nodes): `test_format_context_without_domain_or_type`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test No Home Leak Test Save Co`** (1 nodes): `test_save_config_writes_only_under_tmp_path`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Config Roundtrip Test Con`** (1 nodes): `test_config_filename_is_cashew_json`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Config Roundtrip Test Get`** (1 nodes): `test_get_config_schema_shape_is_list_of_field_descriptors`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Config Roundtrip Test Env`** (1 nodes): `test_env_var_map_has_31_entries`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Config Roundtrip Test Res`** (1 nodes): `test_resolve_db_path_rejects_absolute_paths`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Config Roundtrip Test Loa`** (1 nodes): `test_load_config_env_overrides_file`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Config Roundtrip Test Sav`** (1 nodes): `test_save_config_roundtrip`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Plugin Discovery Test Ent`** (1 nodes): `test_entry_point_registered`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Plugin Discovery Test Ent`** (1 nodes): `test_entry_point_loads`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Plugin Discovery Test Mod`** (1 nodes): `test_module_importable`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Migration Test No Columns`** (1 nodes): `test_no_columns_dropped`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Migration Test Existing D`** (1 nodes): `test_existing_data_preserved`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Migration Test Fresh Db H`** (1 nodes): `test_fresh_db_has_all_columns`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Profile Isolation Forbidd`** (1 nodes): `FORBIDDEN_PATTERNS constant`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Sync Turn Test Drop Oldes`** (1 nodes): `test_drop_oldest_balances_task_done`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Sync Turn Test Sync Turn `** (1 nodes): `test_sync_turn_silent_noop_after_shutdown`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Initialize Lifecycle Test`** (1 nodes): `test_fresh_provider_is_not_available`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Initialize Lifecycle Test`** (1 nodes): `test_initialize_then_shutdown_returns_to_baseline_threads`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Initialize Lifecycle Test`** (1 nodes): `test_initialize_succeeds_on_fresh_hermes_home`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test E2E Install Lifecycle Tes`** (1 nodes): `test_e2e_install_lifecycle_saves_config_and_reports_available`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Recall Test Prefetch Form`** (1 nodes): `test_prefetch_format_context_exception_logs_once_and_returns_empty`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Sync Worker Fake End Sess`** (1 nodes): `fake_end_session_raises helper`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Sync Worker Test Worker S`** (1 nodes): `test_worker_starts_after_initialize_on_happy_path`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Sync Worker Test Worker P`** (1 nodes): `test_worker_processes_multiple_turns_fifo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Sync Worker Test Shutdown`** (1 nodes): `test_shutdown_hung_worker_logs_warning_no_raise`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Sync Worker Test Worker N`** (1 nodes): `test_worker_not_started_on_corrupt_config`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Handle Tool Call Test Max`** (1 nodes): `test_max_nodes_override_wins_over_recall_k`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Handle Tool Call Test Nod`** (1 nodes): `test_node_count_matches_retrieved_length`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Quality Metrics Test Last`** (1 nodes): `test_last_accessed_updated_after_retrieval`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Quality Metrics Test Perm`** (1 nodes): `test_permanent_nodes_ranked_higher`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Quality Metrics Test Rece`** (1 nodes): `test_recency_fallback_to_timestamp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Quality Metrics Test Empt`** (1 nodes): `test_empty_filter_returns_all_nodes`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Quality Metrics Test Hybr`** (1 nodes): `test_hybrid_scoring_includes_confidence`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Quality Metrics Test Perm`** (1 nodes): `test_permanent_flag_visible_in_context`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CashewMemoryProvider` connect `CashewMemoryProvider Core` to `Configuration System`, `Test Fixtures & E2E Install`, `Tool Schema Validation`, `Tool Call Handling`, `Sync Turn Queue`, `Sync Worker Thread`, `Memory Manager Stub`, `Database Migration`, `Recall & Prefetch`, `Sync Burst & Threading`, `Retrieval & Context`, `Cashew Extract`, `Plugin Registration`, `Quality Metrics`, `Verify Script`?**
  _High betweenness centrality (0.455) - this node is a cross-community bridge._
- **Why does `CashewConfig` connect `Configuration System` to `CashewMemoryProvider Core`, `Test Fixtures & E2E Install`, `Tool Schema Validation`, `Sync Burst & Threading`, `Cashew Extract`, `Plugin Registration`?**
  _High betweenness centrality (0.160) - this node is a cross-community bridge._
- **Why does `test_provider_get_tool_schemas_returns_single_cashew_query_schema()` connect `Tool Schema Validation` to `CashewMemoryProvider Core`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Are the 54 inferred relationships involving `CashewMemoryProvider` (e.g. with `CashewConfig` and `main()`) actually correct?**
  _`CashewMemoryProvider` has 54 INFERRED edges - model-reasoned connections that need verification._
- **Are the 66 inferred relationships involving `CashewConfig` (e.g. with `CashewMemoryProvider` and `Cashew thought-graph memory provider for Hermes Agent.`) actually correct?**
  _`CashewConfig` has 66 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `resolve_db_path()` (e.g. with `.initialize()` and `db_path()`) actually correct?**
  _`resolve_db_path()` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `load_config()` (e.g. with `.initialize()` and `test_load_config_returns_defaults_when_file_absent()`) actually correct?**
  _`load_config()` has 17 INFERRED edges - model-reasoned connections that need verification._