# Technology Stack

**Project:** hermes-cashew (v0.2.0 Retrieval Core)  
**Researched:** 2026-04-21  
**Scope:** Stack additions for sqlite-vec embedding search, automatic schema migration, and expanded config (~30+ keys) with sane defaults.

## Recommended Stack

### Core Framework
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python | >=3.10 | Runtime | Existing constraint; no change needed. |
| cashew-brain | >=1.0.0,<2.0.0 | Backing store | Existing PyPI dep; upstream already contains `sqlite-vec` retrieval path in `core/retrieval.py` and `core/session.py`. |
| sqlite-vec | >=0.1.9,<0.2.0 | SQLite vector-search extension | Provides `O(log N)` ANN via `vec0` virtual tables + `vec_distance_cosine()`. Zero-dependency C extension; wheels are ~130–290 KB. Python API is `sqlite_vec.load(conn)`. |
| sqlite3 (stdlib) | — | SQLite driver | Existing; `enable_load_extension(True)` required before `sqlite_vec.load()`. |

### Database / Migration
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| sqlite3 + PRAGMA | stdlib | Schema introspection | `PRAGMA table_info(thought_nodes)` to detect missing columns. No ORM or migration framework needed. |
| `ALTER TABLE … ADD COLUMN` | SQL | Additive migration | Matches upstream Cashew pattern (`core/session.py::_ensure_schema`). SQLite supports additive ALTER natively; backward-compatible with v0.1.0 DBs. |

### Config
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `dataclasses` | stdlib | Typed config object | Existing pattern (`CashewConfig`). Expand to ~30 fields; flat JSON stays compatible with `hermes memory setup`. |
| `json` | stdlib | Config serialization | Existing pattern (`cashew.json`). No YAML runtime parser needed — defaults are hard-coded in Python to match upstream `config.yaml.template`. |
| `os.environ` | stdlib | Env-var overrides | Milestone requires env overrides; zero-dependency. |

### Supporting Libraries
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| PyYAML | >=6.0 | Dev/test only | Already in `[dev]`; used to validate upstream YAML config parity in tests. **Do NOT add to runtime deps** — keep config JSON-only to stay aligned with Hermes setup contract. |
| jsonschema | >=4.18,<5 | Dev/test only | Already in `[dev]`; test-time validation of tool schemas. Not needed at runtime. |
| numpy | optional | Float32 vector buffers | sqlite-vec accepts NumPy arrays via buffer protocol, but also provides `serialize_float32()` for plain Python lists. **Avoid runtime numpy dep** — use `sqlite_vec.serialize_float32()` to keep install footprint small. |

## What Changes in `pyproject.toml`

### Runtime dependency addition
```toml
dependencies = [
    "cashew-brain>=1.0.0,<2.0.0",
    "sqlite-vec>=0.1.9,<0.2.0",
]
```

### Dev dependencies (no change required)
```toml
dev = [
    "pytest>=8.2,<10",
    "pytest-asyncio>=1.3",
    "pytest-mock>=3.15",
    "jsonschema>=4.18,<5",
    "PyYAML>=6.0",
    "build>=1.0",
]
```

## Detailed Rationale

### sqlite-vec (not sqlite-vss, not chromadb, not faiss)
- **Why sqlite-vec:** Upstream Cashew already uses it (`core/retrieval.py` imports `sqlite_vec`, creates `vec0` virtual tables, calls `vec_distance_cosine`). The plugin must ensure the extension is present so upstream can load it.
- **Version 0.1.9:** Current stable release (2026-03-31). Pre-1.0, but API is stable enough for `load()` + `serialize_float32()` + `vec_distance_cosine()`. Pin `<0.2.0` to avoid breaking changes.
- **Platform coverage:** Wheels for macOS x86_64/arm64, Linux x86_64/aarch64, Windows amd64. Satisfies existing v0.1.0 targets (macOS + Linux; Windows is out-of-scope but works anyway).
- **Size:** ~160 KB per wheel. No impact on CI time or wheel size.
- **HF_HUB_OFFLINE safety:** sqlite-vec is a pure C extension; it does not download models. Embedding generation still happens via upstream Cashew (sentence-transformers), which remains mocked in tests via `fake_embedder`.

### No Alembic / SQLAlchemy / ORM for migrations
- **Why not:** The plugin’s DB surface is tiny (one SQLite file, one primary table, additive changes only). Alembic adds ~2 MB + SQLAlchemy (~10 MB) and requires migration script directories, versioning tables, and `alembic.ini` — all overkill.
- **What instead:** Replicate upstream’s `_ensure_schema()` pattern:
  1. `PRAGMA table_info(thought_nodes)` → list current columns.
  2. For each required column (`reasoning`, `mood_state`, `metadata`, `last_updated`, `last_accessed`, `access_count`, `permanent`, `referent_time`, etc.), if missing: `ALTER TABLE thought_nodes ADD COLUMN …`.
  3. `CREATE INDEX IF NOT EXISTS …` for new indexes.
  4. Wrap in a transaction.
- **Backward compatibility:** v0.1.0 DBs already have `thought_nodes` with a subset of columns. Additive ALTER is safe and idempotent.

### Config: flat JSON, not YAML
- **Why keep JSON:** Hermes `memory setup` writes JSON. Changing to YAML would break the Hermes integration contract.
- **How to map upstream YAML:** Flatten nested keys with underscore separators (e.g., `performance_token_budget`, `gc_mode`, `domains_default`). This matches the existing 4-key flat style (`cashew_db_path`, `sync_queue_timeout`).
- **Sane defaults:** Hard-code defaults in Python (`DEFAULTS` dict) to match `config.yaml.template` and `config.example.yaml` from upstream. Example:
  - `performance_token_budget: 2000`
  - `performance_top_k_results: 10`
  - `performance_walk_depth: 2`
  - `gc_mode: soft`
  - `gc_threshold: 0.05`
  - `features_auto_extraction: true`
  - `features_think_cycles: true`
  - etc.
- **Env var overrides:** Load `CASHEW_*` env vars at `load_config()` time, merge over JSON values. E.g., `CASHEW_PERFORMANCE_TOKEN_BUDGET=4000` overrides the file value.

## Integration with Existing cashew-brain

| Concern | Approach |
|---------|----------|
| **Extension loading** | Plugin does not need to load sqlite-vec itself in most cases — upstream `core/session.py` and `core/retrieval.py` handle `sqlite_vec.load(conn)`. The plugin must only ensure `sqlite-vec` is installed in the same Python environment. |
| **Retrieval path** | Replace the current `ContextRetriever.retrieve(query, max_nodes=…)` call with the upstream `retrieve_recursive_bfs()` (or equivalent) if available in `cashew-brain>=1.0.0`. If upstream does not yet expose it as a stable API, implement a thin retrieval wrapper in the plugin that uses `sqlite_vec` + BFS traversal against the same DB. |
| **Schema alignment** | Plugin’s `_ensure_schema()` must create/maintain all columns that upstream `core/session.py::_ensure_schema()` expects, so upstream SQL queries never fail with "no such column". |
| **Silent degrade** | If `sqlite-vec` fails to load (e.g., missing wheel for platform), fall back to the existing keyword-extraction retrieval path. Do not crash the agent. |

## What NOT to Add

| Technology | Why Rejected |
|------------|--------------|
| **Alembic / SQLAlchemy** | Massive overhead for additive-only SQLite migrations. Raw SQL + `PRAGMA` is sufficient and matches upstream. |
| **numpy** | sqlite-vec provides `serialize_float32()` for plain Python lists. Avoid ~15 MB runtime dependency. |
| **pydantic** | Config validation can be done with `dataclasses` + manual coercion. Adding pydantic (~30 MB with compiled deps) violates the lightweight plugin contract. |
| **YAML parser (runtime)** | Hermes writes/reads JSON. YAML is test-only (PyYAML already in dev deps). |
| **sentence-transformers** | Embedding model is managed by upstream `cashew-brain`, not the plugin. Plugin must never import it directly. CI `HF_HUB_OFFLINE=1` must remain effective. |
| **chromadb / faiss / pgvector** | Not SQLite-local; incompatible with the single-file, offline-first design. |

## Installation

```bash
# Core (what users get with `pip install hermes-cashew`)
pip install cashew-brain>=1.0.0,<2.0.0 sqlite-vec>=0.1.9,<0.2.0

# Dev (existing)
pip install -e ".[dev]"
```

## Sources

- sqlite-vec PyPI (version 0.1.9, release date 2026-03-31): https://pypi.org/project/sqlite-vec/  
- sqlite-vec Python docs (`sqlite_vec.load()`, `serialize_float32()`): https://alexgarcia.xyz/sqlite-vec/python.html  
- sqlite-vec API reference (`vec_distance_cosine`, `vec0` virtual tables): https://alexgarcia.xyz/sqlite-vec/api-reference.html  
- Cashew upstream `core/session.py` additive migration pattern (`PRAGMA table_info` + `ALTER TABLE ADD COLUMN`): https://github.com/rajkripal/cashew/blob/main/core/session.py  
- Cashew upstream `core/retrieval.py` sqlite-vec integration: https://github.com/rajkripal/cashew/blob/main/core/retrieval.py  
- Cashew upstream `config.yaml.template` (~30 keys): https://github.com/rajkripal/cashew/blob/main/config.yaml.template  
- Cashew upstream `config.example.yaml`: https://github.com/rajkripal/cashew/blob/main/config.example.yaml  
- hermes-cashew current `pyproject.toml` (v0.1.5): `/Volumes/tank01/magnus/git/hermes-cashew/pyproject.toml`  
- hermes-cashew current `config.py` (4-key flat JSON): `/Volumes/tank01/magnus/git/hermes-cashew/plugins/memory/cashew/config.py`
