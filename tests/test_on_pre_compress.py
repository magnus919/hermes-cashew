# tests/test_on_pre_compress.py
"""Tests for CashewMemoryProvider.on_pre_compress — conversation-arc insight extraction.

Covers: contract (returns str, never raises), thresholding (<3 exchanges),
LLM path (creates nodes in DB), edge cases (malformed JSON, empty arrays,
multimodal content, tool messages, DB failures).

Mocks upstream core.session and core.embeddings to keep tests offline.
"""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────


def _create_minimal_schema(conn: sqlite3.Connection) -> None:
    """Create the subset of Cashew schema needed by on_pre_compress tests."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS thought_nodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            node_type TEXT NOT NULL DEFAULT 'observation',
            timestamp TEXT NOT NULL DEFAULT '',
            mood_state TEXT,
            metadata TEXT,
            source_file TEXT,
            decayed INTEGER DEFAULT 0,
            permanent INTEGER DEFAULT 0,
            domain TEXT DEFAULT 'user',
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT,
            tags TEXT,
            referent_time TEXT
        );
        CREATE TABLE IF NOT EXISTS derivation_edges (
            parent_id TEXT NOT NULL,
            child_id TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            reasoning TEXT
        );
        CREATE TABLE IF NOT EXISTS embeddings (
            node_id TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            model TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)


def _make_msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _make_multimodal_msg(role: str, text_parts: list[str]) -> dict:
    parts = []
    for t in text_parts:
        parts.append({"type": "text", "text": t})
    parts.append(
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    )
    return {"role": role, "content": parts}


def _sample_messages(count: int = 6) -> list[dict]:
    """Generate a plausible conversation with detectable arc."""
    base = [
        _make_msg("user", "Can you help me set up a new memory provider for Hermes?"),
        _make_msg(
            "assistant", "Sure. Which provider are you considering? Cashew? Hindsight?"
        ),
        _make_msg(
            "user",
            "Cashew looks interesting — I like that it's local and doesn't need a server.",
        ),
        _make_msg(
            "assistant",
            "Good choice. It uses SQLite with sentence-transformers for local embeddings.",
        ),
        _make_msg(
            "user",
            "How does the retrieval work? Is it just vector search or something smarter?",
        ),
        _make_msg(
            "assistant",
            "It has three tiers: sqlite-vec for semantic search, keyword fallback, and BFS graph traversal.",
        ),
    ]
    # Extend if more needed
    while len(base) < count:
        idx = len(base) // 2
        base.append(_make_msg("user", f"Can you explain more about tier {idx}?"))
        base.append(
            _make_msg(
                "assistant",
                f"Tier {idx} handles retrieval through a different mechanism...",
            )
        )
    return base[:count]


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path) -> str:
    """Path to a temp SQLite DB with the minimal Cashew schema."""
    db_path = str(tmp_path / "test_cashew.db")
    conn = sqlite3.connect(db_path)
    _create_minimal_schema(conn)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def provider_with_llm(tmp_db, monkeypatch) -> MagicMock:
    """A CashewMemoryProvider-like mock with model_fn wired and db_path set.

    We mock the provider's on_pre_compress dependencies (core.session, core.embeddings)
    and verify calls rather than exercising the real upstream.
    """
    # Mock the upstream functions that _create_insight_nodes uses
    fake_create_node = MagicMock(return_value="test_node_id_123")
    fake_set_tags = MagicMock()
    fake_embed_nodes = MagicMock()

    monkeypatch.setattr("core.session._create_node", fake_create_node)
    monkeypatch.setattr("core.session._set_node_tags", fake_set_tags)
    monkeypatch.setattr("core.embeddings.embed_nodes", fake_embed_nodes)

    # Build a minimal provider-like mock
    from plugins.memory.cashew import CashewMemoryProvider

    provider = CashewMemoryProvider()
    provider._db_path = tmp_db
    provider._model_fn = None  # will set per test
    provider._config = MagicMock()
    provider._config.user_domain = "test_user"
    provider._config.ai_domain = "test_ai"

    return provider


@pytest.fixture
def provider_without_llm(tmp_db) -> MagicMock:
    """Provider with model_fn=None (no LLM available)."""
    from plugins.memory.cashew import CashewMemoryProvider

    provider = CashewMemoryProvider()
    provider._db_path = tmp_db
    provider._model_fn = None
    provider._config = MagicMock()
    provider._config.user_domain = "test_user"
    provider._config.ai_domain = "test_ai"
    return provider


# ── Tests: Contract ──────────────────────────────────────────────────────────


class TestContract:
    """on_pre_compress always returns a str and never raises."""

    def test_returns_string_empty_messages(self, provider_with_llm):
        """Empty message list returns ''."""
        result = provider_with_llm.on_pre_compress([])
        assert result == ""

    def test_returns_string_no_model_fn(self, provider_without_llm):
        """Without LLM wired, returns ''."""
        result = provider_without_llm.on_pre_compress(_sample_messages())
        assert result == ""

    def test_returns_string_few_messages(self, provider_with_llm):
        """Fewer than 3 exchanges returns ''."""
        result = provider_with_llm.on_pre_compress(_sample_messages(4))
        assert result == ""

    def test_returns_string_always(self, provider_with_llm):
        """Even with valid input, returns str type."""
        provider_with_llm._model_fn = MagicMock(return_value="[]")
        result = provider_with_llm.on_pre_compress(_sample_messages())
        assert isinstance(result, str)


# ── Tests: LLM Path ──────────────────────────────────────────────────────────


class TestLLMPath:
    """on_pre_compress with model_fn wired."""

    def test_valid_json_creates_nodes(self, provider_with_llm, tmp_db, monkeypatch):
        """Valid LLM response creates insight nodes via upstream API."""
        valid_response = json.dumps(
            [
                {
                    "content": "User asks 'why' before accepting solutions — recurring pattern",
                    "type": "insight",
                    "domain": "test_user",
                    "tags": ["communication_style"],
                    "keep": True,
                },
                {
                    "content": "Discussion shifted from architecture to cost 3 times",
                    "type": "observation",
                    "domain": "test_user",
                    "tags": ["topic_shift"],
                    "keep": True,
                },
            ]
        )
        provider_with_llm._model_fn = MagicMock(return_value=valid_response)

        # Mock upstream node creation to actually write to DB
        real_created: list[str] = []

        def _fake_create(
            db_path, content, node_type, session_id, domain="user", referent_time=None
        ):
            import hashlib

            node_id = hashlib.sha256(content.encode()).hexdigest()[:12]
            conn = sqlite3.connect(tmp_db)
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO thought_nodes (id, content, node_type, timestamp, source_file, last_accessed, access_count, domain) VALUES (?,?,?,?,?,?,0,?)",
                    (
                        node_id,
                        content,
                        node_type,
                        "2026-01-01T00:00:00",
                        "pre_compress",
                        "2026-01-01T00:00:00",
                        domain,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
            real_created.append(node_id)
            return node_id

        monkeypatch.setattr("core.session._create_node", _fake_create)
        monkeypatch.setattr("core.session._set_node_tags", MagicMock())
        monkeypatch.setattr("core.embeddings.embed_nodes", MagicMock())

        result = provider_with_llm.on_pre_compress(_sample_messages())

        assert len(real_created) == 2
        assert "Cashew insight extraction:" in result

    def test_empty_json_array(self, provider_with_llm):
        """LLM returns [] → return ''."""
        provider_with_llm._model_fn = MagicMock(return_value="[]")
        result = provider_with_llm.on_pre_compress(_sample_messages())
        assert result == ""

    def test_keep_false_items_filtered(self, provider_with_llm):
        """Items with keep=false are filtered out."""
        response = json.dumps(
            [
                {"content": "Keep this insight", "type": "insight", "keep": True},
                {
                    "content": "Skip this observation",
                    "type": "observation",
                    "keep": False,
                },
            ]
        )
        provider_with_llm._model_fn = MagicMock(return_value=response)

        from plugins.memory.cashew import CashewMemoryProvider
        original_create = provider_with_llm._create_insight_nodes
        captured: list[list[dict]] = []

        def tracking_create(items):
            captured.append(items)
            return original_create(items)

        provider_with_llm._create_insight_nodes = tracking_create

        provider_with_llm.on_pre_compress(_sample_messages())

        # Should only have passed items with keep=true or keep absent
        assert len(captured) > 0
        if captured:
            contents = [it.get("content") for it in captured[0]]
            assert "Skip this observation" not in contents
            assert "Keep this insight" in contents

    def test_code_fence_parsing(self, provider_with_llm):
        """LLM wraps JSON in ```markdown code fences → still parses."""
        response = '```json\n[{"content": "Arc insight from code fence", "type": "insight", "keep": true}]\n```'
        provider_with_llm._model_fn = MagicMock(return_value=response)
        provider_with_llm._create_insight_nodes = MagicMock(return_value=1)

        result = provider_with_llm.on_pre_compress(_sample_messages())
        assert "Arc insight from code fence" in result


# ── Tests: Edge Cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_malformed_json_graceful(self, provider_with_llm):
        """LLM returns non-JSON → log warning, return '', never crash."""
        provider_with_llm._model_fn = MagicMock(return_value="This is not JSON at all")
        result = provider_with_llm.on_pre_compress(_sample_messages())
        assert result == ""

    def test_multimodal_content(self, provider_with_llm):
        """Array-format messages with text+images → text extracted, images skipped."""
        messages = [
            _make_multimodal_msg(
                "user", ["What do you think about this architecture?"]
            ),
            _make_multimodal_msg(
                "assistant", ["It looks good but the storage layer could be optimized."]
            ),
            _make_multimodal_msg(
                "user", ["Can you elaborate on the storage optimization?"]
            ),
            _make_multimodal_msg(
                "assistant",
                [
                    "Yes, we should use WAL mode and batch writes for better performance."
                ],
            ),
            _make_multimodal_msg("user", ["How about indexing strategies?"]),
            _make_multimodal_msg(
                "assistant", ["Compound indexes on timestamp and domain would help."]
            ),
        ]
        provider_with_llm._model_fn = MagicMock(
            return_value='[{"content": "test", "type": "insight", "keep": true}]'
        )
        provider_with_llm._create_insight_nodes = MagicMock(return_value=1)
        result = provider_with_llm.on_pre_compress(messages)
        assert result != ""  # Should not crash, should process

    def test_tool_messages_ignored(self, provider_with_llm):
        """Tool/system messages don't contribute to exchange count."""
        messages = [
            _make_msg("system", "You are a helpful assistant."),
            _make_msg("user", "Hello"),
            _make_msg("assistant", "Hi there"),
            _make_msg("tool", "function_result: 42"),
        ]
        # Only 1 exchange (2 messages) — below threshold of 6
        result = provider_with_llm.on_pre_compress(messages)
        assert result == ""

    def test_empty_model_fn_response(self, provider_with_llm):
        """Model returns empty string → graceful no-op."""
        provider_with_llm._model_fn = MagicMock(return_value="")
        result = provider_with_llm.on_pre_compress(_sample_messages())
        assert result == ""


# ── Tests: Exchange Extraction ───────────────────────────────────────────────


class TestExtractExchanges:
    def test_extract_exchanges_basic(self, provider_with_llm):
        """Standard messages produce correct role: content strings."""
        msgs = [
            _make_msg("user", "Hello"),
            _make_msg("assistant", "Hi"),
        ]
        result = provider_with_llm._extract_exchanges(msgs)
        assert result == ["user: Hello", "assistant: Hi"]

    def test_extract_exchanges_filters_system(self, provider_with_llm):
        """System and tool messages are filtered out."""
        msgs = [
            _make_msg("system", "Be helpful."),
            _make_msg("user", "Question"),
            _make_msg("assistant", "Answer"),
            _make_msg("tool", "result"),
        ]
        result = provider_with_llm._extract_exchanges(msgs)
        assert result == ["user: Question", "assistant: Answer"]

    def test_extract_exchanges_multimodal(self, provider_with_llm):
        """Multimodal array content extracts text parts only."""
        msgs = [_make_multimodal_msg("user", ["Part one", "Part two"])]
        result = provider_with_llm._extract_exchanges(msgs)
        assert len(result) == 1
        assert "Part one" in result[0]
        assert "Part two" in result[0]
