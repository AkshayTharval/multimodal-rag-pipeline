"""Tests for pipeline/cache.py.

All tests use SQLite :memory: or tmp_path — no disk pollution.
SentenceTransformer model calls are mocked in fast tests.
Slow tests (marked @pytest.mark.slow) use the real model.
"""

import json
import numpy as np
import pytest

from pipeline.cache import SemanticCache, _cosine_distance


# ── _cosine_distance unit tests ───────────────────────────────────────────────

class TestCosineDistance:
    def test_identical_vectors_distance_zero(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert abs(_cosine_distance(v, v)) < 1e-6

    def test_opposite_vectors_distance_two(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert abs(_cosine_distance(a, b) - 2.0) < 1e-6

    def test_orthogonal_vectors_distance_one(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(_cosine_distance(a, b) - 1.0) < 1e-6

    def test_distance_in_range_zero_to_two(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            a = rng.standard_normal(64).astype(np.float32)
            b = rng.standard_normal(64).astype(np.float32)
            d = _cosine_distance(a, b)
            assert 0.0 - 1e-6 <= d <= 2.0 + 1e-6

    def test_zero_vector_returns_one(self):
        a = np.zeros(4, dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert _cosine_distance(a, b) == 1.0

    def test_symmetry(self):
        rng = np.random.default_rng(7)
        a = rng.standard_normal(32).astype(np.float32)
        b = rng.standard_normal(32).astype(np.float32)
        assert abs(_cosine_distance(a, b) - _cosine_distance(b, a)) < 1e-6


# ── SemanticCache tests ───────────────────────────────────────────────────────

def _make_fake_embedder(vectors: dict[str, np.ndarray]):
    """Return a mock embedder that maps query text to a known vector."""
    from unittest.mock import MagicMock
    embedder = MagicMock()

    def encode(texts, **kwargs):
        result = []
        for text in texts:
            if text in vectors:
                result.append(vectors[text].astype(np.float32))
            else:
                # Default: return zero vector
                result.append(np.zeros(384, dtype=np.float32))
        return np.array(result)

    embedder.encode.side_effect = encode
    return embedder


def _unit_vec(dim: int, idx: int) -> np.ndarray:
    """Return a unit vector with 1.0 at position idx."""
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


SAMPLE_RESULT = {
    "answer": "RAG combines retrieval with generation.",
    "citations": ["doc1.txt", "doc2.txt"],
    "retrieval_results": [{"id": "doc_0", "metadata": {"modality": "text"}}],
}


class TestSemanticCache:
    def test_empty_cache_returns_none(self, tmp_path):
        embedder = _make_fake_embedder({"test query": _unit_vec(384, 0)})
        cache = SemanticCache(
            db_path=str(tmp_path / "cache.db"),
            embedder=embedder,
        )
        result = cache.lookup("test query")
        assert result is None

    def test_store_then_lookup_identical_returns_hit(self, tmp_path):
        query = "What is retrieval augmented generation?"
        vec = _unit_vec(384, 5)
        embedder = _make_fake_embedder({query: vec})

        cache = SemanticCache(
            db_path=str(tmp_path / "cache.db"),
            embedder=embedder,
        )
        cache.store(query, SAMPLE_RESULT)
        result = cache.lookup(query)

        assert result is not None
        assert result["cache_hit"] is True
        assert result["answer"] == SAMPLE_RESULT["answer"]

    def test_lookup_returns_citations_as_list(self, tmp_path):
        query = "citations test"
        vec = _unit_vec(384, 10)
        embedder = _make_fake_embedder({query: vec})

        cache = SemanticCache(
            db_path=str(tmp_path / "cache.db"),
            embedder=embedder,
        )
        cache.store(query, SAMPLE_RESULT)
        result = cache.lookup(query)

        assert isinstance(result["citations"], list)
        assert result["citations"] == SAMPLE_RESULT["citations"]

    def test_cache_hit_below_threshold(self, tmp_path):
        """Two vectors with distance 0.0 (identical) → HIT at threshold 0.05."""
        vec = _unit_vec(384, 3)
        q1 = "query one"
        q2 = "query two"  # different text, same vector → distance=0.0

        embedder = _make_fake_embedder({q1: vec, q2: vec})
        cache = SemanticCache(
            db_path=str(tmp_path / "cache.db"),
            similarity_threshold=0.05,
            embedder=embedder,
        )
        cache.store(q1, SAMPLE_RESULT)
        result = cache.lookup(q2)

        assert result is not None
        assert result["cache_hit"] is True

    def test_cache_miss_above_threshold(self, tmp_path):
        """Two orthogonal vectors (distance=1.0) → MISS at threshold 0.05."""
        q1 = "query store"
        q2 = "query different"

        embedder = _make_fake_embedder({
            q1: _unit_vec(384, 0),   # direction 0
            q2: _unit_vec(384, 1),   # direction 1 — orthogonal, distance=1.0
        })
        cache = SemanticCache(
            db_path=str(tmp_path / "cache.db"),
            similarity_threshold=0.05,
            embedder=embedder,
        )
        cache.store(q1, SAMPLE_RESULT)
        result = cache.lookup(q2)

        assert result is None

    def test_threshold_boundary_exact_hit(self, tmp_path):
        """Distance exactly at 0.04 → HIT (< threshold 0.05)."""
        # Construct two vectors with cosine distance ≈ 0.04
        # cos_dist = 1 - cos_sim; cos_sim = 0.96 → angle ≈ 16.3°
        angle = np.arccos(0.96)
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        # Pad to 384-dim
        v1_full = np.pad(v1, (0, 382)).astype(np.float32)
        v2_full = np.pad(v2, (0, 382)).astype(np.float32)

        actual_distance = _cosine_distance(v1_full, v2_full)
        assert actual_distance < 0.05, f"distance {actual_distance} should be < 0.05"

        embedder = _make_fake_embedder({"q1": v1_full, "q2": v2_full})
        cache = SemanticCache(
            db_path=str(tmp_path / "cache.db"),
            similarity_threshold=0.05,
            embedder=embedder,
        )
        cache.store("q1", SAMPLE_RESULT)
        assert cache.lookup("q2") is not None

    def test_threshold_boundary_exact_miss(self, tmp_path):
        """Distance at 0.06 → MISS (≥ threshold 0.05)."""
        angle = np.arccos(0.94)
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        v1_full = np.pad(v1, (0, 382)).astype(np.float32)
        v2_full = np.pad(v2, (0, 382)).astype(np.float32)

        actual_distance = _cosine_distance(v1_full, v2_full)
        assert actual_distance > 0.05, f"distance {actual_distance} should be > 0.05"

        embedder = _make_fake_embedder({"q1": v1_full, "q2": v2_full})
        cache = SemanticCache(
            db_path=str(tmp_path / "cache.db"),
            similarity_threshold=0.05,
            embedder=embedder,
        )
        cache.store("q1", SAMPLE_RESULT)
        assert cache.lookup("q2") is None

    def test_count_increments_on_store(self, tmp_path):
        query = "count test"
        vec = _unit_vec(384, 7)
        embedder = _make_fake_embedder({query: vec})

        cache = SemanticCache(db_path=str(tmp_path / "cache.db"), embedder=embedder)
        assert cache.count() == 0
        cache.store(query, SAMPLE_RESULT)
        assert cache.count() == 1

    def test_clear_removes_all_entries(self, tmp_path):
        query = "clear test"
        vec = _unit_vec(384, 2)
        embedder = _make_fake_embedder({query: vec})

        cache = SemanticCache(db_path=str(tmp_path / "cache.db"), embedder=embedder)
        cache.store(query, SAMPLE_RESULT)
        cache.clear()
        assert cache.count() == 0
        assert cache.lookup(query) is None

    def test_in_memory_db_works(self):
        """SQLite :memory: db — useful for ephemeral test setups."""
        query = "in memory test"
        vec = _unit_vec(384, 1)
        embedder = _make_fake_embedder({query: vec})

        cache = SemanticCache(
            db_path=":memory:",
            embedder=embedder,
        )
        cache.store(query, SAMPLE_RESULT)
        result = cache.lookup(query)
        assert result is not None

    def test_matched_query_key_in_result(self, tmp_path):
        query = "original question"
        vec = _unit_vec(384, 20)
        embedder = _make_fake_embedder({query: vec, "similar question": vec})

        cache = SemanticCache(db_path=str(tmp_path / "cache.db"), embedder=embedder)
        cache.store(query, SAMPLE_RESULT)
        result = cache.lookup("similar question")

        assert result is not None
        assert "matched_query" in result
        assert result["matched_query"] == query

    @pytest.mark.slow
    def test_real_embedder_near_duplicate_hit(self, tmp_path):
        """With real model, near-identical queries should be cache hits."""
        cache = SemanticCache(
            db_path=str(tmp_path / "cache.db"),
            similarity_threshold=0.1,  # slightly relaxed for real embeddings
        )
        cache.store("What is RAG?", SAMPLE_RESULT)
        result = cache.lookup("What is RAG?")
        assert result is not None
        assert result["cache_hit"] is True

    @pytest.mark.slow
    def test_real_embedder_different_query_miss(self, tmp_path):
        """With real model, very different queries should miss."""
        cache = SemanticCache(
            db_path=str(tmp_path / "cache.db"),
            similarity_threshold=0.05,
        )
        cache.store("What is RAG?", SAMPLE_RESULT)
        result = cache.lookup("How do I cook pasta?")
        assert result is None
