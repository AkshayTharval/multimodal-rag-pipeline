"""Tests for pipeline/retrieve.py.

All tests except @pytest.mark.slow run without any external services or
model downloads. BM25 and RRF tests are purely algorithmic and fast.
Cross-encoder tests mock the model's predict() method.
"""

import pytest
import numpy as np
import chromadb

from pipeline import get_collections
from pipeline.retrieve import (
    BM25Index,
    dense_query,
    rerank,
    rrf_fusion,
)


# ── RRF fusion unit tests ─────────────────────────────────────────────────────

class TestRrfFusion:
    def test_single_list_passthrough(self):
        ranked = [
            {"id": "a", "rank": 1, "document": "doc a", "metadata": {}},
            {"id": "b", "rank": 2, "document": "doc b", "metadata": {}},
        ]
        result = rrf_fusion([ranked])
        assert result[0]["id"] == "a"
        assert result[1]["id"] == "b"

    def test_exact_rrf_math_two_lists(self):
        """Manually verify RRF formula: score(d) = Σ 1/(k + rank)."""
        k = 60
        list1 = [
            {"id": "a", "rank": 1, "document": "d", "metadata": {}},
            {"id": "b", "rank": 2, "document": "d", "metadata": {}},
        ]
        list2 = [
            {"id": "b", "rank": 1, "document": "d", "metadata": {}},
            {"id": "a", "rank": 2, "document": "d", "metadata": {}},
        ]
        result = rrf_fusion([list1, list2], k=k)

        expected_a = 1 / (k + 1) + 1 / (k + 2)   # rank 1 in list1, rank 2 in list2
        expected_b = 1 / (k + 2) + 1 / (k + 1)   # rank 2 in list1, rank 1 in list2

        scores = {r["id"]: r["rrf_score"] for r in result}
        assert abs(scores["a"] - expected_a) < 1e-9
        assert abs(scores["b"] - expected_b) < 1e-9

    def test_tied_scores_both_present(self):
        """a and b both rank 1 in one list each — scores should be equal."""
        k = 60
        list1 = [{"id": "a", "rank": 1, "document": "d", "metadata": {}}]
        list2 = [{"id": "b", "rank": 1, "document": "d", "metadata": {}}]
        result = rrf_fusion([list1, list2], k=k)
        scores = {r["id"]: r["rrf_score"] for r in result}
        assert abs(scores["a"] - scores["b"]) < 1e-9

    def test_no_overlap_all_docs_present(self):
        """Lists with completely different docs — all must appear in output."""
        list1 = [{"id": "x", "rank": 1, "document": "d", "metadata": {}}]
        list2 = [{"id": "y", "rank": 1, "document": "d", "metadata": {}}]
        result = rrf_fusion([list1, list2])
        ids = {r["id"] for r in result}
        assert ids == {"x", "y"}

    def test_k_sensitivity_lower_k_amplifies_top_rank(self):
        """Lower k means rank-1 docs score relatively higher."""
        ranked = [{"id": "a", "rank": 1, "document": "d", "metadata": {}}]

        result_k1 = rrf_fusion([ranked], k=1)
        result_k60 = rrf_fusion([ranked], k=60)

        assert result_k1[0]["rrf_score"] > result_k60[0]["rrf_score"]

    def test_empty_lists_returns_empty(self):
        assert rrf_fusion([]) == []
        assert rrf_fusion([[]]) == []

    def test_output_sorted_descending(self):
        ranked = [
            {"id": "a", "rank": 3, "document": "d", "metadata": {}},
            {"id": "b", "rank": 1, "document": "d", "metadata": {}},
            {"id": "c", "rank": 2, "document": "d", "metadata": {}},
        ]
        result = rrf_fusion([ranked])
        scores = [r["rrf_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_score_upper_bound(self):
        """With k=60 and rank=1 from one list, score <= 1/(60+1) ≈ 0.0164."""
        ranked = [{"id": "a", "rank": 1, "document": "d", "metadata": {}}]
        result = rrf_fusion([ranked], k=60)
        assert result[0]["rrf_score"] <= 1 / 61 + 1e-9

    def test_multiple_lists_boost_shared_doc(self):
        """A document appearing in more lists should score higher."""
        doc_in_both = {"id": "shared", "rank": 1, "document": "d", "metadata": {}}
        doc_once = {"id": "unique", "rank": 1, "document": "d", "metadata": {}}

        list1 = [doc_in_both]
        list2 = [doc_in_both, doc_once]
        result = rrf_fusion([list1, list2])
        scores = {r["id"]: r["rrf_score"] for r in result}
        assert scores["shared"] > scores["unique"]


# ── BM25Index tests ───────────────────────────────────────────────────────────

def _populate_text_collection(
    collection: chromadb.Collection, n: int = 20
) -> list[str]:
    """Insert n fake text documents into the collection."""
    docs = [f"document about topic {i} with unique term uniqueterm{i}" for i in range(n)]
    ids = [f"doc_{i}" for i in range(n)]
    # Use simple zero vectors (BM25 doesn't use embeddings)
    embeddings = [[0.0] * 384 for _ in range(n)]
    metadatas = [{"source_id": f"src_{i}", "modality": "text"} for i in range(n)]
    collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metadatas)
    return docs


class TestBM25Index:
    def test_empty_collection_returns_empty_results(self, ephemeral_client):
        text_col, _, _ = get_collections(ephemeral_client)
        index = BM25Index(text_col)
        results = index.query("anything", top_k=5)
        assert results == []

    def test_returns_at_most_top_k(self, ephemeral_client):
        text_col, _, _ = get_collections(ephemeral_client)
        _populate_text_collection(text_col, n=20)
        index = BM25Index(text_col)
        results = index.query("topic document", top_k=5)
        assert len(results) <= 5

    def test_returns_top_k_when_enough_docs(self, ephemeral_client):
        text_col, _, _ = get_collections(ephemeral_client)
        _populate_text_collection(text_col, n=20)
        index = BM25Index(text_col)
        results = index.query("topic document", top_k=10)
        assert len(results) == 10

    def test_relevant_doc_scores_higher(self, ephemeral_client):
        """A doc containing the exact query term should outscore unrelated docs."""
        text_col, _, _ = get_collections(ephemeral_client)
        _populate_text_collection(text_col, n=10)
        index = BM25Index(text_col)
        results = index.query("uniqueterm5", top_k=5)
        # The doc with uniqueterm5 should be first
        assert "uniqueterm5" in results[0]["document"]

    def test_result_has_required_keys(self, ephemeral_client):
        text_col, _, _ = get_collections(ephemeral_client)
        _populate_text_collection(text_col, n=5)
        index = BM25Index(text_col)
        results = index.query("topic", top_k=3)
        for r in results:
            assert "id" in r
            assert "score" in r
            assert "rank" in r
            assert "document" in r
            assert "metadata" in r

    def test_ranks_are_sequential_from_one(self, ephemeral_client):
        text_col, _, _ = get_collections(ephemeral_client)
        _populate_text_collection(text_col, n=5)
        index = BM25Index(text_col)
        results = index.query("topic", top_k=5)
        ranks = [r["rank"] for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_fewer_docs_than_top_k_returns_all(self, ephemeral_client):
        text_col, _, _ = get_collections(ephemeral_client)
        _populate_text_collection(text_col, n=3)
        index = BM25Index(text_col)
        results = index.query("topic", top_k=10)
        assert len(results) == 3


# ── dense_query tests ─────────────────────────────────────────────────────────

def _make_unit_vector(dim: int, index: int) -> list[float]:
    """Return a unit vector with 1.0 at position `index`."""
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _populate_with_known_embeddings(collection: chromadb.Collection) -> None:
    """Insert 5 orthogonal unit vectors as documents."""
    dim = 384
    for i in range(5):
        collection.upsert(
            ids=[f"unit_{i}"],
            embeddings=[_make_unit_vector(dim, i)],
            documents=[f"document {i}"],
            metadatas=[{"source_id": f"src_{i}", "modality": "text"}],
        )


class TestDenseQuery:
    def test_empty_collection_returns_empty(self, ephemeral_client):
        text_col, _, _ = get_collections(ephemeral_client)
        results = dense_query(text_col, [0.0] * 384, top_k=5)
        assert results == []

    def test_identical_vector_is_top_result(self, ephemeral_client):
        text_col, _, _ = get_collections(ephemeral_client)
        _populate_with_known_embeddings(text_col)
        # Query with unit_2's exact vector → unit_2 should be rank 1
        query_vec = _make_unit_vector(384, 2)
        results = dense_query(text_col, query_vec, top_k=5)
        assert results[0]["id"] == "unit_2"

    def test_returns_at_most_top_k(self, ephemeral_client):
        text_col, _, _ = get_collections(ephemeral_client)
        _populate_with_known_embeddings(text_col)
        results = dense_query(text_col, _make_unit_vector(384, 0), top_k=3)
        assert len(results) <= 3

    def test_scores_between_minus_one_and_one(self, ephemeral_client):
        """Cosine similarity must be in [-1, 1]."""
        text_col, _, _ = get_collections(ephemeral_client)
        _populate_with_known_embeddings(text_col)
        results = dense_query(text_col, _make_unit_vector(384, 0), top_k=5)
        for r in results:
            assert -1.0 - 1e-6 <= r["score"] <= 1.0 + 1e-6

    def test_result_has_required_keys(self, ephemeral_client):
        text_col, _, _ = get_collections(ephemeral_client)
        _populate_with_known_embeddings(text_col)
        results = dense_query(text_col, _make_unit_vector(384, 0), top_k=2)
        for r in results:
            for key in ("id", "score", "rank", "document", "metadata"):
                assert key in r

    def test_scores_sorted_descending(self, ephemeral_client):
        text_col, _, _ = get_collections(ephemeral_client)
        _populate_with_known_embeddings(text_col)
        results = dense_query(text_col, _make_unit_vector(384, 0), top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ── rerank tests ──────────────────────────────────────────────────────────────

class TestRerank:
    def _make_candidates(self, modalities_and_docs: list[tuple[str, str]]) -> list[dict]:
        return [
            {
                "id": f"doc_{i}",
                "rank": i + 1,
                "rrf_score": 1.0 / (60 + i + 1),
                "document": doc,
                "metadata": {"source_id": f"src_{i}", "modality": modality},
            }
            for i, (modality, doc) in enumerate(modalities_and_docs)
        ]

    def test_rerank_orders_by_cross_encoder_score(self, mocker):
        """CrossEncoder scores [0.1, 0.9, 0.5] → order should be [1, 2, 0]."""
        mock_ce = mocker.MagicMock()
        mock_ce.predict.return_value = np.array([0.1, 0.9, 0.5])

        candidates = self._make_candidates([
            ("text", "doc 0"),
            ("text", "doc 1"),
            ("text", "doc 2"),
        ])
        result = rerank("query", candidates, mock_ce, top_k=3)
        assert result[0]["id"] == "doc_1"   # score 0.9
        assert result[1]["id"] == "doc_2"   # score 0.5
        assert result[2]["id"] == "doc_0"   # score 0.1

    def test_rerank_score_added_to_text_candidates(self, mocker):
        mock_ce = mocker.MagicMock()
        mock_ce.predict.return_value = np.array([0.7, 0.3])

        candidates = self._make_candidates([("text", "a"), ("text", "b")])
        result = rerank("query", candidates, mock_ce, top_k=2)
        for r in result:
            assert "rerank_score" in r

    def test_visual_candidates_not_passed_to_cross_encoder(self, mocker):
        """Image/video candidates bypass the cross-encoder entirely."""
        mock_ce = mocker.MagicMock()
        mock_ce.predict.return_value = np.array([0.5])

        candidates = self._make_candidates([
            ("text", "text doc"),
            ("image", "image doc"),
            ("video", "video doc"),
        ])
        rerank("query", candidates, mock_ce, top_k=3)

        # predict called once with exactly 1 pair (the text doc only)
        mock_ce.predict.assert_called_once()
        pairs_passed = mock_ce.predict.call_args[0][0]
        assert len(pairs_passed) == 1

    def test_visual_candidates_appended_after_text(self, mocker):
        mock_ce = mocker.MagicMock()
        mock_ce.predict.return_value = np.array([0.5])

        candidates = self._make_candidates([
            ("text", "text doc"),
            ("image", "image doc"),
        ])
        result = rerank("query", candidates, mock_ce, top_k=2)
        modalities = [r["metadata"]["modality"] for r in result]
        # text should come first, then image
        assert modalities.index("text") < modalities.index("image")

    def test_top_k_respected(self, mocker):
        mock_ce = mocker.MagicMock()
        mock_ce.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        candidates = self._make_candidates([("text", f"doc {i}") for i in range(5)])
        result = rerank("query", candidates, mock_ce, top_k=3)
        assert len(result) == 3

    def test_empty_candidates_returns_empty(self, mocker):
        mock_ce = mocker.MagicMock()
        result = rerank("query", [], mock_ce, top_k=5)
        assert result == []
        mock_ce.predict.assert_not_called()

    def test_all_visual_no_cross_encoder_call(self, mocker):
        mock_ce = mocker.MagicMock()
        candidates = self._make_candidates([
            ("image", "img 0"),
            ("video", "vid 0"),
        ])
        result = rerank("query", candidates, mock_ce, top_k=2)
        mock_ce.predict.assert_not_called()
        assert len(result) == 2
