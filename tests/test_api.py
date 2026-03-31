"""Tests for api/app.py.

Uses httpx.AsyncClient with the FastAPI app directly (no live server needed).
All external dependencies (ChromaDB, ML models, Ollama, SemanticCache) are
injected via app.state overrides in fixtures.

Tests never call real Ollama or load real ML models.
"""

import io
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.app import app, _detect_modality


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_app_state(tmp_path):
    """Override app.state with mock objects before each test."""
    import chromadb
    from pipeline.cache import SemanticCache

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    from pipeline import get_collections
    text_col, image_col, video_col = get_collections(client)

    import numpy as np
    mock_embedder = MagicMock()
    # Must return numpy arrays — pipeline code calls .tolist() on each element
    mock_embedder.encode.side_effect = lambda texts, **kw: np.zeros((len(texts), 384))

    mock_clip_model = MagicMock()
    mock_clip_processor = MagicMock()
    import torch
    mock_clip_model.get_image_features.return_value = torch.ones(1, 512)
    mock_clip_processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}

    mock_cross_encoder = MagicMock()
    # Return one score per pair so rerank() doesn't IndexError
    mock_cross_encoder.predict.side_effect = lambda pairs: [0.5] * len(pairs)

    mock_ollama = MagicMock()
    mock_ollama.generate_text = AsyncMock(return_value="Mocked answer.")
    mock_ollama.generate_vision = AsyncMock(return_value="Mocked vision answer.")
    mock_ollama.is_available = AsyncMock(return_value=True)

    mock_cache_embedder = MagicMock()
    mock_cache_embedder.encode.side_effect = lambda texts, **kw: [[0.1] * 384 for _ in texts]

    cache = SemanticCache(db_path=":memory:", embedder=mock_cache_embedder)

    # Pre-populate text collection so retrieval returns results
    import numpy as _np
    text_col.upsert(
        ids=["seed_doc_0"],
        embeddings=[_np.zeros(384).tolist()],
        documents=["RAG combines retrieval with language model generation."],
        metadatas=[{"source_id": "seed.txt", "modality": "text"}],
    )

    # Inject into app.state
    app.state.chroma_client = client
    app.state.text_col = text_col
    app.state.image_col = image_col
    app.state.video_col = video_col
    app.state.text_embedder = mock_embedder
    app.state.clip_model = mock_clip_model
    app.state.clip_processor = mock_clip_processor
    app.state.cross_encoder = mock_cross_encoder
    app.state.ollama_client = mock_ollama
    app.state.cache = cache

    return app


@pytest.fixture
async def async_client(mock_app_state):
    """httpx AsyncClient pointed at the FastAPI app (no live server).

    httpx >= 0.28 removed the `app=` kwarg; use ASGITransport instead.
    """
    transport = httpx.ASGITransport(app=mock_app_state)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        yield client


# ── /health tests ─────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    async def test_health_returns_200(self, async_client):
        with patch("api.app.httpx.AsyncClient") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_httpx.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_resp
            )
            response = await async_client.get("/health")
        assert response.status_code == 200

    async def test_health_response_has_status_key(self, async_client):
        with patch("api.app.httpx.AsyncClient") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_httpx.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_resp
            )
            response = await async_client.get("/health")
        data = response.json()
        assert "status" in data

    async def test_health_marks_ollama_unreachable(self, async_client):
        with patch("api.app.httpx.AsyncClient") as mock_httpx:
            mock_httpx.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("refused")
            )
            response = await async_client.get("/health")
        data = response.json()
        assert data["ollama"] == "unreachable"

    async def test_health_chroma_ok(self, async_client):
        with patch("api.app.httpx.AsyncClient") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_httpx.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_resp
            )
            response = await async_client.get("/health")
        data = response.json()
        assert data.get("chroma") == "ok"


# ── /ingest tests ─────────────────────────────────────────────────────────────

class TestIngestEndpoint:
    async def test_ingest_text_file_returns_text_modality(self, async_client):
        content = b"This is a test document about RAG pipelines." * 20
        response = await async_client.post(
            "/ingest",
            files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
            data={"modality": "text"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["modality"] == "text"
        assert data["status"] == "ok"

    async def test_ingest_image_file_returns_image_modality(self, async_client):
        from PIL import Image
        img_bytes = io.BytesIO()
        Image.new("RGB", (10, 10)).save(img_bytes, format="PNG")
        img_bytes.seek(0)

        response = await async_client.post(
            "/ingest",
            files={"file": ("photo.png", img_bytes, "image/png")},
            data={"modality": "image"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["modality"] == "image"

    async def test_ingest_returns_document_id(self, async_client):
        content = b"Document content for ID test." * 20
        response = await async_client.post(
            "/ingest",
            files={"file": ("doc.txt", io.BytesIO(content), "text/plain")},
            data={"modality": "text"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert len(data["document_id"]) > 0

    async def test_ingest_unsupported_modality_returns_422(self, async_client):
        content = b"unknown content"
        response = await async_client.post(
            "/ingest",
            files={"file": ("file.xyz", io.BytesIO(content), "application/octet-stream")},
            data={"modality": "auto"},
        )
        assert response.status_code == 422


# ── /query tests ──────────────────────────────────────────────────────────────

class TestQueryEndpoint:
    async def test_query_returns_answer(self, async_client):
        response = await async_client.post(
            "/query",
            json={"query": "What is RAG?", "top_k": 3, "use_cache": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "Mocked answer."

    async def test_query_returns_cache_hit_false_on_miss(self, async_client):
        response = await async_client.post(
            "/query",
            json={"query": "unique query xyz123", "top_k": 3, "use_cache": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["cache_hit"] is False

    async def test_query_cache_hit_skips_ollama(self, async_client, mock_app_state):
        """Pre-populate cache → second query should hit cache, Ollama not called."""
        from unittest.mock import MagicMock
        import numpy as np

        # Use a mock embedder that returns identical vectors for the same query
        identical_vec = np.array([1.0] + [0.0] * 383, dtype=np.float32)
        mock_cache_embedder = MagicMock()
        mock_cache_embedder.encode.return_value = [identical_vec]

        from pipeline.cache import SemanticCache
        cache = SemanticCache(db_path=":memory:", embedder=mock_cache_embedder)
        cached_result = {
            "answer": "Cached answer.",
            "citations": ["cached_doc.txt"],
            "retrieval_results": [],
        }
        cache.store("What is RAG?", cached_result)
        mock_app_state.state.cache = cache

        response = await async_client.post(
            "/query",
            json={"query": "What is RAG?", "top_k": 3, "use_cache": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["cache_hit"] is True
        assert data["answer"] == "Cached answer."
        # Ollama should not have been called
        mock_app_state.state.ollama_client.generate_text.assert_not_awaited()

    async def test_query_has_latency_ms(self, async_client):
        response = await async_client.post(
            "/query",
            json={"query": "test", "use_cache": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    async def test_query_returns_citations_list(self, async_client):
        response = await async_client.post(
            "/query",
            json={"query": "test query", "use_cache": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert "citations" in data
        assert isinstance(data["citations"], list)


# ── /stats tests ──────────────────────────────────────────────────────────────

class TestStatsEndpoint:
    async def test_stats_returns_200(self, async_client):
        response = await async_client.get("/stats")
        assert response.status_code == 200

    async def test_stats_has_expected_keys(self, async_client):
        response = await async_client.get("/stats")
        data = response.json()
        assert "total_queries" in data
        assert "cache_hit_rate" in data
        assert "text_chunks" in data
        assert "image_embeddings" in data
        assert "video_keyframes" in data
        assert "cached_queries" in data


# ── /metrics tests ────────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    async def test_metrics_returns_200(self, async_client):
        response = await async_client.get("/metrics")
        assert response.status_code == 200

    async def test_metrics_content_type_is_text_plain(self, async_client):
        response = await async_client.get("/metrics")
        assert "text/plain" in response.headers.get("content-type", "")

    async def test_metrics_contains_prometheus_format(self, async_client):
        response = await async_client.get("/metrics")
        text = response.text
        assert "rag_" in text  # all our metrics start with "rag_"


# ── _detect_modality unit tests ───────────────────────────────────────────────

class TestDetectModality:
    def test_explicit_hint_overrides_everything(self):
        assert _detect_modality("file.mp4", "video/mp4", "text") == "text"

    def test_auto_detects_text_from_content_type(self):
        assert _detect_modality("file.txt", "text/plain", "auto") == "text"

    def test_auto_detects_image_from_content_type(self):
        assert _detect_modality("photo.jpg", "image/jpeg", "auto") == "image"

    def test_auto_detects_video_from_content_type(self):
        assert _detect_modality("clip.mp4", "video/mp4", "auto") == "video"

    def test_auto_falls_back_to_extension_for_text(self):
        assert _detect_modality("doc.txt", "application/octet-stream", "auto") == "text"

    def test_auto_falls_back_to_extension_for_image(self):
        assert _detect_modality("photo.png", "application/octet-stream", "auto") == "image"

    def test_auto_falls_back_to_extension_for_video(self):
        assert _detect_modality("video.mp4", "application/octet-stream", "auto") == "video"

    def test_unknown_extension_returns_unknown(self):
        assert _detect_modality("file.xyz", "application/octet-stream", "auto") == "unknown"
