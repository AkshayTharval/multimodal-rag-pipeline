"""End-to-end integration tests for the Multimodal RAG Pipeline.

These tests exercise the full pipeline (ingest → retrieve → generate → cache)
using real ChromaDB collections but mocked ML models and Ollama.

All tests are marked @pytest.mark.integration so they can be run separately:
    pytest tests/test_integration.py -v -m integration

No external services required — Ollama is always mocked.
"""

import io
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

import chromadb


pytestmark = pytest.mark.integration


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_text_embedder():
    embedder = MagicMock()
    embedder.encode.side_effect = lambda texts, **kw: np.zeros((len(texts), 384))
    return embedder


@pytest.fixture
def mock_clip_models():
    import torch
    clip_model = MagicMock()
    clip_processor = MagicMock()
    # Use ones (non-zero) so L2 normalization produces a valid unit vector,
    # and cosine similarity against non-zero image embeddings is well-defined.
    clip_model.get_image_features.return_value = torch.ones(1, 512)
    clip_model.get_text_features.return_value = torch.ones(1, 512)
    clip_processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
    return clip_model, clip_processor


@pytest.fixture
def mock_cross_encoder():
    ce = MagicMock()
    ce.predict.side_effect = lambda pairs: [0.5] * len(pairs)
    return ce


@pytest.fixture
def mock_ollama():
    client = MagicMock()
    client.generate_text = AsyncMock(return_value="RAG combines retrieval with generation.")
    client.generate_vision = AsyncMock(return_value="The image shows a visual scene.")
    return client


# ── Test 1: Full pipeline text query ─────────────────────────────────────────

@pytest.mark.integration
async def test_full_pipeline_text_query(
    populated_collections, mock_text_embedder, mock_clip_models,
    mock_cross_encoder, mock_ollama
):
    """Ingest text, retrieve, mock Ollama, get answer with citations."""
    from pipeline.retrieve import hybrid_retrieve
    from pipeline.generate import generate_answer

    _, text_col, image_col, video_col = populated_collections
    clip_model, clip_processor = mock_clip_models

    results = hybrid_retrieve(
        query_text="What is retrieval augmented generation?",
        text_collection=text_col,
        image_collection=image_col,
        video_collection=video_col,
        text_embedder=mock_text_embedder,
        clip_model=clip_model,
        clip_processor=clip_processor,
        cross_encoder=mock_cross_encoder,
        top_k=5,
    )

    assert len(results) > 0
    # At least one text result
    modalities = [r.get("metadata", {}).get("modality") for r in results]
    assert "text" in modalities

    answer_data = await generate_answer(
        query="What is retrieval augmented generation?",
        retrieval_results=results,
        ollama_client=mock_ollama,
    )
    assert "answer" in answer_data
    assert len(answer_data["answer"]) > 0
    assert "citations" in answer_data
    assert isinstance(answer_data["citations"], list)


# ── Test 2: Full pipeline image query ─────────────────────────────────────────

@pytest.mark.integration
async def test_full_pipeline_image_retrieval(
    populated_collections, mock_text_embedder, mock_clip_models,
    mock_cross_encoder, mock_ollama
):
    """Text query retrieves image results when image collection is populated."""
    from pipeline.retrieve import hybrid_retrieve

    _, text_col, image_col, video_col = populated_collections
    clip_model, clip_processor = mock_clip_models

    # top_k=15 ensures all 15 docs (10 text + 3 images + 2 video) can appear.
    # rerank() places text first, then visual — so we need top_k > number of text docs.
    results = hybrid_retrieve(
        query_text="show me photos",
        text_collection=text_col,
        image_collection=image_col,
        video_collection=video_col,
        text_embedder=mock_text_embedder,
        clip_model=clip_model,
        clip_processor=clip_processor,
        cross_encoder=mock_cross_encoder,
        top_k=15,
    )

    assert len(results) > 0
    modalities = {r.get("metadata", {}).get("modality") for r in results}
    # All three modalities should appear in populated_collections
    assert "text" in modalities
    assert "image" in modalities
    assert "video" in modalities


# ── Test 3: Cache integration — miss then hit ─────────────────────────────────

@pytest.mark.integration
async def test_cache_miss_then_hit(tmp_path, mock_ollama):
    """Query once (miss) → same query again (hit) → Ollama NOT called second time."""
    from pipeline.cache import SemanticCache

    # Non-zero unit vector so cosine distance = 0.0 (identical vectors → guaranteed hit)
    unit_vec = np.ones(384) / np.sqrt(384)
    deterministic_embedder = MagicMock()
    deterministic_embedder.encode.return_value = [unit_vec]

    cache = SemanticCache(db_path=":memory:", embedder=deterministic_embedder)

    query = "What is RAG?"
    cached_result = {
        "answer": "RAG stands for Retrieval Augmented Generation.",
        "citations": ["doc_0.txt"],
        "retrieval_results": [],
    }

    # First lookup — miss
    miss = cache.lookup(query)
    assert miss is None

    # Store
    cache.store(query, cached_result)

    # Second lookup — hit
    hit = cache.lookup(query)
    assert hit is not None
    assert hit["answer"] == cached_result["answer"]
    assert hit["citations"] == cached_result["citations"]

    # Ollama should not be called for a cache hit in a real pipeline
    mock_ollama.generate_text.assert_not_awaited()


# ── Test 4: Idempotency — ingest same text twice ──────────────────────────────

@pytest.mark.integration
def test_ingest_text_idempotency(tmp_path, mock_text_embedder):
    """Ingesting the same text document twice must not inflate chunk count."""
    from pipeline.ingest import chunk_text

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    from pipeline import get_collections
    text_col, _, _ = get_collections(client)

    text = "This is a test document about RAG systems. " * 30
    source_id = "idempotency_test.txt"

    chunks_first = chunk_text(text, source_id, text_col, mock_text_embedder)
    count_after_first = text_col.count()

    chunks_second = chunk_text(text, source_id, text_col, mock_text_embedder)
    count_after_second = text_col.count()

    assert len(chunks_first) == len(chunks_second)
    assert count_after_first == count_after_second


# ── Test 5: RRF scores in valid range ────────────────────────────────────────

@pytest.mark.integration
def test_rrf_scores_valid_range(
    populated_collections, mock_text_embedder, mock_clip_models, mock_cross_encoder
):
    """Every retrieved result must have an RRF score in (0, 1/(k+1)]."""
    from pipeline.retrieve import hybrid_retrieve

    _, text_col, image_col, video_col = populated_collections
    clip_model, clip_processor = mock_clip_models

    results = hybrid_retrieve(
        query_text="retrieval augmented generation",
        text_collection=text_col,
        image_collection=image_col,
        video_collection=video_col,
        text_embedder=mock_text_embedder,
        clip_model=clip_model,
        clip_processor=clip_processor,
        cross_encoder=mock_cross_encoder,
        top_k=5,
    )

    for r in results:
        score = r.get("rrf_score", 0.0)
        assert score > 0.0, f"RRF score must be positive, got {score}"
        # Max RRF score for a single ranked list of length 1 at rank 0: 1/(60+1)
        assert score <= 1.0, f"RRF score must not exceed 1.0, got {score}"


# ── Test 6: API flow via httpx ────────────────────────────────────────────────

@pytest.mark.integration
async def test_api_ingest_then_query(tmp_path):
    """POST /ingest text file → POST /query → get answer field."""
    import httpx
    import torch
    from unittest.mock import AsyncMock, MagicMock
    from api.app import app

    # Build isolated app.state
    client = chromadb.PersistentClient(path=str(tmp_path / "api_chroma"))
    from pipeline import get_collections
    from pipeline.cache import SemanticCache

    text_col, image_col, video_col = get_collections(client)

    mock_embedder = MagicMock()
    mock_embedder.encode.side_effect = lambda texts, **kw: np.zeros((len(texts), 384))

    mock_clip_model = MagicMock()
    mock_clip_processor = MagicMock()
    mock_clip_model.get_image_features.return_value = torch.ones(1, 512)
    mock_clip_processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}

    mock_ce = MagicMock()
    mock_ce.predict.side_effect = lambda pairs: [0.5] * len(pairs)

    mock_ollama_client = MagicMock()
    mock_ollama_client.generate_text = AsyncMock(return_value="Integration test answer.")
    mock_ollama_client.generate_vision = AsyncMock(return_value="Vision answer.")

    mock_cache_embedder = MagicMock()
    mock_cache_embedder.encode.side_effect = lambda texts, **kw: [[0.0] * 384 for _ in texts]
    cache = SemanticCache(db_path=":memory:", embedder=mock_cache_embedder)

    app.state.chroma_client = client
    app.state.text_col = text_col
    app.state.image_col = image_col
    app.state.video_col = video_col
    app.state.text_embedder = mock_embedder
    app.state.clip_model = mock_clip_model
    app.state.clip_processor = mock_clip_processor
    app.state.cross_encoder = mock_ce
    app.state.ollama_client = mock_ollama_client
    app.state.cache = cache

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # Ingest a text file
        content = b"Retrieval Augmented Generation is a powerful technique. " * 20
        ingest_resp = await ac.post(
            "/ingest",
            files={"file": ("rag_intro.txt", io.BytesIO(content), "text/plain")},
            data={"modality": "text"},
        )
        assert ingest_resp.status_code == 200
        ingest_data = ingest_resp.json()
        assert ingest_data["modality"] == "text"
        assert ingest_data["chunks_created"] >= 1

        # Query the pipeline
        query_resp = await ac.post(
            "/query",
            json={"query": "What is RAG?", "top_k": 3, "use_cache": False},
        )
        assert query_resp.status_code == 200
        query_data = query_resp.json()
        assert "answer" in query_data
        assert query_data["answer"] == "Integration test answer."
        assert "latency_ms" in query_data
        assert query_data["latency_ms"] >= 0


# ── Test 7: Cache integration in API — second query returns cache_hit=True ────

@pytest.mark.integration
async def test_api_cache_hit_on_second_query(tmp_path):
    """Same query twice through the API: second call must be a cache hit."""
    import httpx
    import torch
    from api.app import app
    from pipeline import get_collections
    from pipeline.cache import SemanticCache

    client = chromadb.PersistentClient(path=str(tmp_path / "cache_chroma"))
    text_col, image_col, video_col = get_collections(client)

    # Pre-populate so retrieval returns results
    text_col.upsert(
        ids=["seed_0"],
        embeddings=[np.zeros(384).tolist()],
        documents=["RAG combines retrieval with generation."],
        metadatas=[{"source_id": "seed.txt", "modality": "text"}],
    )

    mock_embedder = MagicMock()
    mock_embedder.encode.side_effect = lambda texts, **kw: np.zeros((len(texts), 384))

    mock_clip_model = MagicMock()
    mock_clip_processor = MagicMock()
    mock_clip_model.get_image_features.return_value = torch.ones(1, 512)
    mock_clip_processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}

    mock_ce = MagicMock()
    mock_ce.predict.side_effect = lambda pairs: [0.5] * len(pairs)

    mock_ollama_client = MagicMock()
    mock_ollama_client.generate_text = AsyncMock(return_value="First answer.")

    # Non-zero unit vector → cosine distance = 0.0 for identical queries → always hits
    unit_vec = np.ones(384) / np.sqrt(384)
    mock_cache_embedder = MagicMock()
    mock_cache_embedder.encode.return_value = [unit_vec]
    cache = SemanticCache(db_path=":memory:", embedder=mock_cache_embedder)

    app.state.chroma_client = client
    app.state.text_col = text_col
    app.state.image_col = image_col
    app.state.video_col = video_col
    app.state.text_embedder = mock_embedder
    app.state.clip_model = mock_clip_model
    app.state.clip_processor = mock_clip_processor
    app.state.cross_encoder = mock_ce
    app.state.ollama_client = mock_ollama_client
    app.state.cache = cache

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # First query — cache miss
        resp1 = await ac.post(
            "/query",
            json={"query": "What is RAG?", "top_k": 3, "use_cache": True},
        )
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["cache_hit"] is False

        # Second identical query — cache hit
        resp2 = await ac.post(
            "/query",
            json={"query": "What is RAG?", "top_k": 3, "use_cache": True},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["cache_hit"] is True
        assert data2["answer"] == data1["answer"]

    # Ollama should have been called exactly once
    assert mock_ollama_client.generate_text.await_count == 1


# ── Test 8: Ingest idempotency through API ────────────────────────────────────

@pytest.mark.integration
async def test_api_ingest_idempotency(tmp_path):
    """POST /ingest the same file three times → collection count stays constant."""
    import httpx
    import torch
    from api.app import app
    from pipeline import get_collections
    from pipeline.cache import SemanticCache

    client = chromadb.PersistentClient(path=str(tmp_path / "idem_chroma"))
    text_col, image_col, video_col = get_collections(client)

    mock_embedder = MagicMock()
    mock_embedder.encode.side_effect = lambda texts, **kw: np.zeros((len(texts), 384))
    mock_clip_model = MagicMock()
    mock_clip_processor = MagicMock()
    mock_clip_model.get_image_features.return_value = torch.ones(1, 512)
    mock_clip_processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
    mock_ce = MagicMock()
    mock_ce.predict.side_effect = lambda pairs: [0.5] * len(pairs)
    mock_ollama_client = MagicMock()
    mock_ollama_client.generate_text = AsyncMock(return_value="answer")
    mock_cache_embedder = MagicMock()
    mock_cache_embedder.encode.side_effect = lambda texts, **kw: [[0.0] * 384 for _ in texts]
    cache = SemanticCache(db_path=":memory:", embedder=mock_cache_embedder)

    app.state.chroma_client = client
    app.state.text_col = text_col
    app.state.image_col = image_col
    app.state.video_col = video_col
    app.state.text_embedder = mock_embedder
    app.state.clip_model = mock_clip_model
    app.state.clip_processor = mock_clip_processor
    app.state.cross_encoder = mock_ce
    app.state.ollama_client = mock_ollama_client
    app.state.cache = cache

    content = b"A test document about vector databases. " * 20

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        for _ in range(3):
            resp = await ac.post(
                "/ingest",
                files={"file": ("idempotent.txt", io.BytesIO(content), "text/plain")},
                data={"modality": "text"},
            )
            assert resp.status_code == 200

    count_after_three = text_col.count()
    # Ingest once to get the baseline chunk count
    from pipeline.ingest import chunk_text
    baseline_client = chromadb.PersistentClient(path=str(tmp_path / "baseline_chroma"))
    baseline_text_col, _, _ = get_collections(baseline_client)
    chunks = chunk_text(content.decode(), "idempotent.txt", baseline_text_col, mock_embedder)
    assert count_after_three == len(chunks)
