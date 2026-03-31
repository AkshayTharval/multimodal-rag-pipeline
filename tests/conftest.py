"""Shared pytest fixtures for the multimodal RAG pipeline test suite.

All tests that need ChromaDB use the ephemeral_client fixture (in-memory,
no disk I/O). This ensures tests are hermetic and can run without external
services.
"""

import io
import pytest
import chromadb
from PIL import Image


@pytest.fixture
def ephemeral_client(tmp_path) -> chromadb.ClientAPI:
    """A fully isolated ChromaDB client backed by a unique tmp_path per test.

    Uses PersistentClient with a per-test temp directory so there is zero
    shared state between tests. tmp_path is cleaned up automatically by pytest.
    This avoids the EphemeralClient singleton issue in chromadb 0.6.x.
    """
    return chromadb.PersistentClient(path=str(tmp_path / "chroma"))


@pytest.fixture
def tiny_image() -> Image.Image:
    """A tiny 10×10 white RGB PIL image for fast embedding tests."""
    return Image.new("RGB", (10, 10), color=(255, 255, 255))


@pytest.fixture
def tiny_image_path(tmp_path: "pathlib.Path", tiny_image: Image.Image) -> str:
    """Save the tiny image to a temp file and return the path string."""
    path = tmp_path / "test_image.jpg"
    tiny_image.save(str(path), format="JPEG")
    return str(path)


@pytest.fixture
def sample_text() -> str:
    """A multi-paragraph text sample long enough to produce multiple chunks."""
    return (
        "Retrieval Augmented Generation (RAG) is a technique that combines "
        "information retrieval with large language model generation. "
        "Instead of relying solely on parametric knowledge encoded in model weights, "
        "RAG systems retrieve relevant documents from an external corpus at query time "
        "and condition the generator on those documents. "
        "This allows the model to access up-to-date information and reduces hallucination. "
        "The retrieval component typically uses dense vector search, sparse BM25, "
        "or a hybrid of both. Dense retrieval encodes documents and queries into a "
        "shared embedding space using models like sentence-transformers or CLIP. "
        "Sparse retrieval uses term-frequency statistics to score documents against a query. "
        "Reciprocal Rank Fusion (RRF) is a commonly used technique to combine multiple "
        "ranked lists from different retrieval methods into a single unified ranking. "
        "Cross-encoder reranking further improves precision by scoring query-document "
        "pairs jointly, at the cost of higher compute. "
        "Multimodal RAG extends this paradigm to non-text modalities: images are embedded "
        "with CLIP, video keyframes are extracted and similarly embedded, enabling "
        "text queries to retrieve visually relevant content. "
        "Production RAG systems require observability: per-stage latency tracking, "
        "cache hit rates, and retrieval quality metrics like MRR and recall@k."
    )


@pytest.fixture
def collections(ephemeral_client: chromadb.ClientAPI):
    """Return the three standard collections on an ephemeral client."""
    from pipeline import get_collections
    return get_collections(ephemeral_client)


@pytest.fixture(scope="session")
def populated_collections(tmp_path_factory):
    """Session-scoped fixture with pre-populated text, image, and video collections.

    Contains 10 text docs, 3 images, and 2 video keyframes (simulated).
    Used exclusively by @pytest.mark.integration tests.
    """
    import numpy as np
    import chromadb
    from pipeline import get_collections

    db_path = str(tmp_path_factory.mktemp("integration_chroma"))
    client = chromadb.PersistentClient(path=db_path)
    text_col, image_col, video_col = get_collections(client)

    # ── 10 text documents (384-dim zero embeddings) ───────────────────────────
    text_docs = [
        "Retrieval Augmented Generation combines retrieval with language model generation.",
        "BM25 is a sparse retrieval method based on term frequency and inverse document frequency.",
        "Dense retrieval uses neural embeddings to find semantically similar documents.",
        "Reciprocal Rank Fusion merges multiple ranked lists into a single unified ranking.",
        "Cross-encoder reranking scores query-document pairs jointly for higher precision.",
        "ChromaDB is an open-source vector database supporting cosine and L2 similarity.",
        "CLIP encodes both images and text into a shared 512-dimensional embedding space.",
        "Semantic caching reduces LLM calls by returning cached answers for similar queries.",
        "OpenTelemetry provides distributed tracing for observing multi-stage pipelines.",
        "Prometheus counters and histograms expose latency and throughput metrics.",
    ]
    text_col.upsert(
        ids=[f"text_doc_{i}" for i in range(len(text_docs))],
        embeddings=[np.zeros(384).tolist() for _ in text_docs],
        documents=text_docs,
        metadatas=[
            {"source_id": f"doc_{i}.txt", "modality": "text", "chunk_index": 0}
            for i in range(len(text_docs))
        ],
    )

    # ── 3 image embeddings (512-dim, non-zero so cosine similarity is defined) ─
    img_emb = (np.ones(512) / np.sqrt(512)).tolist()
    image_col.upsert(
        ids=["img_0", "img_1", "img_2"],
        embeddings=[img_emb for _ in range(3)],
        documents=["", "", ""],
        metadatas=[
            {"source_id": f"photo_{i}.jpg", "modality": "image", "thumbnail_b64": ""}
            for i in range(3)
        ],
    )

    # ── 2 video keyframes (512-dim, non-zero) ─────────────────────────────────
    vid_emb = (np.ones(512) / np.sqrt(512)).tolist()
    video_col.upsert(
        ids=["vid_frame_0", "vid_frame_1"],
        embeddings=[vid_emb for _ in range(2)],
        documents=["", ""],
        metadatas=[
            {
                "source_id": "demo_video.mp4",
                "modality": "video",
                "timestamp_sec": i * 5,
                "thumbnail_b64": "",
            }
            for i in range(2)
        ],
    )

    return client, text_col, image_col, video_col
