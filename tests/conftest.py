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
def ephemeral_client() -> chromadb.ClientAPI:
    """An in-memory ChromaDB client — zero disk I/O, no cleanup needed."""
    return chromadb.EphemeralClient()


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
