"""Pipeline package — ingestion, retrieval, generation, cache."""

from pipeline.config import (
    CHROMA_DB_PATH,
    TEXT_COLLECTION,
    IMAGE_COLLECTION,
    VIDEO_COLLECTION,
)

import chromadb


def get_chroma_client(path: str = CHROMA_DB_PATH) -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client backed by the given path.

    Args:
        path: Directory path for ChromaDB storage.

    Returns:
        A ChromaDB PersistentClient instance.
    """
    return chromadb.PersistentClient(path=path)


def get_collections(
    client: chromadb.ClientAPI,
) -> tuple[chromadb.Collection, chromadb.Collection, chromadb.Collection]:
    """Get or create the three ChromaDB collections.

    Collections use cosine distance space and are dimension-specific:
    - text_chunks: 384-dim (all-MiniLM-L6-v2)
    - image_embeddings: 512-dim (CLIP ViT-B/32)
    - video_keyframes: 512-dim (CLIP ViT-B/32)

    IMPORTANT: hnsw:space is set at creation time and is immutable.

    Args:
        client: An active ChromaDB client (PersistentClient or EphemeralClient).

    Returns:
        Tuple of (text_collection, image_collection, video_collection).
    """
    text_col = client.get_or_create_collection(
        TEXT_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    image_col = client.get_or_create_collection(
        IMAGE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    video_col = client.get_or_create_collection(
        VIDEO_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return text_col, image_col, video_col
