"""Central configuration for the multimodal RAG pipeline.

All environment variables and constants live here.
Import from this module — never hard-code values in pipeline code.
"""

import os

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# Collection names
TEXT_COLLECTION: str = "text_chunks"
IMAGE_COLLECTION: str = "image_embeddings"
VIDEO_COLLECTION: str = "video_keyframes"

# ── Embedding models ──────────────────────────────────────────────────────────
# Text: 384-dim
TEXT_EMBED_MODEL: str = "all-MiniLM-L6-v2"
TEXT_EMBED_DIM: int = 384

# Image/Video: 512-dim
CLIP_MODEL_ID: str = "openai/clip-vit-base-patch32"
CLIP_EMBED_DIM: int = 512

# ── Ingestion ─────────────────────────────────────────────────────────────────
TEXT_CHUNK_SIZE: int = 512       # characters per chunk
TEXT_CHUNK_OVERLAP: int = 50     # character overlap between chunks
VIDEO_SAMPLE_INTERVAL_SEC: int = 5  # sample one frame every N seconds
THUMBNAIL_MAX_SIZE: tuple[int, int] = (256, 256)

# ── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K: int = 5
RRF_K: int = 60  # RRF constant; formula: 1 / (RRF_K + rank)

# ── Cross-encoder ─────────────────────────────────────────────────────────────
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TEXT_MODEL: str = "llama3.2"
OLLAMA_VISION_MODEL: str = "llama3.2-vision"
OLLAMA_TEXT_TIMEOUT: float = 120.0   # seconds
OLLAMA_VISION_TIMEOUT: float = 180.0  # seconds; vision inference is slower

# ── Semantic cache ────────────────────────────────────────────────────────────
CACHE_DB_PATH: str = os.getenv("CACHE_DB_PATH", "./cache.db")
# IMPORTANT: this is cosine DISTANCE (0=identical, 2=opposite).
# threshold=0.05 means similarity > 0.95 is required for a cache HIT.
CACHE_DISTANCE_THRESHOLD: float = float(os.getenv("CACHE_THRESHOLD", "0.05"))
