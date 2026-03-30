"""Multimodal ingestion pipeline.

Three entry points:
- chunk_text()       — text documents → chunked → sentence-transformer embedded → ChromaDB
- embed_image()      — image files → CLIP embedded → ChromaDB
- extract_keyframes() — video files → OpenCV keyframe sampling → CLIP embedded → ChromaDB

All functions use upsert semantics: re-ingesting the same source_id does NOT duplicate entries.
Document IDs are deterministic, derived from source_id + chunk/frame index.
"""

import base64
import io
import logging
from pathlib import Path

import chromadb
import cv2
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from pipeline.config import (
    CLIP_MODEL_ID,
    TEXT_CHUNK_OVERLAP,
    TEXT_CHUNK_SIZE,
    TEXT_EMBED_MODEL,
    THUMBNAIL_MAX_SIZE,
    VIDEO_SAMPLE_INTERVAL_SEC,
)

logger = logging.getLogger(__name__)


# ── Text ingestion ────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source_id: str,
    collection: chromadb.Collection,
    embedder: SentenceTransformer,
    chunk_size: int = TEXT_CHUNK_SIZE,
    overlap: int = TEXT_CHUNK_OVERLAP,
) -> list[str]:
    """Chunk a text document and upsert embeddings into ChromaDB.

    Uses a sliding window over characters with a fixed overlap.
    Document IDs are ``{source_id}_chunk_{i}`` — idempotent on re-ingest.

    Args:
        text: Raw text content to ingest.
        source_id: Unique identifier for the source document (e.g. filename).
        collection: ChromaDB collection for text chunks (384-dim, cosine).
        embedder: SentenceTransformer model (all-MiniLM-L6-v2).
        chunk_size: Maximum characters per chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        List of chunk strings that were upserted.
    """
    if not text.strip():
        logger.warning("chunk_text: empty text for source_id=%s, skipping", source_id)
        return []

    chunks = _sliding_window_chunks(text, chunk_size, overlap)
    if not chunks:
        return []

    embeddings = embedder.encode(chunks, batch_size=32, show_progress_bar=False)

    ids = [f"{source_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source_id": source_id,
            "modality": "text",
            "chunk_index": i,
            "char_start": i * (chunk_size - overlap),
            "char_end": i * (chunk_size - overlap) + len(chunk),
        }
        for i, chunk in enumerate(chunks)
    ]

    collection.upsert(
        ids=ids,
        embeddings=[emb.tolist() for emb in embeddings],
        documents=chunks,
        metadatas=metadatas,
    )
    logger.info("chunk_text: upserted %d chunks for source_id=%s", len(chunks), source_id)
    return chunks


def _sliding_window_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character-level windows.

    Args:
        text: Input text.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    if chunk_size <= overlap:
        raise ValueError(f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})")

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += step
    return chunks


# ── Image ingestion ───────────────────────────────────────────────────────────

def embed_image(
    image_path: str,
    source_id: str,
    collection: chromadb.Collection,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
) -> str:
    """Embed an image with CLIP and upsert into ChromaDB.

    Stores a base64-encoded JPEG thumbnail (max 256×256) in the metadata
    for display in the Streamlit retrieval provenance UI.

    Document ID is ``img_{source_id}`` — idempotent on re-ingest.

    Args:
        image_path: Path to the image file.
        source_id: Unique identifier (typically the filename without path).
        collection: ChromaDB collection for images (512-dim, cosine).
        clip_model: Loaded CLIPModel instance.
        clip_processor: Loaded CLIPProcessor instance.

    Returns:
        The document ID that was upserted.
    """
    image = Image.open(image_path).convert("RGB")
    embedding = _clip_embed_image(image, clip_model, clip_processor)
    thumbnail_b64 = _encode_thumbnail(image)

    doc_id = f"img_{source_id}"
    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[f"Image: {source_id}"],
        metadatas=[
            {
                "source_id": source_id,
                "modality": "image",
                "file_path": image_path,
                "thumbnail_b64": thumbnail_b64,
            }
        ],
    )
    logger.info("embed_image: upserted image source_id=%s", source_id)
    return doc_id


# ── Video keyframe ingestion ──────────────────────────────────────────────────

def extract_keyframes(
    video_path: str,
    source_id: str,
    collection: chromadb.Collection,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    sample_interval_sec: int = VIDEO_SAMPLE_INTERVAL_SEC,
) -> list[str]:
    """Extract keyframes from a video, embed with CLIP, upsert into ChromaDB.

    Samples one frame every ``sample_interval_sec`` seconds.
    Document IDs are ``vid_{source_id}_frame_{frame_index}`` — idempotent.

    Args:
        video_path: Path to the video file.
        source_id: Unique identifier for the video (e.g. filename).
        collection: ChromaDB collection for video keyframes (512-dim, cosine).
        clip_model: Loaded CLIPModel instance.
        clip_processor: Loaded CLIPProcessor instance.
        sample_interval_sec: Seconds between sampled frames.

    Returns:
        List of document IDs that were upserted.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"extract_keyframes: cannot open video at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(fps * sample_interval_sec))

    doc_ids: list[str] = []
    frame_index = 0
    sample_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            timestamp_sec = frame_index / fps
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            embedding = _clip_embed_image(image, clip_model, clip_processor)
            thumbnail_b64 = _encode_thumbnail(image)

            doc_id = f"vid_{source_id}_frame_{sample_count}"
            collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[f"Video frame: {source_id} at {timestamp_sec:.1f}s"],
                metadatas=[
                    {
                        "source_id": source_id,
                        "modality": "video",
                        "file_path": video_path,
                        "frame_index": sample_count,
                        "timestamp_sec": round(timestamp_sec, 2),
                        "thumbnail_b64": thumbnail_b64,
                    }
                ],
            )
            doc_ids.append(doc_id)
            sample_count += 1

        frame_index += 1

    cap.release()
    logger.info(
        "extract_keyframes: upserted %d frames for source_id=%s", len(doc_ids), source_id
    )
    return doc_ids


# ── Shared CLIP utilities ─────────────────────────────────────────────────────

def _clip_embed_image(
    image: Image.Image,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
) -> list[float]:
    """Encode a PIL image with CLIP and return an L2-normalised 512-dim vector.

    Args:
        image: RGB PIL Image.
        clip_model: Loaded CLIPModel.
        clip_processor: Loaded CLIPProcessor.

    Returns:
        L2-normalised embedding as a Python list of floats (length 512).
    """
    inputs = clip_processor(images=image, return_tensors="pt")
    features = clip_model.get_image_features(**inputs)
    features = features.detach().float()
    # L2 normalise
    norm = features.norm(dim=-1, keepdim=True)
    normalised = (features / norm).squeeze(0)
    return normalised.tolist()


def clip_embed_text(
    text: str,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
) -> list[float]:
    """Encode a text string with CLIP's text encoder (for image/video retrieval).

    This is the cross-modal bridge: encode a query with CLIP text encoder
    to retrieve visually similar images and video frames.

    Args:
        text: Query string.
        clip_model: Loaded CLIPModel.
        clip_processor: Loaded CLIPProcessor.

    Returns:
        L2-normalised 512-dim embedding as a list of floats.
    """
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    features = clip_model.get_text_features(**inputs)
    features = features.detach().float()
    norm = features.norm(dim=-1, keepdim=True)
    normalised = (features / norm).squeeze(0)
    return normalised.tolist()


def _encode_thumbnail(image: Image.Image, max_size: tuple[int, int] = THUMBNAIL_MAX_SIZE) -> str:
    """Resize image and encode as base64 JPEG string for metadata storage.

    Args:
        image: PIL Image (any mode).
        max_size: Maximum (width, height) — image is resized proportionally.

    Returns:
        Base64-encoded JPEG string (no data URI prefix).
    """
    thumb = image.copy()
    thumb.thumbnail(max_size, Image.LANCZOS)
    if thumb.mode != "RGB":
        thumb = thumb.convert("RGB")
    buffer = io.BytesIO()
    thumb.save(buffer, format="JPEG", quality=70)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ── Model loader helpers ──────────────────────────────────────────────────────

def load_text_embedder(model_name: str = TEXT_EMBED_MODEL) -> SentenceTransformer:
    """Load and return a SentenceTransformer text embedder.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded SentenceTransformer instance.
    """
    return SentenceTransformer(model_name)


def load_clip_model(model_id: str = CLIP_MODEL_ID) -> tuple[CLIPModel, CLIPProcessor]:
    """Load and return a CLIP model and processor.

    Args:
        model_id: HuggingFace model identifier.

    Returns:
        Tuple of (CLIPModel, CLIPProcessor).
    """
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.eval()
    return model, processor
