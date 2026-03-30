"""Tests for pipeline/ingest.py.

All tests use EphemeralClient (in-memory ChromaDB) — no disk I/O.
CLIP and SentenceTransformer model tests are marked @pytest.mark.slow
because they download models on first run.
"""

import io
import os
import tempfile

import cv2
import numpy as np
import pytest
from PIL import Image

import chromadb
from pipeline import get_collections
from pipeline.ingest import (
    _encode_thumbnail,
    _sliding_window_chunks,
    chunk_text,
    clip_embed_text,
    embed_image,
    extract_keyframes,
    load_clip_model,
    load_text_embedder,
)


# ── _sliding_window_chunks unit tests ────────────────────────────────────────

class TestSlidingWindowChunks:
    def test_short_text_produces_one_chunk(self):
        text = "Hello world"
        chunks = _sliding_window_chunks(text, chunk_size=512, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_long_text_produces_multiple_chunks(self):
        text = "A" * 1200
        chunks = _sliding_window_chunks(text, chunk_size=512, overlap=50)
        assert len(chunks) > 1

    def test_overlap_content_shared_between_chunks(self):
        text = "X" * 600
        chunks = _sliding_window_chunks(text, chunk_size=512, overlap=100)
        # Each chunk except the last should be ~chunk_size chars
        assert len(chunks[0]) == 512

    def test_empty_text_returns_empty_list(self):
        chunks = _sliding_window_chunks("", chunk_size=512, overlap=50)
        assert chunks == []

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            _sliding_window_chunks("hello", chunk_size=50, overlap=50)

    def test_exact_chunk_size_produces_one_chunk(self):
        text = "B" * 512
        chunks = _sliding_window_chunks(text, chunk_size=512, overlap=50)
        assert len(chunks) == 1


# ── _encode_thumbnail unit tests ─────────────────────────────────────────────

class TestEncodeThumbnail:
    def test_returns_non_empty_base64_string(self, tiny_image):
        b64 = _encode_thumbnail(tiny_image)
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_is_valid_base64(self, tiny_image):
        import base64
        b64 = _encode_thumbnail(tiny_image)
        # Should not raise
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_decoded_is_valid_jpeg(self, tiny_image):
        import base64
        b64 = _encode_thumbnail(tiny_image)
        decoded = base64.b64decode(b64)
        image = Image.open(io.BytesIO(decoded))
        assert image.format == "JPEG"

    def test_thumbnail_respects_max_size(self):
        large_image = Image.new("RGB", (1000, 1000), color=(100, 150, 200))
        b64 = _encode_thumbnail(large_image, max_size=(256, 256))
        import base64
        decoded = base64.b64decode(b64)
        thumb = Image.open(io.BytesIO(decoded))
        assert thumb.width <= 256
        assert thumb.height <= 256

    def test_rgba_image_converts_to_rgb(self):
        rgba_image = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
        b64 = _encode_thumbnail(rgba_image)
        import base64
        decoded = base64.b64decode(b64)
        thumb = Image.open(io.BytesIO(decoded))
        assert thumb.mode == "RGB"


# ── chunk_text tests (uses EphemeralClient, no model download) ────────────────

class TestChunkText:
    @pytest.mark.slow
    def test_creates_multiple_chunks_for_long_text(self, ephemeral_client, sample_text):
        embedder = load_text_embedder()
        text_col, _, _ = get_collections(ephemeral_client)
        chunks = chunk_text(sample_text, "doc_001", text_col, embedder)
        assert len(chunks) > 1

    @pytest.mark.slow
    def test_idempotency_no_duplication(self, ephemeral_client, sample_text):
        """Re-ingesting the same document must not increase collection count."""
        embedder = load_text_embedder()
        text_col, _, _ = get_collections(ephemeral_client)

        chunk_text(sample_text, "doc_idempotent", text_col, embedder)
        count_after_first = text_col.count()

        chunk_text(sample_text, "doc_idempotent", text_col, embedder)
        count_after_second = text_col.count()

        assert count_after_first == count_after_second

    @pytest.mark.slow
    def test_metadata_has_required_fields(self, ephemeral_client, sample_text):
        embedder = load_text_embedder()
        text_col, _, _ = get_collections(ephemeral_client)
        chunk_text(sample_text, "doc_meta", text_col, embedder)

        results = text_col.get(include=["metadatas"])
        for meta in results["metadatas"]:
            assert meta["source_id"] == "doc_meta"
            assert meta["modality"] == "text"
            assert "chunk_index" in meta

    @pytest.mark.slow
    def test_embeddings_have_correct_dimension(self, ephemeral_client, sample_text):
        embedder = load_text_embedder()
        text_col, _, _ = get_collections(ephemeral_client)
        chunk_text(sample_text, "doc_dim", text_col, embedder)

        results = text_col.get(include=["embeddings"])
        for emb in results["embeddings"]:
            assert len(emb) == 384  # all-MiniLM-L6-v2 produces 384-dim

    def test_empty_text_returns_empty_list(self, ephemeral_client, mocker):
        mock_embedder = mocker.MagicMock()
        text_col, _, _ = get_collections(ephemeral_client)
        result = chunk_text("   ", "doc_empty", text_col, mock_embedder)
        assert result == []
        mock_embedder.encode.assert_not_called()

    def test_chunk_ids_are_deterministic(self, ephemeral_client, mocker):
        """IDs follow pattern source_id_chunk_N."""
        mock_embedder = mocker.MagicMock()
        # Return N embeddings matching however many chunks are produced
        mock_embedder.encode.side_effect = lambda texts, **kw: np.zeros((len(texts), 384))
        text_col, _, _ = get_collections(ephemeral_client)

        long_text = "word " * 300  # ~1500 chars → 3+ chunks at 512
        chunk_text(long_text, "doc_ids", text_col, mock_embedder)

        results = text_col.get(include=["metadatas"])
        ids = text_col.get()["ids"]
        for doc_id in ids:
            assert doc_id.startswith("doc_ids_chunk_")


# ── embed_image tests ─────────────────────────────────────────────────────────

class TestEmbedImage:
    @pytest.mark.slow
    def test_produces_512_dim_embedding(self, ephemeral_client, tiny_image_path):
        clip_model, clip_processor = load_clip_model()
        _, image_col, _ = get_collections(ephemeral_client)

        embed_image(tiny_image_path, "img_001", image_col, clip_model, clip_processor)

        results = image_col.get(include=["embeddings"])
        assert len(results["embeddings"]) == 1
        assert len(results["embeddings"][0]) == 512

    @pytest.mark.slow
    def test_metadata_contains_thumbnail_b64(self, ephemeral_client, tiny_image_path):
        clip_model, clip_processor = load_clip_model()
        _, image_col, _ = get_collections(ephemeral_client)

        embed_image(tiny_image_path, "img_thumb", image_col, clip_model, clip_processor)

        results = image_col.get(include=["metadatas"])
        meta = results["metadatas"][0]
        assert "thumbnail_b64" in meta
        assert len(meta["thumbnail_b64"]) > 0

    @pytest.mark.slow
    def test_modality_is_image(self, ephemeral_client, tiny_image_path):
        clip_model, clip_processor = load_clip_model()
        _, image_col, _ = get_collections(ephemeral_client)

        embed_image(tiny_image_path, "img_modality", image_col, clip_model, clip_processor)

        results = image_col.get(include=["metadatas"])
        assert results["metadatas"][0]["modality"] == "image"

    @pytest.mark.slow
    def test_idempotency_no_duplication(self, ephemeral_client, tiny_image_path):
        clip_model, clip_processor = load_clip_model()
        _, image_col, _ = get_collections(ephemeral_client)

        embed_image(tiny_image_path, "img_idem", image_col, clip_model, clip_processor)
        count_after_first = image_col.count()
        embed_image(tiny_image_path, "img_idem", image_col, clip_model, clip_processor)
        count_after_second = image_col.count()

        assert count_after_first == count_after_second == 1

    def test_embed_image_with_mocked_clip(self, ephemeral_client, tiny_image_path, mocker):
        """Fast test: mock CLIP to avoid model download."""
        import torch
        mock_model = mocker.MagicMock()
        mock_processor = mocker.MagicMock()

        fake_features = torch.ones(1, 512)
        mock_model.get_image_features.return_value = fake_features
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}

        _, image_col, _ = get_collections(ephemeral_client)
        doc_id = embed_image(tiny_image_path, "img_mock", image_col, mock_model, mock_processor)

        assert doc_id == "img_img_mock"
        assert image_col.count() == 1


# ── extract_keyframes tests ───────────────────────────────────────────────────

def _create_test_video(path: str, num_frames: int = 10, fps: int = 5) -> None:
    """Write a synthetic AVI video to path using OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(path, fourcc, fps, (64, 64))
    for i in range(num_frames):
        frame = np.full((64, 64, 3), fill_value=(i * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestExtractKeyframes:
    @pytest.mark.slow
    def test_extracts_at_least_one_keyframe(self, ephemeral_client, tmp_path):
        video_path = str(tmp_path / "test.avi")
        _create_test_video(video_path, num_frames=15, fps=5)

        clip_model, clip_processor = load_clip_model()
        _, _, video_col = get_collections(ephemeral_client)

        doc_ids = extract_keyframes(
            video_path, "video_001", video_col, clip_model, clip_processor,
            sample_interval_sec=1,
        )
        assert len(doc_ids) >= 1

    @pytest.mark.slow
    def test_keyframe_metadata_has_timestamp(self, ephemeral_client, tmp_path):
        video_path = str(tmp_path / "test_ts.avi")
        _create_test_video(video_path, num_frames=20, fps=5)

        clip_model, clip_processor = load_clip_model()
        _, _, video_col = get_collections(ephemeral_client)

        extract_keyframes(
            video_path, "video_ts", video_col, clip_model, clip_processor,
            sample_interval_sec=1,
        )
        results = video_col.get(include=["metadatas"])
        for meta in results["metadatas"]:
            assert "timestamp_sec" in meta
            assert meta["modality"] == "video"
            assert "thumbnail_b64" in meta

    @pytest.mark.slow
    def test_idempotency_no_duplication(self, ephemeral_client, tmp_path):
        video_path = str(tmp_path / "test_idem.avi")
        _create_test_video(video_path, num_frames=10, fps=5)

        clip_model, clip_processor = load_clip_model()
        _, _, video_col = get_collections(ephemeral_client)

        extract_keyframes(
            video_path, "video_idem", video_col, clip_model, clip_processor,
            sample_interval_sec=1,
        )
        count_first = video_col.count()

        extract_keyframes(
            video_path, "video_idem", video_col, clip_model, clip_processor,
            sample_interval_sec=1,
        )
        count_second = video_col.count()

        assert count_first == count_second

    def test_invalid_video_path_raises(self, ephemeral_client, mocker):
        mock_model = mocker.MagicMock()
        mock_processor = mocker.MagicMock()
        _, _, video_col = get_collections(ephemeral_client)

        with pytest.raises(ValueError, match="cannot open video"):
            extract_keyframes(
                "/nonexistent/path.mp4", "bad_video",
                video_col, mock_model, mock_processor,
            )

    def test_extract_keyframes_with_mocked_clip(self, ephemeral_client, tmp_path, mocker):
        """Fast test: mock CLIP model to avoid download."""
        import torch
        video_path = str(tmp_path / "test_mock.avi")
        _create_test_video(video_path, num_frames=10, fps=5)

        mock_model = mocker.MagicMock()
        mock_processor = mocker.MagicMock()
        fake_features = torch.ones(1, 512)
        mock_model.get_image_features.return_value = fake_features
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}

        _, _, video_col = get_collections(ephemeral_client)
        doc_ids = extract_keyframes(
            video_path, "video_mock", video_col, mock_model, mock_processor,
            sample_interval_sec=1,
        )
        assert len(doc_ids) >= 1
        # IDs follow expected pattern
        for doc_id in doc_ids:
            assert doc_id.startswith("vid_video_mock_frame_")


# ── clip_embed_text test ──────────────────────────────────────────────────────

class TestClipEmbedText:
    @pytest.mark.slow
    def test_returns_512_dim_vector(self):
        clip_model, clip_processor = load_clip_model()
        embedding = clip_embed_text("a photo of a dog", clip_model, clip_processor)
        assert len(embedding) == 512

    @pytest.mark.slow
    def test_embedding_is_l2_normalised(self):
        clip_model, clip_processor = load_clip_model()
        embedding = clip_embed_text("outdoor scene", clip_model, clip_processor)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5
