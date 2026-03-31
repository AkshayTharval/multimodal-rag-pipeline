"""Batch ingestion CLI.

Usage:
    python cli/ingest_cli.py data/
    python cli/ingest_cli.py data/texts/ data/images/ --modality auto
    python cli/ingest_cli.py path/to/file.txt --chroma-path ./chroma_db
"""

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from pipeline import get_chroma_client, get_collections
from pipeline.config import CHROMA_DB_PATH, CLIP_MODEL_ID, TEXT_EMBED_MODEL
from pipeline.ingest import (
    chunk_text,
    embed_image,
    extract_keyframes,
    load_clip_model,
    load_text_embedder,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def detect_modality(path: Path, hint: str) -> str:
    """Detect modality from file extension or hint."""
    if hint != "auto":
        return hint
    ext = path.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return "text"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return "unknown"


def collect_files(paths: list[Path], modality_hint: str) -> list[tuple[Path, str]]:
    """Walk paths and collect (file, modality) pairs."""
    files: list[tuple[Path, str]] = []
    for path in paths:
        if path.is_file():
            mod = detect_modality(path, modality_hint)
            if mod != "unknown":
                files.append((path, mod))
            else:
                logger.warning("Skipping unknown file type: %s", path)
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    mod = detect_modality(child, modality_hint)
                    if mod != "unknown":
                        files.append((child, mod))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch ingest files into the Multimodal RAG pipeline."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to ingest.",
    )
    parser.add_argument(
        "--modality",
        choices=["auto", "text", "image", "video"],
        default="auto",
        help="Force a specific modality (default: auto-detect from extension).",
    )
    parser.add_argument(
        "--chroma-path",
        default=CHROMA_DB_PATH,
        help=f"Path to ChromaDB storage (default: {CHROMA_DB_PATH}).",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=5,
        help="Seconds between video keyframe samples (default: 5).",
    )
    args = parser.parse_args()

    # Collect files
    files = collect_files(args.paths, args.modality)
    if not files:
        print("No supported files found.")
        sys.exit(0)

    print(f"Found {len(files)} file(s) to ingest. Loading models...")

    # Load models
    text_embedder = load_text_embedder(TEXT_EMBED_MODEL)
    clip_model, clip_processor = load_clip_model(CLIP_MODEL_ID)

    # Connect to ChromaDB
    client = get_chroma_client(args.chroma_path)
    text_col, image_col, video_col = get_collections(client)

    # Stats
    stats = {"text": 0, "image": 0, "video": 0, "skipped": 0, "errors": 0}

    with tqdm(files, desc="Ingesting", unit="file") as pbar:
        for file_path, modality in pbar:
            pbar.set_postfix_str(f"{file_path.name} ({modality})")
            try:
                source_id = file_path.name

                if modality == "text":
                    text = file_path.read_text(encoding="utf-8", errors="replace")
                    chunks = chunk_text(text, source_id, text_col, text_embedder)
                    stats["text"] += 1
                    tqdm.write(f"  ✓ {file_path.name} → {len(chunks)} chunks")

                elif modality == "image":
                    embed_image(
                        str(file_path), source_id, image_col, clip_model, clip_processor
                    )
                    stats["image"] += 1
                    tqdm.write(f"  ✓ {file_path.name} → 1 embedding")

                elif modality == "video":
                    doc_ids = extract_keyframes(
                        str(file_path), source_id, video_col,
                        clip_model, clip_processor,
                        sample_interval_sec=args.sample_interval,
                    )
                    stats["video"] += 1
                    tqdm.write(f"  ✓ {file_path.name} → {len(doc_ids)} keyframes")

            except Exception as exc:
                stats["errors"] += 1
                tqdm.write(f"  ✗ {file_path.name}: {exc}")

    print(
        f"\nDone. text={stats['text']}, image={stats['image']}, "
        f"video={stats['video']}, errors={stats['errors']}"
    )


if __name__ == "__main__":
    main()
