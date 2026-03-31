"""Download demo dataset for the Multimodal RAG Pipeline.

Downloads:
  - 5 Wikipedia-based text excerpts (written inline, no network for text)
  - 5 CC0 images from Wikimedia Commons
  - 1 public-domain short video from Wikimedia Commons

Usage:
    python cli/download_demo.py
    python cli/download_demo.py --output-dir ./my_data
"""

import argparse
import sys
from pathlib import Path

import httpx

# ── Text content (inline — no download needed) ────────────────────────────────

TEXT_DOCS = {
    "rag_overview.txt": (
        "Retrieval Augmented Generation (RAG) is a machine learning technique that "
        "enhances large language model (LLM) outputs by incorporating information "
        "retrieved from external knowledge sources. Unlike purely parametric models "
        "that rely solely on knowledge encoded during training, RAG systems dynamically "
        "fetch relevant documents at inference time. This allows them to provide "
        "up-to-date information, cite sources, and reduce hallucination. "
        "The retrieval step typically uses dense vector search, sparse BM25, or "
        "a combination of both. The retrieved context is appended to the user query "
        "before being passed to the language model, which generates a grounded answer. "
        "RAG was introduced by Lewis et al. (2020) in 'Retrieval-Augmented Generation "
        "for Knowledge-Intensive NLP Tasks' and has since become a standard pattern "
        "for production AI applications."
    ),
    "vector_databases.txt": (
        "Vector databases are specialized data stores designed to efficiently index "
        "and retrieve high-dimensional embedding vectors. Unlike traditional relational "
        "databases that use exact matching, vector databases support approximate nearest "
        "neighbor (ANN) search, returning the most semantically similar items to a query. "
        "Common indexing algorithms include HNSW (Hierarchical Navigable Small World), "
        "IVF (Inverted File Index), and FAISS flat indices. "
        "Popular vector databases include ChromaDB, Pinecone, Weaviate, Qdrant, and Milvus. "
        "ChromaDB is an open-source, embeddable vector database that supports cosine, "
        "L2, and inner-product similarity metrics. It can run in-memory (EphemeralClient) "
        "or persist to disk (PersistentClient) and supports metadata filtering. "
        "Vector databases are the backbone of RAG systems, enabling sub-second semantic "
        "search over millions of document chunks."
    ),
    "clip_model.txt": (
        "CLIP (Contrastive Language-Image Pretraining) is a neural network model "
        "developed by OpenAI that learns visual concepts from natural language supervision. "
        "It is trained on 400 million (image, text) pairs from the internet using a "
        "contrastive objective: image and text encoders are trained so that matching "
        "pairs have high cosine similarity in a shared embedding space, while "
        "non-matching pairs have low similarity. "
        "The ViT-B/32 variant uses a Vision Transformer with 32-pixel patch size and "
        "produces 512-dimensional embeddings for both images and text. "
        "This shared embedding space enables zero-shot image classification, "
        "cross-modal retrieval (text queries → image results), and image captioning. "
        "In multimodal RAG systems, CLIP is used to embed both images and text queries "
        "into the same 512-dim space, enabling text-to-image retrieval without "
        "explicit image captions or labels."
    ),
    "hybrid_retrieval.txt": (
        "Hybrid retrieval combines sparse and dense retrieval methods to improve "
        "recall and precision in information retrieval systems. "
        "Sparse retrieval methods, such as BM25 (Best Match 25), score documents "
        "based on term frequency and inverse document frequency. They excel at "
        "exact keyword matching and are computationally efficient, but fail on "
        "paraphrasing or semantic equivalence. "
        "Dense retrieval methods use neural embeddings to capture semantic meaning. "
        "A query and documents are encoded into a shared vector space, and "
        "approximate nearest neighbor search finds semantically similar documents "
        "even when exact keywords don't match. "
        "Reciprocal Rank Fusion (RRF) is a simple but effective technique for "
        "combining multiple ranked lists. The RRF score for document d is "
        "sum(1/(k + rank_i(d))) across all ranked lists i, where k=60 by default. "
        "Cross-encoder reranking further refines the top candidates by scoring "
        "query-document pairs jointly using a transformer model."
    ),
    "observability.txt": (
        "Observability in AI systems refers to the ability to understand internal "
        "system behavior from external outputs. For RAG pipelines, key observability "
        "dimensions include latency (per-stage timing for retrieval, reranking, "
        "and generation), cache performance (hit rate, average cache distance), "
        "retrieval quality (MRR, recall@k, precision@k), and error rates. "
        "Prometheus is an open-source monitoring toolkit that collects time-series "
        "metrics via pull-based scraping. Metrics are defined as counters (monotonically "
        "increasing), gauges (arbitrary up/down values), and histograms (value "
        "distribution with configurable buckets). "
        "OpenTelemetry is a vendor-neutral observability framework for distributed "
        "tracing, metrics, and logging. It provides context propagation across "
        "service boundaries, enabling end-to-end traces of multi-stage pipelines. "
        "Combining Prometheus metrics with OpenTelemetry traces gives a complete "
        "picture of pipeline performance in production."
    ),
}

# ── CC0 images from Wikimedia Commons ─────────────────────────────────────────
# All images are in the public domain or CC0 license.

IMAGE_URLS = [
    (
        "mountain_landscape.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/320px-24701-nature-natural-beauty.jpg",
    ),
    (
        "sunset_ocean.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/GoldenGateBridge-001.jpg/320px-GoldenGateBridge-001.jpg",
    ),
    (
        "forest_path.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Forest_Path.jpg/320px-Forest_Path.jpg",
    ),
    (
        "city_skyline.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Southwest_corner_of_Central_Park%2C_looking_east%2C_NYC.jpg/320px-Southwest_corner_of_Central_Park%2C_looking_east%2C_NYC.jpg",
    ),
    (
        "abstract_colors.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
    ),
]

# ── Public-domain short video from Wikimedia Commons ─────────────────────────

VIDEO_URLS = [
    (
        "big_buck_bunny_clip.webm",
        "https://upload.wikimedia.org/wikipedia/commons/transcoded/c/c0/Big_Buck_Bunny_4K.webm/Big_Buck_Bunny_4K.webm.360p.webm",
    ),
]


def write_text_files(output_dir: Path) -> None:
    """Write inline text documents to disk."""
    text_dir = output_dir / "texts"
    text_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in TEXT_DOCS.items():
        path = text_dir / filename
        path.write_text(content, encoding="utf-8")
        print(f"  ✓ {path}")


def download_file(url: str, dest: Path, client: httpx.Client) -> bool:
    """Download a single file with progress output. Returns True on success."""
    try:
        with client.stream("GET", url, follow_redirects=True) as resp:
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)
        print(f"  ✓ {dest.name} ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as exc:
        print(f"  ✗ {dest.name}: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download demo dataset for the Multimodal RAG Pipeline."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Root directory for downloaded files (default: ./data).",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image downloads.",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip video download (video is large, ~50 MB).",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Text files (no network) ───────────────────────────────────────────────
    print(f"\nWriting {len(TEXT_DOCS)} text documents → {output_dir}/texts/")
    write_text_files(output_dir)

    errors = 0

    with httpx.Client(timeout=60.0) as client:
        # ── Images ───────────────────────────────────────────────────────────
        if not args.skip_images:
            print(f"\nDownloading {len(IMAGE_URLS)} images → {output_dir}/images/")
            img_dir = output_dir / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            for filename, url in IMAGE_URLS:
                ok = download_file(url, img_dir / filename, client)
                if not ok:
                    errors += 1
        else:
            print("\nSkipping images (--skip-images).")

        # ── Video ─────────────────────────────────────────────────────────────
        if not args.skip_video:
            print(f"\nDownloading {len(VIDEO_URLS)} video(s) → {output_dir}/videos/")
            print("  (video may be large — use --skip-video to skip)")
            vid_dir = output_dir / "videos"
            vid_dir.mkdir(parents=True, exist_ok=True)
            for filename, url in VIDEO_URLS:
                ok = download_file(url, vid_dir / filename, client)
                if not ok:
                    errors += 1
        else:
            print("\nSkipping video (--skip-video).")

    print(f"\nDone. Errors: {errors}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
