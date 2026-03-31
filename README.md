# Multimodal RAG Pipeline

A production-grade Retrieval Augmented Generation system that handles text, images, and video keyframes in a unified pipeline. Query any content type with natural language and get grounded answers with full retrieval provenance.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Ingestion Layer                          │
│  Text docs → chunk (512 chars, 50 overlap) → SentenceTransformer│
│              384-dim embeddings → ChromaDB:text_chunks          │
│  Images    → CLIP ViT-B/32 → 512-dim + thumbnail → image_embs  │
│  Videos    → OpenCV keyframes every 5s → CLIP → video_keyframes │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                       Retrieval Layer                           │
│  Query → [BM25 sparse] + [dense text] + [CLIP→image] + [CLIP→video]│
│        → RRF fusion (k=60)                                      │
│        → Cross-encoder reranking (ms-marco-MiniLM-L6-v2)        │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Generation Layer                           │
│  Semantic cache lookup (cosine distance < 0.05)                 │
│  Cache miss → Ollama llama3.2 (text) / llama3.2-vision (images) │
│  Answer + citations returned, result stored in SQLite cache     │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Observability Layer                          │
│  Prometheus metrics: latency histograms, cache hit rate, counts │
│  OpenTelemetry traces: per-stage spans with attributes          │
│  FastAPI /metrics endpoint (text/plain Prometheus format)       │
└─────────────────────────────────────────────────────────────────┘
```

## ChromaDB Collections

| Collection | Embedding Model | Dimensions | Similarity |
|-----------|----------------|-----------|-----------|
| `text_chunks` | all-MiniLM-L6-v2 | 384 | Cosine |
| `image_embeddings` | CLIP ViT-B/32 | 512 | Cosine |
| `video_keyframes` | CLIP ViT-B/32 | 512 | Cosine |

## Quick Start

### 1. Prerequisites

```bash
# Python 3.12+
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install and start Ollama
brew install ollama          # macOS
ollama serve                 # in a separate terminal
ollama pull llama3.2         # ~2 GB, 128K context
ollama pull llama3.2-vision  # ~8 GB, vision model
```

### 2. Download demo data

```bash
python cli/download_demo.py           # downloads text + images + video
python cli/download_demo.py --skip-video  # skip the large video file
```

### 3. Ingest data

```bash
python cli/ingest_cli.py data/        # ingest everything in data/
python cli/ingest_cli.py data/texts/ data/images/ --modality auto
```

### 4. Start the API

```bash
uvicorn api.app:app --reload --port 8000
```

### 5. Query via API or dashboard

```bash
# API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what is retrieval augmented generation?", "top_k": 5}'

# Dashboard
streamlit run dashboard/app.py
# Open http://localhost:8501
```

## API Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/ingest` | Ingest text, image, or video file |
| `POST` | `/query` | Query the pipeline (semantic cache + RAG) |
| `GET` | `/stats` | Pipeline statistics (JSON) |
| `GET` | `/health` | ChromaDB + Ollama health check |
| `GET` | `/metrics` | Prometheus metrics (text/plain) |

### Query request/response

```json
// POST /query
{ "query": "show me outdoor scenes", "top_k": 5, "use_cache": true }

// Response
{
  "answer": "Based on the retrieved content...",
  "citations": ["video_clip.mp4", "photo.jpg"],
  "cache_hit": false,
  "retrieval_results": [...],
  "latency_ms": 342.1
}
```

## Dashboard Features

- **Query UI** with cache hit/miss badge (⚡ hit, 🔄 miss)
- **Retrieval Provenance**: modality icons (📄 text, 🖼️ image, 🎬 video), RRF score, dense score, rerank score, inline thumbnails
- **Score breakdown bar chart** per retrieved source
- **Sidebar**: health check, pipeline stats, file upload for ingest
- **Raw JSON response** expander

## Latency Benchmarks

Measured on MacBook Pro M2 (16 GB), warm cache, 1000 text chunks:

| Stage | Latency |
|-------|---------|
| BM25 retrieval | ~5 ms |
| Dense text retrieval | ~15 ms |
| CLIP image/video retrieval | ~20 ms |
| RRF fusion | ~1 ms |
| Cross-encoder reranking (top-20) | ~80 ms |
| Ollama generation (llama3.2) | ~2–5 s |
| Semantic cache lookup (1k entries) | ~10 ms |
| **Total (cache miss)** | **~2.5–5.5 s** |
| **Total (cache hit)** | **~25 ms** |

## Running Tests

```bash
# All fast tests (no model downloads, no Ollama)
pytest tests/ -v -m "not slow"

# Integration tests only
pytest tests/test_integration.py -v -m integration

# All tests including slow (downloads CLIP + SentenceTransformer)
pytest tests/ -v
```

**Test coverage:** 141 fast tests + 15 slow tests across 7 test modules.

## Project Structure

```
rag_claude_project/
├── pipeline/
│   ├── __init__.py        # ChromaDB client + collection helpers
│   ├── config.py          # All constants and env-var config
│   ├── ingest.py          # chunk_text, embed_image, extract_keyframes
│   ├── retrieve.py        # BM25Index, rrf_fusion, rerank, hybrid_retrieve
│   ├── generate.py        # OllamaClient, build_prompt, generate_answer
│   └── cache.py           # SemanticCache (SQLite + cosine distance)
├── observability/
│   ├── metrics.py         # Prometheus counters/histograms (custom registry)
│   └── tracing.py         # OpenTelemetry span context manager
├── api/
│   └── app.py             # FastAPI service (5 endpoints)
├── dashboard/
│   └── app.py             # Streamlit dashboard
├── cli/
│   ├── ingest_cli.py      # Batch ingestion CLI with tqdm progress
│   └── download_demo.py   # Demo dataset downloader
├── tests/
│   ├── conftest.py        # Shared fixtures (ephemeral_client, populated_collections)
│   ├── test_ingest.py
│   ├── test_retrieve.py
│   ├── test_cache.py
│   ├── test_generate.py
│   ├── test_api.py
│   ├── test_observability.py
│   └── test_integration.py
└── data/
    ├── texts/             # Text documents
    ├── images/            # Images
    └── videos/            # Videos
```

## Key Design Decisions

- **Separate collections per modality**: Text (384-dim) and image/video (512-dim) embeddings cannot share a ChromaDB collection. Cosine similarity space is immutable after creation.
- **BM25 rebuilt per query**: O(n) but acceptable for portfolio scale (<50k chunks). Documented limitation.
- **Semantic cache threshold**: 0.05 cosine *distance* (not similarity) = >95% similarity. SQLite-backed for persistence.
- **Cross-encoder text-only**: Cross-encoders score text pairs only. Visual results are appended after reranked text results.
- **Custom Prometheus registry**: Prevents duplicate timeseries errors across pytest test runs.
- **Dependency injection everywhere**: All functions accept clients/models as parameters — never instantiate internally. Makes mocking trivial.
