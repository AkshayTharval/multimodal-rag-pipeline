# Multimodal RAG Pipeline — CLAUDE.md

## Project Context
A production-grade Retrieval Augmented Generation system handling text, images, and video
keyframes in a unified pipeline. Uses only free, local models — no API keys, no cost.
Built as a portfolio project demonstrating multimodal AI at production scale.

**Use case:** Media asset search — query "show me scenes with outdoor settings" and retrieve
relevant video keyframes, image stills, and associated text metadata.

---

## Phase Tracker
- [x] Phase 1: Project bootstrap (structure, CLAUDE.md, requirements.txt, venv)
- [ ] Phase 2: Ingestion pipeline (text, image, video)
- [ ] Phase 3: Hybrid retrieval (BM25 + dense + RRF + cross-encoder)
- [ ] Phase 4: Answer generation (Ollama, citations)
- [ ] Phase 5: Semantic cache + observability
- [ ] Phase 6: FastAPI service + batch CLI
- [ ] Phase 7: Streamlit dashboard (retrieval provenance UI)
- [ ] Phase 8: Integration tests
- [ ] Phase 9: Demo dataset + README + GitHub

---

## Architecture Overview

```
[Ingestion Pipeline]
  Text docs  → chunk (512 chars, 50 overlap) → SentenceTransformer embed (384-dim) → ChromaDB:text_chunks
  Images     → CLIP embed (512-dim)          → ChromaDB:image_embeddings
  Videos     → OpenCV keyframe extract       → CLIP embed (512-dim) → ChromaDB:video_keyframes

[Query Pipeline]
  User query → SentenceTransformer embed (text search) + CLIP text encode (image/video search)
             → BM25 (text_chunks) + dense cosine (all 3 collections)
             → RRF fusion (k=60)
             → Cross-encoder rerank (text results only)
             → Ollama llama3.2 (text) or llama3.2-vision (vision)
             → Answer + citations

[Observability]
  Prometheus metrics (custom registry) + OpenTelemetry tracing per pipeline stage
  Semantic cache (SQLite, cosine distance < 0.05 = hit)
  Streamlit dashboard with retrieval provenance UI
```

---

## CRITICAL: Embedding Dimension Split
**Text embeddings: 384-dim** (`all-MiniLM-L6-v2`)
**Image/Video embeddings: 512-dim** (`openai/clip-vit-base-patch32`)

These CANNOT share a ChromaDB collection. Separate collections are MANDATORY.

---

## ChromaDB Collections

| Collection | Model | Dimensions | Space |
|-----------|-------|-----------|-------|
| `text_chunks` | all-MiniLM-L6-v2 | 384 | cosine |
| `image_embeddings` | CLIP ViT-B/32 | 512 | cosine |
| `video_keyframes` | CLIP ViT-B/32 | 512 | cosine |

`hnsw:space` must be set at collection creation — it is immutable afterward.

---

## ChromaDB API (v1.5.5) — BREAKING CHANGE NOTE
**Use `PersistentClient` and `EphemeralClient`, NOT the old Settings-based API.**

```python
# Production
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")

# Tests (no disk I/O)
client = chromadb.EphemeralClient()

# Collection creation
col = client.get_or_create_collection(
    "text_chunks",
    metadata={"hnsw:space": "cosine"}
)
```

---

## Technology Choices

| Component | Library / Model | Reason |
|-----------|----------------|--------|
| Text embedding | `sentence-transformers/all-MiniLM-L6-v2` | Free, local, 384-dim, well-known |
| Image/Video embedding | `openai/clip-vit-base-patch32` | Open source, 512-dim, standard baseline |
| Vector store | ChromaDB 1.5.5 (`PersistentClient`) | Local, zero-cost, production-tested |
| Sparse retrieval | `rank-bm25` (BM25Okapi) | Functional for portfolio scale |
| Reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` | State-of-the-art text reranker |
| Text LLM | `llama3.2` via Ollama | 128K context, free, local |
| Vision LLM | `llama3.2-vision` via Ollama | Better OCR/charts than llava |
| Video processing | OpenCV (`opencv-python-headless`) | Keyframe extraction |
| API | FastAPI 0.135.2 | Async, fast, production-standard |
| Dashboard | Streamlit | Rapid multimodal UI |
| Metrics | Prometheus client (custom registry) | Production observability |
| Tracing | OpenTelemetry SDK | Per-stage latency spans |
| Cache | SQLite (stdlib) + cosine distance | Zero-dep, sufficient for portfolio |

---

## Ollama Setup
```bash
# Install
brew install ollama

# In separate terminal
ollama serve

# Pull models
ollama pull llama3.2          # ~2GB, 128K context
ollama pull llama3.2-vision   # ~8GB, vision + text
```

Ollama base URL: `http://localhost:11434`

---

## Code Conventions

### Type Hints
All public functions must have complete type annotations:
```python
def chunk_text(
    text: str,
    source_id: str,
    collection: chromadb.Collection,
    embedder: SentenceTransformer,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[str]:
```

### Docstrings
All public functions require docstrings:
```python
def rrf_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each list is a ranked list of result dicts with 'id' and 'rank' keys.
        k: RRF constant (default 60). Higher k reduces impact of rank differences.

    Returns:
        Single merged ranked list sorted by descending RRF score.
        Each item has 'rrf_score' added to its dict.
    """
```

### Dependency Injection
Never instantiate models or DB clients inside functions. Always accept them as parameters.
This makes unit testing without external services trivial.

```python
# CORRECT
def embed_image(image_path: str, source_id: str,
                collection: chromadb.Collection,
                clip_model: CLIPModel,
                clip_processor: CLIPProcessor) -> None:
    ...

# WRONG — creates hidden dependencies, untestable
def embed_image(image_path: str, source_id: str) -> None:
    client = chromadb.PersistentClient(path="./chroma_db")  # DON'T DO THIS
    model = CLIPModel.from_pretrained(...)                   # DON'T DO THIS
```

### Error Handling
- Ingestion failures: log and skip (never crash batch)
- Retrieval failures: raise `HTTPException(500)` in API layer
- Ollama unreachable: return structured error dict, not unhandled exception
- Cache failures: log and continue (cache is non-fatal)

---

## Testing Conventions

### What to Mock
- Ollama HTTP calls — always mock with `httpx.AsyncMock`
- ChromaDB — use `EphemeralClient()` (real, but in-memory)
- File system — use `tmp_path` pytest fixture
- Prometheus registry — use custom `CollectorRegistry()` per test to avoid conflicts

### What NOT to Mock
- ChromaDB operations (use EphemeralClient — it's real code, in-memory)
- `numpy` / `sklearn` computations
- PIL image operations
- SQLite (use `:memory:` or `tmp_path`)

### Test Execution
```bash
# All tests (no external services required)
pytest tests/ -v

# Skip slow tests (CLIP model downloads)
pytest tests/ -v -m "not slow"

# Integration tests only
pytest tests/ -v -m integration
```

### Pytest Marks
- `@pytest.mark.slow` — tests that download ML models (CLIP, SentenceTransformer)
- `@pytest.mark.integration` — end-to-end tests using EphemeralClient
- `@pytest.mark.asyncio` — async test functions

---

## Performance Targets
- Text query end-to-end: **< 3 seconds**
- Vision query end-to-end: **< 8 seconds**
- Video keyframe extraction: 60s video in **< 30 seconds**
- BM25 known limitation: index rebuilt in-memory each query (O(n)). Acceptable for portfolio.

---

## Running the Stack
```bash
# Virtual environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Start Ollama (separate terminal)
ollama serve

# Start API server (separate terminal)
uvicorn api.app:app --reload --port 8000

# Start dashboard (separate terminal)
streamlit run dashboard/app.py

# Batch ingest demo data
python cli/download_demo.py
python cli/ingest_cli.py data/

# Query via curl
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what is retrieval augmented generation?", "top_k": 5}'
```

---

## Key Algorithms

### RRF Fusion
```
score(d) = Σ_i  1 / (k + rank_i(d))
```
k=60, summed over all ranked lists. Documents not present in a list contribute 0.

### Semantic Cache Threshold
`threshold = 0.05` is a **cosine DISTANCE** threshold (not similarity).
- cosine_distance = 1 - cosine_similarity
- distance < 0.05 → similarity > 0.95 → cache HIT
- distance ≥ 0.05 → cache MISS

### Cross-Encoder Reranking
Only text candidates are passed to the cross-encoder — it cannot score images.
Image and video results are appended after reranked text results.

---

## Continuing From a Previous Session
1. `source venv/bin/activate`
2. Check phase tracker at top of this file — find the first unchecked phase
3. Read the relevant test file to understand what's expected before writing code
4. Run `pytest tests/ -v -m "not slow"` to see current test status
5. Implement, run tests, mark phase complete in tracker above
