"""FastAPI application for the Multimodal RAG Pipeline.

Endpoints:
    POST /ingest    — ingest a file (text, image, or video)
    POST /query     — query the pipeline, returns answer + retrieved sources
    GET  /stats     — pipeline statistics (JSON)
    GET  /health    — health check (ChromaDB + Ollama)
    GET  /metrics   — Prometheus metrics (text/plain)

All shared resources (ChromaDB client, models, cache) are initialised once
in the lifespan context manager and stored in app.state.
"""

import logging
import mimetypes
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

import chromadb
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from observability.metrics import (
    CACHE_HITS,
    CACHE_MISSES,
    GENERATION_LATENCY,
    INGEST_COUNT,
    QUERY_COUNT,
    RETRIEVAL_LATENCY,
    TOTAL_LATENCY,
    get_metrics_summary,
    get_prometheus_output,
)
from observability.tracing import trace_stage
from pipeline import get_chroma_client, get_collections
from pipeline.cache import SemanticCache
from pipeline.config import (
    CACHE_DB_PATH,
    CHROMA_DB_PATH,
    CLIP_MODEL_ID,
    OLLAMA_BASE_URL,
    RERANKER_MODEL,
    TEXT_EMBED_MODEL,
)
from pipeline.generate import OllamaClient, generate_answer
from pipeline.ingest import (
    chunk_text,
    embed_image,
    extract_keyframes,
    load_clip_model,
    load_text_embedder,
)
from pipeline.retrieve import hybrid_retrieve

logger = logging.getLogger(__name__)


# ── Lifespan: initialise / tear down shared resources ────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise all shared resources on startup."""
    logger.info("Loading models and connecting to ChromaDB...")

    app.state.chroma_client = get_chroma_client(CHROMA_DB_PATH)
    (
        app.state.text_col,
        app.state.image_col,
        app.state.video_col,
    ) = get_collections(app.state.chroma_client)

    app.state.text_embedder = load_text_embedder(TEXT_EMBED_MODEL)
    app.state.clip_model, app.state.clip_processor = load_clip_model(CLIP_MODEL_ID)
    app.state.cross_encoder = CrossEncoder(RERANKER_MODEL)
    app.state.ollama_client = OllamaClient(base_url=OLLAMA_BASE_URL)
    app.state.cache = SemanticCache(
        db_path=CACHE_DB_PATH,
        embedder=app.state.text_embedder,  # reuse loaded embedder
    )

    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Multimodal RAG Pipeline",
    description="Production-grade RAG for text, images, and video keyframes.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_cache: bool = True


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    cache_hit: bool
    retrieval_results: list[dict]
    latency_ms: float
    error: str | None = None


class IngestResponse(BaseModel):
    status: str
    document_id: str
    chunks_created: int
    modality: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    modality: str = Form(default="auto"),
) -> IngestResponse:
    """Ingest a file into the RAG pipeline.

    Supports text (.txt), images (.jpg, .png, .webp), and videos (.mp4, .avi, .mov).
    Set ``modality="auto"`` to detect from the file's content type.
    """
    import tempfile
    import os

    content_type = file.content_type or ""
    filename = file.filename or "unknown"

    detected = _detect_modality(filename, content_type, modality)

    # Save upload to a temp file (needed by OpenCV and Pillow)
    suffix = "." + (filename.rsplit(".", 1)[-1] if "." in filename else "bin")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        source_id = filename

        if detected == "text":
            with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            chunks = chunk_text(
                text, source_id, app.state.text_col, app.state.text_embedder
            )
            INGEST_COUNT.labels(modality="text").inc()
            return IngestResponse(
                status="ok",
                document_id=f"{source_id}_chunk_0",
                chunks_created=len(chunks),
                modality="text",
            )

        elif detected == "image":
            doc_id = embed_image(
                tmp_path, source_id,
                app.state.image_col,
                app.state.clip_model,
                app.state.clip_processor,
            )
            INGEST_COUNT.labels(modality="image").inc()
            return IngestResponse(
                status="ok",
                document_id=doc_id,
                chunks_created=1,
                modality="image",
            )

        elif detected == "video":
            doc_ids = extract_keyframes(
                tmp_path, source_id,
                app.state.video_col,
                app.state.clip_model,
                app.state.clip_processor,
            )
            INGEST_COUNT.labels(modality="video").inc()
            return IngestResponse(
                status="ok",
                document_id=doc_ids[0] if doc_ids else f"vid_{source_id}_frame_0",
                chunks_created=len(doc_ids),
                modality="video",
            )

        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported modality '{detected}'. Expected text, image, or video.",
            )
    finally:
        os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse)
async def query_pipeline(request: QueryRequest) -> QueryResponse:
    """Query the RAG pipeline.

    Checks the semantic cache first. On a miss, runs full hybrid retrieval +
    reranking + Ollama generation, then stores the result in the cache.
    """
    t0 = time.perf_counter()

    # 1. Cache lookup
    if request.use_cache:
        with trace_stage("cache_lookup", {"query": request.query[:80]}):
            cached = app.state.cache.lookup(request.query)

        if cached:
            CACHE_HITS.inc()
            QUERY_COUNT.labels(status="cache_hit").inc()
            latency_ms = (time.perf_counter() - t0) * 1000
            return QueryResponse(
                answer=cached["answer"],
                citations=cached["citations"],
                cache_hit=True,
                retrieval_results=cached["retrieval_results"],
                latency_ms=round(latency_ms, 1),
            )

        CACHE_MISSES.inc()

    # 2. Hybrid retrieval
    with trace_stage("retrieval", {"query": request.query[:80], "top_k": request.top_k}):
        with RETRIEVAL_LATENCY.time():
            retrieval_results = hybrid_retrieve(
                query_text=request.query,
                text_collection=app.state.text_col,
                image_collection=app.state.image_col,
                video_collection=app.state.video_col,
                text_embedder=app.state.text_embedder,
                clip_model=app.state.clip_model,
                clip_processor=app.state.clip_processor,
                cross_encoder=app.state.cross_encoder,
                top_k=request.top_k,
            )

    # 3. Answer generation
    with trace_stage("generation", {"query": request.query[:80]}):
        with GENERATION_LATENCY.time():
            result = await generate_answer(
                query=request.query,
                retrieval_results=retrieval_results,
                ollama_client=app.state.ollama_client,
            )

    latency_ms = (time.perf_counter() - t0) * 1000
    TOTAL_LATENCY.observe(latency_ms / 1000)

    # 4. Cache store (only on success)
    if request.use_cache and "error" not in result:
        app.state.cache.store(request.query, result)

    status = "error" if "error" in result else "success"
    QUERY_COUNT.labels(status=status).inc()

    return QueryResponse(
        answer=result["answer"],
        citations=result.get("citations", []),
        cache_hit=False,
        retrieval_results=result.get("retrieval_results", []),
        latency_ms=round(latency_ms, 1),
        error=result.get("error"),
    )


@app.get("/stats")
async def get_stats() -> dict[str, Any]:
    """Return pipeline statistics as JSON."""
    summary = get_metrics_summary()
    summary["text_chunks"] = app.state.text_col.count()
    summary["image_embeddings"] = app.state.image_col.count()
    summary["video_keyframes"] = app.state.video_col.count()
    summary["cached_queries"] = app.state.cache.count()
    return summary


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Check health of ChromaDB and Ollama."""
    result: dict[str, str] = {"status": "healthy"}

    # ChromaDB
    try:
        app.state.chroma_client.heartbeat()
        result["chroma"] = "ok"
    except Exception as exc:
        result["chroma"] = f"error: {exc}"
        result["status"] = "degraded"

    # Ollama
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            result["ollama"] = "ok" if resp.status_code == 200 else f"http_{resp.status_code}"
    except (httpx.ConnectError, httpx.TimeoutException):
        result["ollama"] = "unreachable"

    return result


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    """Expose Prometheus metrics in text format."""
    output, content_type = get_prometheus_output()
    return Response(content=output, media_type=content_type)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_modality(filename: str, content_type: str, hint: str) -> str:
    """Detect file modality from hint, content_type, or filename extension.

    Args:
        filename: Original filename.
        content_type: MIME type from the upload.
        hint: User-provided modality hint ("auto", "text", "image", "video").

    Returns:
        One of "text", "image", "video", or "unknown".
    """
    if hint != "auto":
        return hint

    if content_type.startswith("text/"):
        return "text"
    if content_type.startswith("image/"):
        return "image"
    if content_type.startswith("video/"):
        return "video"

    # Fall back to extension
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in ("txt", "md", "rst", "csv", "json"):
        return "text"
    if ext in ("jpg", "jpeg", "png", "webp", "gif", "bmp", "tiff"):
        return "image"
    if ext in ("mp4", "avi", "mov", "mkv", "webm", "flv"):
        return "video"

    return "unknown"
