"""Prometheus metrics for the RAG pipeline.

Uses a custom CollectorRegistry (not the default global registry) to prevent
"Duplicated timeseries" errors when tests import this module multiple times
in the same process.

Usage:
    from observability.metrics import (
        QUERY_COUNT, CACHE_HITS, RETRIEVAL_LATENCY, record_prometheus_metrics
    )

    QUERY_COUNT.labels(status="success").inc()
    with RETRIEVAL_LATENCY.time():
        results = hybrid_retrieve(...)
"""

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Custom registry — avoids conflicts when running pytest with multiple test modules
REGISTRY = CollectorRegistry()

# ── Counters ──────────────────────────────────────────────────────────────────

QUERY_COUNT = Counter(
    "rag_queries_total",
    "Total number of queries processed by the pipeline",
    ["status"],      # labels: "success", "cache_hit", "error"
    registry=REGISTRY,
)

CACHE_HITS = Counter(
    "rag_cache_hits_total",
    "Number of semantic cache hits",
    registry=REGISTRY,
)

CACHE_MISSES = Counter(
    "rag_cache_misses_total",
    "Number of semantic cache misses",
    registry=REGISTRY,
)

INGEST_COUNT = Counter(
    "rag_ingested_documents_total",
    "Total number of documents ingested",
    ["modality"],    # labels: "text", "image", "video"
    registry=REGISTRY,
)

# ── Histograms ────────────────────────────────────────────────────────────────

EMBED_LATENCY = Histogram(
    "rag_embed_latency_seconds",
    "Time spent embedding queries",
    buckets=[0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    registry=REGISTRY,
)

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Time spent on hybrid retrieval (BM25 + dense + RRF)",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    registry=REGISTRY,
)

RERANK_LATENCY = Histogram(
    "rag_rerank_latency_seconds",
    "Time spent on cross-encoder reranking",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0],
    registry=REGISTRY,
)

GENERATION_LATENCY = Histogram(
    "rag_generation_latency_seconds",
    "Time spent on LLM answer generation",
    buckets=[1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 180.0],
    registry=REGISTRY,
)

TOTAL_LATENCY = Histogram(
    "rag_total_query_latency_seconds",
    "End-to-end query latency",
    buckets=[0.5, 1.0, 3.0, 5.0, 8.0, 15.0, 30.0],
    registry=REGISTRY,
)


def get_prometheus_output() -> tuple[bytes, str]:
    """Generate Prometheus text-format metrics output.

    Returns:
        Tuple of (metrics_bytes, content_type_string).
    """
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def get_metrics_summary() -> dict:
    """Return a JSON-serialisable metrics summary for the /stats endpoint.

    Returns:
        Dict with current metric values.
    """
    def _counter_value(counter) -> float:
        try:
            return sum(sample.value for sample in counter.collect()[0].samples)
        except (IndexError, AttributeError):
            return 0.0

    def _histogram_sum(histogram) -> float:
        try:
            samples = histogram.collect()[0].samples
            for sample in samples:
                if sample.name.endswith("_sum"):
                    return sample.value
            return 0.0
        except (IndexError, AttributeError):
            return 0.0

    def _histogram_count(histogram) -> float:
        try:
            samples = histogram.collect()[0].samples
            for sample in samples:
                if sample.name.endswith("_count"):
                    return sample.value
            return 0.0
        except (IndexError, AttributeError):
            return 0.0

    total_queries = _counter_value(QUERY_COUNT)
    cache_hits = _counter_value(CACHE_HITS)
    cache_misses = _counter_value(CACHE_MISSES)

    retrieval_count = _histogram_count(RETRIEVAL_LATENCY)
    retrieval_sum = _histogram_sum(RETRIEVAL_LATENCY)
    generation_count = _histogram_count(GENERATION_LATENCY)
    generation_sum = _histogram_sum(GENERATION_LATENCY)

    return {
        "total_queries": total_queries,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": (
            round(cache_hits / (cache_hits + cache_misses), 3)
            if (cache_hits + cache_misses) > 0
            else 0.0
        ),
        "avg_retrieval_latency_ms": (
            round(retrieval_sum / retrieval_count * 1000, 1)
            if retrieval_count > 0
            else None
        ),
        "avg_generation_latency_ms": (
            round(generation_sum / generation_count * 1000, 1)
            if generation_count > 0
            else None
        ),
    }
