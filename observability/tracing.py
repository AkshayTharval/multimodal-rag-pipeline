"""OpenTelemetry tracing for the RAG pipeline.

Provides a context manager for instrumenting pipeline stages with named spans.
Uses InMemorySpanExporter by default for development/testing; swap for an
OTLP exporter in production.

Usage:
    from observability.tracing import trace_stage

    with trace_stage("retrieval", {"query": query_text, "top_k": 5}):
        results = hybrid_retrieve(...)
"""

import logging
from contextlib import contextmanager
from typing import Any, Generator

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

logger = logging.getLogger(__name__)

# ── Tracer setup ──────────────────────────────────────────────────────────────

# In-memory exporter — useful for tests and development.
# In production, replace with OTLPSpanExporter pointing to a collector.
_span_exporter = InMemorySpanExporter()
_tracer_provider = TracerProvider()
_tracer_provider.add_span_processor(SimpleSpanProcessor(_span_exporter))

# Register as the global provider so opentelemetry.trace.get_tracer() works
trace.set_tracer_provider(_tracer_provider)

_tracer = trace.get_tracer("rag_pipeline", tracer_provider=_tracer_provider)


@contextmanager
def trace_stage(
    stage_name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[trace.Span, None, None]:
    """Context manager that wraps a pipeline stage in an OpenTelemetry span.

    Args:
        stage_name: Name for the span (e.g. "retrieval", "generation", "cache_lookup").
        attributes: Optional dict of span attributes to record (e.g. query text, top_k).

    Yields:
        The active OpenTelemetry Span object.

    Example:
        with trace_stage("retrieval", {"query": "RAG explained", "top_k": 5}):
            results = hybrid_retrieve(...)
    """
    with _tracer.start_as_current_span(stage_name) as span:
        if attributes:
            for key, value in attributes.items():
                try:
                    span.set_attribute(key, str(value))
                except Exception:
                    pass  # span attribute setting is non-fatal
        yield span


def get_finished_spans() -> list:
    """Return all finished spans from the in-memory exporter.

    Primarily used in tests to verify that spans were created.

    Returns:
        List of finished ReadableSpan objects.
    """
    return _span_exporter.get_finished_spans()


def clear_spans() -> None:
    """Clear all stored spans from the in-memory exporter.

    Call this between tests to avoid span accumulation.
    """
    _span_exporter.clear()
