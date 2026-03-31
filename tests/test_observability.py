"""Tests for observability/metrics.py and observability/tracing.py.

Prometheus tests use the module-level custom REGISTRY — no global registry
conflicts. Tracing tests use the InMemorySpanExporter via get_finished_spans().
"""

import time
import pytest

from observability.metrics import (
    CACHE_HITS,
    CACHE_MISSES,
    GENERATION_LATENCY,
    INGEST_COUNT,
    QUERY_COUNT,
    RERANK_LATENCY,
    RETRIEVAL_LATENCY,
    get_metrics_summary,
    get_prometheus_output,
)
from observability.tracing import clear_spans, get_finished_spans, trace_stage


# ── Prometheus metrics tests ──────────────────────────────────────────────────

class TestPrometheusMetrics:
    def test_query_count_increments(self):
        before = _counter_total(QUERY_COUNT)
        QUERY_COUNT.labels(status="success").inc()
        after = _counter_total(QUERY_COUNT)
        assert after == before + 1

    def test_cache_hits_counter_increments(self):
        before = _counter_total(CACHE_HITS)
        CACHE_HITS.inc()
        after = _counter_total(CACHE_HITS)
        assert after == before + 1

    def test_cache_misses_counter_increments(self):
        before = _counter_total(CACHE_MISSES)
        CACHE_MISSES.inc()
        after = _counter_total(CACHE_MISSES)
        assert after == before + 1

    def test_ingest_count_per_modality(self):
        before_text = _counter_label(INGEST_COUNT, "text")
        INGEST_COUNT.labels(modality="text").inc()
        INGEST_COUNT.labels(modality="text").inc()
        after_text = _counter_label(INGEST_COUNT, "text")
        assert after_text == before_text + 2

    def test_retrieval_latency_histogram_records_observation(self):
        before_count = _histogram_count(RETRIEVAL_LATENCY)
        with RETRIEVAL_LATENCY.time():
            time.sleep(0.001)
        after_count = _histogram_count(RETRIEVAL_LATENCY)
        assert after_count == before_count + 1

    def test_generation_latency_histogram_records_observation(self):
        before_count = _histogram_count(GENERATION_LATENCY)
        GENERATION_LATENCY.observe(2.5)
        after_count = _histogram_count(GENERATION_LATENCY)
        assert after_count == before_count + 1

    def test_rerank_latency_histogram_records_observation(self):
        before_count = _histogram_count(RERANK_LATENCY)
        RERANK_LATENCY.observe(0.05)
        after_count = _histogram_count(RERANK_LATENCY)
        assert after_count == before_count + 1

    def test_get_prometheus_output_returns_bytes(self):
        output, content_type = get_prometheus_output()
        assert isinstance(output, bytes)
        assert len(output) > 0
        assert "text/plain" in content_type

    def test_prometheus_output_contains_metric_names(self):
        output, _ = get_prometheus_output()
        text = output.decode("utf-8")
        assert "rag_queries_total" in text
        assert "rag_cache_hits_total" in text
        assert "rag_retrieval_latency_seconds" in text

    def test_get_metrics_summary_returns_dict_with_expected_keys(self):
        summary = get_metrics_summary()
        assert "total_queries" in summary
        assert "cache_hits" in summary
        assert "cache_misses" in summary
        assert "cache_hit_rate" in summary
        assert "avg_retrieval_latency_ms" in summary
        assert "avg_generation_latency_ms" in summary

    def test_cache_hit_rate_calculation(self):
        # Force known counts into the counters
        CACHE_HITS.inc(3)
        CACHE_MISSES.inc(1)
        summary = get_metrics_summary()
        # We can't assert exact values because other tests also increment,
        # but the rate must be between 0 and 1
        assert 0.0 <= summary["cache_hit_rate"] <= 1.0


# ── OpenTelemetry tracing tests ───────────────────────────────────────────────

class TestTracing:
    def setup_method(self):
        """Clear spans before each test to avoid accumulation."""
        clear_spans()

    def test_trace_stage_creates_span(self):
        with trace_stage("test_stage"):
            pass
        spans = get_finished_spans()
        names = [s.name for s in spans]
        assert "test_stage" in names

    def test_trace_stage_span_has_correct_name(self):
        clear_spans()
        with trace_stage("retrieval"):
            pass
        spans = get_finished_spans()
        assert any(s.name == "retrieval" for s in spans)

    def test_trace_stage_with_attributes(self):
        clear_spans()
        with trace_stage("generation", {"query": "test query", "top_k": 5}):
            pass
        spans = get_finished_spans()
        gen_span = next(s for s in spans if s.name == "generation")
        # Attributes are stored as strings
        assert gen_span.attributes.get("query") == "test query"
        assert gen_span.attributes.get("top_k") == "5"

    def test_nested_spans(self):
        clear_spans()
        with trace_stage("outer"):
            with trace_stage("inner"):
                pass
        spans = get_finished_spans()
        names = [s.name for s in spans]
        assert "outer" in names
        assert "inner" in names

    def test_span_created_even_on_exception(self):
        """Span should be recorded even if the body raises."""
        clear_spans()
        try:
            with trace_stage("failing_stage"):
                raise ValueError("intentional error")
        except ValueError:
            pass
        spans = get_finished_spans()
        assert any(s.name == "failing_stage" for s in spans)

    def test_clear_spans_removes_all(self):
        with trace_stage("temp"):
            pass
        clear_spans()
        spans = get_finished_spans()
        assert len(spans) == 0

    def test_get_finished_spans_returns_sequence(self):
        # InMemorySpanExporter returns a tuple (iterable sequence)
        result = get_finished_spans()
        assert hasattr(result, "__iter__")

    def test_multiple_stages_all_recorded(self):
        clear_spans()
        stages = ["embed", "retrieve", "rerank", "generate"]
        for stage in stages:
            with trace_stage(stage):
                pass
        spans = get_finished_spans()
        recorded_names = {s.name for s in spans}
        for stage in stages:
            assert stage in recorded_names


# ── Helper functions ──────────────────────────────────────────────────────────

def _counter_total(counter) -> float:
    """Sum all sample values for a Counter (across all labels)."""
    try:
        return sum(
            s.value for s in counter.collect()[0].samples
            if s.name.endswith("_total")
        )
    except (IndexError, AttributeError):
        return 0.0


def _counter_label(counter, label_value: str) -> float:
    """Get counter value for a specific label."""
    try:
        for sample in counter.collect()[0].samples:
            if label_value in sample.labels.values() and sample.name.endswith("_total"):
                return sample.value
        return 0.0
    except (IndexError, AttributeError):
        return 0.0


def _histogram_count(histogram) -> float:
    """Get the _count sample from a Histogram."""
    try:
        for sample in histogram.collect()[0].samples:
            if sample.name.endswith("_count"):
                return sample.value
        return 0.0
    except (IndexError, AttributeError):
        return 0.0
