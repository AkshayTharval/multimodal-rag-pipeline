"""Semantic cache for the RAG pipeline.

Caches query results in SQLite using cosine distance between query embeddings.
A cache HIT occurs when the new query is within CACHE_DISTANCE_THRESHOLD cosine
distance of a stored query (threshold = 0.05 → similarity > 0.95).

IMPORTANT: The threshold is a cosine DISTANCE (0=identical, 2=opposite).
0.05 distance means >95% cosine similarity — a very strict cache condition.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.config import (
    CACHE_DB_PATH,
    CACHE_DISTANCE_THRESHOLD,
    TEXT_EMBED_MODEL,
)

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS query_cache (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text  TEXT    NOT NULL,
    query_embedding BLOB NOT NULL,
    answer      TEXT    NOT NULL,
    citations   TEXT    NOT NULL,
    retrieval_results TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class SemanticCache:
    """SQLite-backed semantic cache using cosine distance between query embeddings.

    On lookup, embeds the new query and computes cosine distance against all
    stored embeddings. Returns the cached result for the closest match if it
    is within the distance threshold.

    Note on scale: loads all embeddings into memory on each lookup. This is
    O(n) per query and acceptable for portfolio scale (<10,000 cached queries).

    Args:
        db_path: Path to the SQLite database file. Use ``:memory:`` for in-memory.
        similarity_threshold: Cosine DISTANCE threshold for cache hits.
            Default 0.05 means similarity > 0.95 required.
        embedder: SentenceTransformer instance. If None, loads the default model.
    """

    def __init__(
        self,
        db_path: str = CACHE_DB_PATH,
        similarity_threshold: float = CACHE_DISTANCE_THRESHOLD,
        embedder: SentenceTransformer | None = None,
    ) -> None:
        self.db_path = db_path
        # threshold is cosine DISTANCE — lower = more similar
        self.threshold = similarity_threshold
        self._embedder = embedder
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy-load the embedder on first use."""
        if self._embedder is None:
            self._embedder = SentenceTransformer(TEXT_EMBED_MODEL)
        return self._embedder

    def lookup(self, query_text: str) -> dict[str, Any] | None:
        """Look up a query in the cache.

        Embeds the query, computes cosine distance against all stored embeddings,
        and returns the cached result for the closest match if within threshold.

        Args:
            query_text: The user's query string.

        Returns:
            Cached result dict (with ``cache_hit=True`` added) if a hit is found,
            otherwise ``None``.
        """
        cursor = self._conn.execute(
            "SELECT query_text, query_embedding, answer, citations, retrieval_results "
            "FROM query_cache"
        )
        rows = cursor.fetchall()

        if not rows:
            return None

        query_embedding = self._embed(query_text)

        best_distance = float("inf")
        best_row = None

        for row in rows:
            stored_embedding = np.frombuffer(row[1], dtype=np.float32)
            distance = _cosine_distance(query_embedding, stored_embedding)
            if distance < best_distance:
                best_distance = distance
                best_row = row

        if best_distance < self.threshold and best_row is not None:
            logger.info(
                "cache HIT — distance=%.4f, matched_query=%r",
                best_distance,
                best_row[0][:60],
            )
            return {
                "answer": best_row[2],
                "citations": json.loads(best_row[3]),
                "retrieval_results": json.loads(best_row[4]),
                "cache_hit": True,
                "matched_query": best_row[0],
                "cache_distance": float(best_distance),
            }

        logger.debug("cache MISS — best_distance=%.4f (threshold=%.4f)", best_distance, self.threshold)
        return None

    def store(self, query_text: str, result: dict[str, Any]) -> None:
        """Store a query result in the cache.

        Args:
            query_text: The user's query string.
            result: The pipeline result dict (must have ``answer``, ``citations``,
                and ``retrieval_results`` keys).
        """
        query_embedding = self._embed(query_text)
        embedding_blob = query_embedding.astype(np.float32).tobytes()

        self._conn.execute(
            "INSERT INTO query_cache (query_text, query_embedding, answer, citations, retrieval_results) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                query_text,
                embedding_blob,
                result.get("answer", ""),
                json.dumps(result.get("citations", [])),
                json.dumps(result.get("retrieval_results", [])),
            ),
        )
        self._conn.commit()
        logger.info("cache STORE — query=%r", query_text[:60])

    def count(self) -> int:
        """Return the number of cached entries."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM query_cache")
        return cursor.fetchone()[0]

    def clear(self) -> None:
        """Remove all cached entries."""
        self._conn.execute("DELETE FROM query_cache")
        self._conn.commit()

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    def _embed(self, text: str) -> np.ndarray:
        """Embed a text string and return a numpy float32 array."""
        vec = self.embedder.encode([text], show_progress_bar=False)[0]
        return vec.astype(np.float32)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors.

    cosine_distance = 1 - cosine_similarity = 1 - dot(a,b) / (|a| * |b|)
    Range: [0, 2]. 0 = identical direction, 2 = opposite direction.

    Args:
        a: First vector (numpy array).
        b: Second vector (numpy array).

    Returns:
        Cosine distance as a Python float.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0  # treat zero vectors as maximally dissimilar
    similarity = np.dot(a, b) / (norm_a * norm_b)
    # Clamp to [-1, 1] to handle floating-point edge cases
    similarity = float(np.clip(similarity, -1.0, 1.0))
    return 1.0 - similarity
