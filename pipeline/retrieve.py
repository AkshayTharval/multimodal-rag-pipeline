"""Hybrid retrieval pipeline.

Combines three retrieval signals and fuses them into a single ranked list:
1. BM25 sparse retrieval (text_chunks collection only)
2. Dense cosine similarity retrieval (all three collections)
3. RRF (Reciprocal Rank Fusion) score fusion
4. Cross-encoder reranking (text candidates only)

Public API:
    BM25Index           — in-memory BM25 index built from a ChromaDB text collection
    dense_query()       — cosine similarity search over a ChromaDB collection
    rrf_fusion()        — merge N ranked lists into one using RRF
    rerank()            — cross-encoder reranking of text candidates
    hybrid_retrieve()   — full pipeline combining all of the above
"""

import logging
from typing import Any

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from pipeline.config import (
    DEFAULT_TOP_K,
    OLLAMA_TEXT_MODEL,
    RERANKER_MODEL,
    RRF_K,
)
from pipeline.ingest import clip_embed_text

logger = logging.getLogger(__name__)


# ── BM25 sparse retrieval ─────────────────────────────────────────────────────

class BM25Index:
    """In-memory BM25 index built from a ChromaDB text collection.

    The index is rebuilt each time this class is instantiated — this is O(n)
    in the number of documents and is acceptable for portfolio-scale corpora
    (<50k chunks). For large-scale production, persist and incrementally update.

    Attributes:
        docs: List of raw document strings from ChromaDB.
        ids: Corresponding document IDs.
        metadatas: Corresponding metadata dicts.
        bm25: The underlying BM25Okapi instance.
    """

    def __init__(self, text_collection: chromadb.Collection) -> None:
        """Build the BM25 index from all documents in the text collection.

        Args:
            text_collection: ChromaDB collection containing text chunks.
        """
        # "ids" is always returned by ChromaDB get() — do not include in the include list
        result = text_collection.get(include=["documents", "metadatas"])
        self.docs: list[str] = result["documents"] or []
        self.ids: list[str] = result["ids"] or []
        self.metadatas: list[dict] = result["metadatas"] or []

        if self.docs:
            tokenized = [doc.lower().split() for doc in self.docs]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    def query(self, query_text: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """Score all documents against the query and return top-k results.

        Args:
            query_text: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            List of result dicts sorted by descending BM25 score. Each dict has:
            ``id``, ``score``, ``rank``, ``document``, ``metadata``.
            Returns empty list if the index has no documents.
        """
        if not self.docs or self.bm25 is None:
            return []

        tokens = query_text.lower().split()
        scores = self.bm25.get_scores(tokens)

        effective_top_k = min(top_k, len(self.docs))
        top_indices = np.argsort(scores)[-effective_top_k:][::-1]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append(
                {
                    "id": self.ids[idx],
                    "score": float(scores[idx]),
                    "rank": rank + 1,
                    "document": self.docs[idx],
                    "metadata": self.metadatas[idx],
                }
            )
        return results


# ── Dense cosine retrieval ────────────────────────────────────────────────────

def dense_query(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Query a ChromaDB collection with a dense embedding vector.

    ChromaDB returns distances in cosine space (distance = 1 - similarity).
    Results are converted to similarity scores: ``score = 1 - distance``.

    Args:
        collection: ChromaDB collection to search (any modality).
        query_embedding: L2-normalised query embedding vector.
        top_k: Maximum number of results to return.

    Returns:
        List of result dicts sorted by descending similarity score. Each dict has:
        ``id``, ``score`` (cosine similarity), ``rank``, ``document``, ``metadata``.
        Returns empty list if the collection is empty.
    """
    if collection.count() == 0:
        return []

    effective_top_k = min(top_k, collection.count())
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=effective_top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        output.append(
            {
                "id": results["ids"][0][i],
                "score": float(1.0 - distance),   # cosine similarity
                "rank": i + 1,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
            }
        )
    return output


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def rrf_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    Formula: score(d) = Σ_i  1 / (k + rank_i(d))
    Documents not present in a list contribute 0 from that list.
    k=60 is the standard value; higher k reduces the impact of rank differences.

    Args:
        ranked_lists: List of ranked result lists. Each inner list contains
            dicts with at least ``id`` and ``rank`` keys.
        k: RRF constant (default 60).

    Returns:
        Single merged list sorted by descending RRF score. Each item is the
        last-seen metadata for that document ID, with ``rrf_score`` added.
    """
    rrf_scores: dict[str, float] = {}
    doc_data: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for item in ranked_list:
            doc_id = item["id"]
            rank = item["rank"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_data[doc_id] = item  # last-write wins for metadata

    merged = [
        {**doc_data[doc_id], "rrf_score": score}
        for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return merged


# ── Cross-encoder reranking ───────────────────────────────────────────────────

def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    cross_encoder: CrossEncoder,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Rerank text candidates using a cross-encoder model.

    Only text-modality candidates are passed to the cross-encoder — it cannot
    meaningfully score images or video frames. Visual candidates are appended
    after the reranked text results, in their original RRF order.

    Args:
        query: The user's search query string.
        candidates: List of retrieved candidates from RRF fusion.
        cross_encoder: Loaded CrossEncoder instance.
        top_k: Number of results to return after reranking.

    Returns:
        Reranked list of up to top_k items. Text items are sorted by
        cross-encoder score (descending); visual items follow in RRF order.
        Each text item gets a ``rerank_score`` key added.
    """
    text_candidates = [
        c for c in candidates if c.get("metadata", {}).get("modality") == "text"
    ]
    visual_candidates = [
        c for c in candidates if c.get("metadata", {}).get("modality") != "text"
    ]

    if text_candidates:
        pairs = [(query, c["document"]) for c in text_candidates]
        scores = cross_encoder.predict(pairs)
        for i, candidate in enumerate(text_candidates):
            candidate["rerank_score"] = float(scores[i])
        text_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    combined = text_candidates + visual_candidates
    return combined[:top_k]


# ── Full hybrid retrieval pipeline ────────────────────────────────────────────

def hybrid_retrieve(
    query_text: str,
    text_collection: chromadb.Collection,
    image_collection: chromadb.Collection,
    video_collection: chromadb.Collection,
    text_embedder: SentenceTransformer,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    cross_encoder: CrossEncoder,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Full hybrid retrieval pipeline.

    Steps:
    1. BM25 retrieval on text_chunks
    2. Dense retrieval: SentenceTransformer embedding → text_chunks
    3. Dense retrieval: CLIP text encoding → image_embeddings
    4. Dense retrieval: CLIP text encoding → video_keyframes
    5. RRF fusion of all four ranked lists
    6. Cross-encoder rerank (text candidates only)

    Args:
        query_text: The user's natural language query.
        text_collection: ChromaDB text_chunks collection.
        image_collection: ChromaDB image_embeddings collection.
        video_collection: ChromaDB video_keyframes collection.
        text_embedder: Loaded SentenceTransformer for text search.
        clip_model: Loaded CLIPModel for image/video search.
        clip_processor: Loaded CLIPProcessor.
        cross_encoder: Loaded CrossEncoder for reranking.
        top_k: Number of final results to return.

    Returns:
        Reranked list of up to top_k retrieval results with full provenance
        metadata including rrf_score and (for text) rerank_score.
    """
    ranked_lists: list[list[dict]] = []

    # 1. BM25 on text
    bm25_index = BM25Index(text_collection)
    bm25_results = bm25_index.query(query_text, top_k=top_k * 2)
    if bm25_results:
        ranked_lists.append(bm25_results)
        logger.debug("hybrid_retrieve: BM25 returned %d results", len(bm25_results))

    # 2. Dense text retrieval
    text_embedding = text_embedder.encode([query_text], show_progress_bar=False)[0].tolist()
    dense_text_results = dense_query(text_collection, text_embedding, top_k=top_k * 2)
    if dense_text_results:
        ranked_lists.append(dense_text_results)
        logger.debug("hybrid_retrieve: dense text returned %d results", len(dense_text_results))

    # 3. CLIP text→image retrieval
    clip_text_embedding = clip_embed_text(query_text, clip_model, clip_processor)
    dense_image_results = dense_query(image_collection, clip_text_embedding, top_k=top_k * 2)
    if dense_image_results:
        ranked_lists.append(dense_image_results)
        logger.debug(
            "hybrid_retrieve: dense image returned %d results", len(dense_image_results)
        )

    # 4. CLIP text→video retrieval
    dense_video_results = dense_query(video_collection, clip_text_embedding, top_k=top_k * 2)
    if dense_video_results:
        ranked_lists.append(dense_video_results)
        logger.debug(
            "hybrid_retrieve: dense video returned %d results", len(dense_video_results)
        )

    if not ranked_lists:
        logger.warning("hybrid_retrieve: all ranked lists empty for query=%r", query_text)
        return []

    # 5. RRF fusion
    fused = rrf_fusion(ranked_lists)

    # 6. Cross-encoder rerank
    reranked = rerank(query_text, fused, cross_encoder, top_k=top_k)
    return reranked
