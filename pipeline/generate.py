"""Answer generation with Ollama.

Builds a grounded prompt from retrieval results, selects the appropriate Ollama
model (text: llama3.2, vision: llama3.2-vision), and returns a structured answer
dict with citations.

Public API:
    OllamaClient    — async HTTP client wrapping the Ollama REST API
    build_prompt()  — format retrieval results into a cited context prompt
    generate_answer() — orchestrate model selection + generation
"""

import logging
from typing import Any

import httpx

from pipeline.config import (
    OLLAMA_BASE_URL,
    OLLAMA_TEXT_MODEL,
    OLLAMA_TEXT_TIMEOUT,
    OLLAMA_VISION_MODEL,
    OLLAMA_VISION_TIMEOUT,
)

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async HTTP client for the Ollama local inference server.

    Wraps the Ollama REST API (http://localhost:11434 by default).
    All methods are async and must be awaited.

    Args:
        base_url: Base URL of the Ollama server.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    async def generate_text(
        self,
        prompt: str,
        model: str = OLLAMA_TEXT_MODEL,
    ) -> str:
        """Generate a text response from a text-only model.

        Args:
            prompt: The full prompt string (context + question).
            model: Ollama model name (default: llama3.2).

        Returns:
            Generated text string.

        Raises:
            httpx.HTTPStatusError: If the Ollama server returns a non-2xx response.
            httpx.ConnectError: If the Ollama server is not running.
        """
        async with httpx.AsyncClient(timeout=OLLAMA_TEXT_TIMEOUT) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return response.json()["response"]

    async def generate_vision(
        self,
        prompt: str,
        image_b64: str,
        model: str = OLLAMA_VISION_MODEL,
    ) -> str:
        """Generate a response that incorporates a base64-encoded image.

        Args:
            prompt: The full prompt string (context + question).
            image_b64: Base64-encoded JPEG image (no data URI prefix).
            model: Ollama vision model name (default: llama3.2-vision).

        Returns:
            Generated text string.

        Raises:
            httpx.HTTPStatusError: If the Ollama server returns a non-2xx response.
            httpx.ConnectError: If the Ollama server is not running.
        """
        async with httpx.AsyncClient(timeout=OLLAMA_VISION_TIMEOUT) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json()["response"]

    async def is_available(self) -> bool:
        """Check if the Ollama server is reachable.

        Returns:
            True if the server responds within 2 seconds, False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False


# ── Prompt construction ───────────────────────────────────────────────────────

def build_prompt(query: str, retrieval_results: list[dict[str, Any]]) -> str:
    """Format retrieval results into a cited context prompt.

    Each result is numbered [1], [2], ... and includes its source identifier
    and modality. Text chunks include their content; images and video frames
    get a ``[visual content]`` placeholder since the model processes the actual
    image separately via the images parameter.

    Args:
        query: The user's original query string.
        retrieval_results: List of retrieval result dicts (from hybrid_retrieve).

    Returns:
        A formatted prompt string ready to send to an Ollama model.
    """
    context_parts: list[str] = []

    for i, result in enumerate(retrieval_results):
        citation_num = i + 1
        meta = result.get("metadata", {})
        modality = meta.get("modality", "unknown")
        source = meta.get("source_id", "unknown")

        if modality == "text":
            context_parts.append(
                f"[{citation_num}] (text, source: {source}):\n{result.get('document', '')}"
            )
        elif modality == "image":
            context_parts.append(
                f"[{citation_num}] (image, source: {source}): [visual content]"
            )
        elif modality == "video":
            timestamp = meta.get("timestamp_sec", "")
            ts_str = f", t={timestamp}s" if timestamp != "" else ""
            context_parts.append(
                f"[{citation_num}] (video frame, source: {source}{ts_str}): [visual content]"
            )
        else:
            context_parts.append(
                f"[{citation_num}] (source: {source}): {result.get('document', '')}"
            )

    context = "\n\n".join(context_parts)

    return (
        f"Answer the following question using only the provided context. "
        f"Cite your sources using [N] notation where N matches the context number.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


# ── Main generation entry point ───────────────────────────────────────────────

async def generate_answer(
    query: str,
    retrieval_results: list[dict[str, Any]],
    ollama_client: OllamaClient,
) -> dict[str, Any]:
    """Generate a grounded answer using Ollama, with source citations.

    Model selection logic:
    - If any of the top-3 results are image or video modality → use vision model
    - Otherwise → use text model

    On Ollama connection failure, returns a structured error dict rather than
    raising an exception, so the API layer can return a graceful response.

    Args:
        query: The user's original query string.
        retrieval_results: List of retrieval result dicts.
        ollama_client: An OllamaClient instance.

    Returns:
        Dict with keys:
            ``answer``: Generated text string (or error message).
            ``citations``: List of source_id strings from retrieval results.
            ``retrieval_results``: The input retrieval results (passed through).
            ``error``: Present only on failure, contains the error message.
    """
    citations = [
        r.get("metadata", {}).get("source_id", "unknown") for r in retrieval_results
    ]

    if not retrieval_results:
        return {
            "answer": "No relevant documents found for your query.",
            "citations": [],
            "retrieval_results": [],
        }

    prompt = build_prompt(query, retrieval_results)

    # Determine if vision model needed: any visual result in top 3
    top_results = retrieval_results[:3]
    has_visual = any(
        r.get("metadata", {}).get("modality") in ("image", "video")
        for r in top_results
    )

    try:
        if has_visual:
            # Find first available image/video thumbnail for vision context
            first_visual = next(
                (
                    r for r in retrieval_results
                    if r.get("metadata", {}).get("modality") in ("image", "video")
                ),
                None,
            )
            image_b64 = (
                first_visual.get("metadata", {}).get("thumbnail_b64", "")
                if first_visual
                else ""
            )
            answer = await ollama_client.generate_vision(prompt, image_b64)
            logger.info("generate_answer: used vision model for query=%r", query[:50])
        else:
            answer = await ollama_client.generate_text(prompt)
            logger.info("generate_answer: used text model for query=%r", query[:50])

    except httpx.ConnectError as exc:
        logger.error("generate_answer: Ollama not reachable — %s", exc)
        return {
            "answer": (
                "Unable to generate an answer: the Ollama server is not running. "
                "Start it with `ollama serve` and ensure the model is pulled."
            ),
            "citations": citations,
            "retrieval_results": retrieval_results,
            "error": "ollama_unavailable",
        }
    except httpx.HTTPStatusError as exc:
        logger.error("generate_answer: Ollama HTTP error — %s", exc)
        return {
            "answer": f"Unable to generate an answer: model returned HTTP {exc.response.status_code}.",
            "citations": citations,
            "retrieval_results": retrieval_results,
            "error": "ollama_http_error",
        }

    return {
        "answer": answer,
        "citations": citations,
        "retrieval_results": retrieval_results,
    }
