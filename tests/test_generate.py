"""Tests for pipeline/generate.py.

All tests mock Ollama via httpx — no real LLM calls are made.
Uses pytest-asyncio (asyncio_mode=auto in pytest.ini).
"""

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from pipeline.generate import (
    OllamaClient,
    build_prompt,
    generate_answer,
)


# ── build_prompt unit tests ───────────────────────────────────────────────────

class TestBuildPrompt:
    def _text_result(self, n: int, source: str = "doc.txt") -> dict:
        return {
            "id": f"doc_{n}",
            "rank": n,
            "document": f"Content of document {n}.",
            "metadata": {"source_id": source, "modality": "text"},
        }

    def _image_result(self, n: int, source: str = "photo.jpg") -> dict:
        return {
            "id": f"img_{n}",
            "rank": n,
            "document": f"Image: {source}",
            "metadata": {
                "source_id": source,
                "modality": "image",
                "thumbnail_b64": "abc123",
            },
        }

    def _video_result(self, n: int, source: str = "clip.mp4", ts: float = 5.0) -> dict:
        return {
            "id": f"vid_{n}",
            "rank": n,
            "document": f"Video frame: {source}",
            "metadata": {
                "source_id": source,
                "modality": "video",
                "timestamp_sec": ts,
                "thumbnail_b64": "xyz789",
            },
        }

    def test_citation_numbers_start_at_one(self):
        results = [self._text_result(1), self._text_result(2)]
        prompt = build_prompt("What is RAG?", results)
        assert "[1]" in prompt
        assert "[2]" in prompt

    def test_text_content_included_in_prompt(self):
        results = [self._text_result(1, source="rag_paper.txt")]
        prompt = build_prompt("test query", results)
        assert "Content of document 1" in prompt
        assert "rag_paper.txt" in prompt

    def test_image_result_shows_visual_content_placeholder(self):
        results = [self._image_result(1)]
        prompt = build_prompt("find photos", results)
        assert "[visual content]" in prompt
        assert "image" in prompt

    def test_video_result_shows_timestamp(self):
        results = [self._video_result(1, ts=12.5)]
        prompt = build_prompt("find scene", results)
        assert "12.5" in prompt
        assert "video frame" in prompt

    def test_query_appears_in_prompt(self):
        results = [self._text_result(1)]
        prompt = build_prompt("What is CLIP?", results)
        assert "What is CLIP?" in prompt

    def test_prompt_contains_answer_marker(self):
        results = [self._text_result(1)]
        prompt = build_prompt("question", results)
        assert "Answer:" in prompt

    def test_empty_results_produces_minimal_prompt(self):
        prompt = build_prompt("query", [])
        assert "query" in prompt
        assert "Answer:" in prompt

    def test_mixed_modalities_all_cited(self):
        results = [
            self._text_result(1),
            self._image_result(2),
            self._video_result(3),
        ]
        prompt = build_prompt("find content", results)
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt

    def test_prompt_instructs_citation_notation(self):
        results = [self._text_result(1)]
        prompt = build_prompt("query", results)
        assert "[N]" in prompt or "citation" in prompt.lower() or "Cite" in prompt


# ── OllamaClient unit tests ───────────────────────────────────────────────────

class TestOllamaClient:
    @pytest.fixture
    def client(self):
        return OllamaClient(base_url="http://localhost:11434")

    async def test_generate_text_calls_correct_endpoint(self, client, mocker):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "test answer"}
        mock_response.raise_for_status = MagicMock()

        mock_post = AsyncMock(return_value=mock_response)
        mocker.patch("httpx.AsyncClient.post", mock_post)

        result = await client.generate_text("What is RAG?")

        assert result == "test answer"
        call_kwargs = mock_post.call_args
        assert "/api/generate" in call_kwargs[0][0]

    async def test_generate_text_sends_correct_model(self, client, mocker):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "answer"}
        mock_response.raise_for_status = MagicMock()
        mock_post = AsyncMock(return_value=mock_response)
        mocker.patch("httpx.AsyncClient.post", mock_post)

        await client.generate_text("prompt", model="llama3.2")

        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "llama3.2"
        assert payload["stream"] is False

    async def test_generate_text_stream_false_explicit(self, client, mocker):
        """stream must be False — without it, response.json() fails on streamed output."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "answer"}
        mock_response.raise_for_status = MagicMock()
        mock_post = AsyncMock(return_value=mock_response)
        mocker.patch("httpx.AsyncClient.post", mock_post)

        await client.generate_text("prompt")

        payload = mock_post.call_args[1]["json"]
        assert payload.get("stream") is False

    async def test_generate_vision_includes_images_key(self, client, mocker):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "vision answer"}
        mock_response.raise_for_status = MagicMock()
        mock_post = AsyncMock(return_value=mock_response)
        mocker.patch("httpx.AsyncClient.post", mock_post)

        await client.generate_vision("describe this", "base64imagedata==")

        payload = mock_post.call_args[1]["json"]
        assert "images" in payload
        assert payload["images"] == ["base64imagedata=="]

    async def test_generate_vision_uses_vision_model(self, client, mocker):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "answer"}
        mock_response.raise_for_status = MagicMock()
        mock_post = AsyncMock(return_value=mock_response)
        mocker.patch("httpx.AsyncClient.post", mock_post)

        await client.generate_vision("prompt", "b64data", model="llama3.2-vision")

        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "llama3.2-vision"

    async def test_generate_text_raises_on_http_error(self, client, mocker):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock(status_code=500)
        )
        mock_post = AsyncMock(return_value=mock_response)
        mocker.patch("httpx.AsyncClient.post", mock_post)

        with pytest.raises(httpx.HTTPStatusError):
            await client.generate_text("prompt")

    async def test_is_available_returns_true_when_server_up(self, client, mocker):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get = AsyncMock(return_value=mock_response)
        mocker.patch("httpx.AsyncClient.get", mock_get)

        result = await client.is_available()
        assert result is True

    async def test_is_available_returns_false_when_server_down(self, client, mocker):
        mocker.patch(
            "httpx.AsyncClient.get",
            side_effect=httpx.ConnectError("connection refused"),
        )
        result = await client.is_available()
        assert result is False


# ── generate_answer integration-style tests (all mocked) ─────────────────────

class TestGenerateAnswer:
    def _make_results(self, modalities: list[str]) -> list[dict]:
        results = []
        for i, mod in enumerate(modalities):
            results.append({
                "id": f"doc_{i}",
                "rank": i + 1,
                "rrf_score": 0.1,
                "document": f"Document {i} content.",
                "metadata": {
                    "source_id": f"source_{i}.txt",
                    "modality": mod,
                    "thumbnail_b64": "fakeb64" if mod in ("image", "video") else "",
                    "timestamp_sec": 5.0 if mod == "video" else "",
                },
            })
        return results

    async def test_text_only_results_use_text_model(self, mocker):
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.generate_text = AsyncMock(return_value="Text answer.")
        mock_client.generate_vision = AsyncMock(return_value="Vision answer.")

        results = self._make_results(["text", "text", "text"])
        response = await generate_answer("What is RAG?", results, mock_client)

        mock_client.generate_text.assert_awaited_once()
        mock_client.generate_vision.assert_not_awaited()
        assert response["answer"] == "Text answer."

    async def test_image_in_top3_uses_vision_model(self, mocker):
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.generate_text = AsyncMock(return_value="Text answer.")
        mock_client.generate_vision = AsyncMock(return_value="Vision answer.")

        results = self._make_results(["text", "image", "text"])
        response = await generate_answer("find photos", results, mock_client)

        mock_client.generate_vision.assert_awaited_once()
        mock_client.generate_text.assert_not_awaited()
        assert response["answer"] == "Vision answer."

    async def test_video_in_top3_uses_vision_model(self, mocker):
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.generate_text = AsyncMock(return_value="Text.")
        mock_client.generate_vision = AsyncMock(return_value="Vision.")

        results = self._make_results(["video", "text", "text"])
        response = await generate_answer("find scenes", results, mock_client)

        mock_client.generate_vision.assert_awaited_once()

    async def test_returns_citations_list(self, mocker):
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.generate_text = AsyncMock(return_value="Answer.")

        results = self._make_results(["text", "text"])
        response = await generate_answer("query", results, mock_client)

        assert "citations" in response
        assert response["citations"] == ["source_0.txt", "source_1.txt"]

    async def test_returns_retrieval_results_passthrough(self, mocker):
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.generate_text = AsyncMock(return_value="Answer.")

        results = self._make_results(["text"])
        response = await generate_answer("query", results, mock_client)

        assert response["retrieval_results"] is results

    async def test_empty_results_returns_no_documents_message(self, mocker):
        mock_client = MagicMock(spec=OllamaClient)
        response = await generate_answer("query", [], mock_client)

        assert "No relevant documents" in response["answer"]
        mock_client.generate_text.assert_not_called()
        mock_client.generate_vision.assert_not_called()

    async def test_connect_error_returns_structured_error_dict(self, mocker):
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.generate_text = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )

        results = self._make_results(["text"])
        response = await generate_answer("query", results, mock_client)

        assert "error" in response
        assert response["error"] == "ollama_unavailable"
        assert "Ollama" in response["answer"] or "ollama" in response["answer"].lower()

    async def test_http_status_error_returns_structured_error_dict(self, mocker):
        mock_client = MagicMock(spec=OllamaClient)
        mock_request = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_client.generate_text = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "404 Not Found", request=mock_request, response=mock_resp
            )
        )

        results = self._make_results(["text"])
        response = await generate_answer("query", results, mock_client)

        assert "error" in response
        assert response["error"] == "ollama_http_error"

    async def test_image_in_position_4_does_not_trigger_vision(self, mocker):
        """Visual result outside top-3 should NOT trigger vision model."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.generate_text = AsyncMock(return_value="Answer.")
        mock_client.generate_vision = AsyncMock(return_value="Vision.")

        # image is at index 3 (4th result) — outside top-3
        results = self._make_results(["text", "text", "text", "image"])
        response = await generate_answer("query", results, mock_client)

        mock_client.generate_text.assert_awaited_once()
        mock_client.generate_vision.assert_not_awaited()
