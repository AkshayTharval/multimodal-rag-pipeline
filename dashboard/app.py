"""Streamlit dashboard for the Multimodal RAG Pipeline.

Connects to the FastAPI service over HTTP — does NOT import pipeline modules.
This architectural boundary keeps the dashboard stateless and deployable
independently of the backend.

Run with:
    streamlit run dashboard/app.py

The FastAPI service must be running at the configured API_URL.
"""

import pandas as pd
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multimodal RAG Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────

MODALITY_ICONS = {
    "text": "📄",
    "image": "🖼️",
    "video": "🎬",
}

DEFAULT_API_URL = "http://localhost:8000"


# ── Helper functions (defined before use) ────────────────────────────────────

def render_provenance(retrieval_results: list[dict]) -> None:
    """Render one row per retrieved result with icon, scores, and thumbnail."""
    for i, result in enumerate(retrieval_results):
        meta = result.get("metadata", {})
        modality = meta.get("modality", "unknown")
        icon = MODALITY_ICONS.get(modality, "❓")
        source = meta.get("source_id", "unknown")
        rrf_score = result.get("rrf_score", 0.0)
        rerank_score = result.get("rerank_score")
        dense_score = result.get("score", 0.0)

        col_icon, col_info, col_thumb = st.columns([0.5, 5, 2])

        with col_icon:
            st.markdown(f"### {icon}")
            st.caption(f"#{i + 1}")

        with col_info:
            st.markdown(f"**{source}**")

            # Score metrics
            score_cols = st.columns(3)
            score_cols[0].metric("RRF Score", f"{rrf_score:.4f}")
            score_cols[1].metric("Dense Score", f"{dense_score:.3f}")
            if rerank_score is not None:
                score_cols[2].metric("Rerank Score", f"{rerank_score:.3f}")

            # Score bar (scale rrf_score for visibility)
            normalised = min(rrf_score * 200, 1.0)
            st.progress(normalised, text=f"RRF contribution: {rrf_score:.4f}")

            # Text chunk
            if modality == "text":
                doc_text = result.get("document", "")
                if doc_text:
                    st.markdown(
                        f'<div style="background:#f0f2f6;padding:8px;border-radius:4px;'
                        f'font-size:0.85em;">'
                        f'{doc_text[:400]}{"..." if len(doc_text) > 400 else ""}'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Video timestamp
            if modality == "video":
                ts = meta.get("timestamp_sec", "")
                if ts != "":
                    st.caption(f"🕐 Timestamp: {ts}s")

        with col_thumb:
            thumbnail_b64 = meta.get("thumbnail_b64", "")
            if thumbnail_b64 and modality in ("image", "video"):
                st.image(
                    f"data:image/jpeg;base64,{thumbnail_b64}",
                    width=120,
                    caption=modality.capitalize(),
                )

        st.divider()


def render_score_breakdown(retrieval_results: list[dict]) -> None:
    """Render a bar chart of each source's RRF score contribution."""
    labels = []
    scores = []
    for i, r in enumerate(retrieval_results):
        source = r.get("metadata", {}).get("source_id", f"result_{i}")
        modality = r.get("metadata", {}).get("modality", "")
        icon = MODALITY_ICONS.get(modality, "")
        labels.append(f"{icon} {source[:20]}")
        scores.append(r.get("rrf_score", 0.0))

    if labels:
        df = pd.DataFrame({"Source": labels, "RRF Score": scores})
        st.subheader("📊 Score Breakdown")
        st.bar_chart(df.set_index("Source"))


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    api_url = st.text_input(
        "API Base URL",
        value=st.session_state.get("api_url", DEFAULT_API_URL),
        help="URL of the running FastAPI service",
    )
    st.session_state["api_url"] = api_url

    top_k = st.slider("Top-K results", min_value=1, max_value=20, value=5)
    use_cache = st.toggle("Use semantic cache", value=True)

    st.divider()

    if st.button("🔍 Health Check"):
        try:
            resp = requests.get(f"{api_url}/health", timeout=3)
            health = resp.json()
            for k, v in health.items():
                icon = "✅" if v in ("ok", "healthy") else "❌"
                st.write(f"{icon} **{k}**: `{v}`")
        except Exception as exc:
            st.error(f"Cannot reach API: {exc}")

    st.divider()

    if st.button("📊 Refresh Stats"):
        try:
            resp = requests.get(f"{api_url}/stats", timeout=3)
            stats = resp.json()
            st.metric("Text chunks", stats.get("text_chunks", 0))
            st.metric("Image embeddings", stats.get("image_embeddings", 0))
            st.metric("Video keyframes", stats.get("video_keyframes", 0))
            st.metric("Cached queries", stats.get("cached_queries", 0))
            hit_rate = stats.get("cache_hit_rate", 0)
            st.metric("Cache hit rate", f"{hit_rate:.1%}")
        except Exception as exc:
            st.error(f"Cannot fetch stats: {exc}")

    st.divider()

    st.subheader("📤 Ingest a File")
    uploaded = st.file_uploader(
        "Upload text, image, or video",
        type=["txt", "md", "jpg", "jpeg", "png", "webp", "mp4", "avi", "mov"],
    )
    if uploaded and st.button("Ingest"):
        try:
            resp = requests.post(
                f"{api_url}/ingest",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                data={"modality": "auto"},
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
            st.success(
                f"✅ Ingested `{uploaded.name}` as **{result['modality']}** "
                f"({result['chunks_created']} chunk(s))"
            )
        except Exception as exc:
            st.error(f"Ingest failed: {exc}")


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🔍 Multimodal RAG Pipeline")
st.caption("Query text, images, and video keyframes in a unified pipeline.")

query = st.text_area(
    "Enter your query",
    placeholder="e.g. 'show me scenes with outdoor settings' or 'what is RAG?'",
    height=80,
)
submit = st.button("🚀 Submit Query", type="primary", disabled=not query.strip())

# ── Query execution ───────────────────────────────────────────────────────────

if submit and query.strip():
    with st.spinner("Running pipeline..."):
        try:
            resp = requests.post(
                f"{api_url}/query",
                json={"query": query, "top_k": top_k, "use_cache": use_cache},
                timeout=180,
            )
            resp.raise_for_status()
            st.session_state["last_response"] = resp.json()
        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the API. "
                "Start it with: `uvicorn api.app:app --reload --port 8000`"
            )
            st.stop()
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            st.stop()

# ── Results display ───────────────────────────────────────────────────────────

if "last_response" in st.session_state:
    data = st.session_state["last_response"]

    # Cache hit badge
    if data.get("cache_hit"):
        st.success("⚡ Cache hit — result served from semantic cache (no LLM call)")
    else:
        st.info("🔄 Cache miss — full pipeline executed")

    latency_ms = data.get("latency_ms", 0)
    st.caption(f"⏱ Total latency: **{latency_ms:.0f} ms**")

    # Answer
    st.subheader("💬 Answer")
    if data.get("error"):
        st.warning(f"⚠️ {data['answer']}")
    else:
        st.markdown(data.get("answer", "_No answer generated._"))

    # Citations
    citations = data.get("citations", [])
    if citations:
        unique_citations = list(dict.fromkeys(citations))
        st.caption("📚 Sources: " + " · ".join(f"`{c}`" for c in unique_citations))

    st.divider()

    # Retrieval provenance
    retrieval_results = data.get("retrieval_results", [])
    if retrieval_results:
        with st.expander(
            f"🔎 Retrieval Provenance ({len(retrieval_results)} results)", expanded=True
        ):
            render_provenance(retrieval_results)

        render_score_breakdown(retrieval_results)

    # Raw response
    with st.expander("📋 Raw API Response"):
        st.json(data)
