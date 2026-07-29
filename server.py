"""
server.py
---------
FastAPI backend for the Runbook RAG Chatbot (production-grade upgrade).

Pipeline for every query:
    query
      → normalize (pipeline.normalizer)
      → cache lookup (pipeline.cache)
      → parallel FAISS top-50 + BM25 top-50 (pipeline.retriever)
      → Reciprocal Rank Fusion → top-30
      → cross-encoder rerank → top 3–8 adaptive (pipeline.reranker)
      → deduplicate near-identical chunks
      → context compression (pipeline.compressor)
      → structured prompt (pipeline.prompt_builder)
      → Qwen2.5-3B-Instruct GGUF streamed via llama-cpp-python
      → answer + citations + confidence (SSE stream)

All existing REST endpoints and response shapes are preserved for frontend compatibility.

Run with:
    python -m uvicorn server:app --port 8000
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# Bootstrap sys.path so config/pipeline/observability are importable
# ---------------------------------------------------------------------------
import sys
BASE_DIR = Path(__file__).parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from config import settings                                      # noqa: E402
from pipeline.embedder import get_embedder                       # noqa: E402
from pipeline.indexer import load_state, save_state, build_full_index, incremental_add  # noqa: E402
from pipeline.normalizer import normalize                        # noqa: E402
from pipeline.retriever import HybridRetriever                   # noqa: E402
from pipeline.reranker import rerank, get_reranker               # noqa: E402
from pipeline.compressor import compress_chunks                  # noqa: E402
from pipeline.prompt_builder import build_prompt                 # noqa: E402
from pipeline import cache as query_cache                        # noqa: E402
from observability.logger import (                               # noqa: E402
    RequestLog, RequestTimer, log_request, get_metrics, get_memory_mb,
)


# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------

app = FastAPI(title="Runbook RAG Chatbot")


class NoCacheMiddleware(BaseHTTPMiddleware):
    """Disable browser caching for frontend so edits to index.html are
    reflected immediately on a normal refresh."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


app.add_middleware(NoCacheMiddleware)


# ---------------------------------------------------------------------------
# Global state (populated at startup)
# ---------------------------------------------------------------------------

class AppState:
    embedder = None
    faiss_index = None
    bm25 = None
    chunks: List[dict] = []
    llm = None
    retriever: HybridRetriever | None = None


state = AppState()


def _build_retriever() -> None:
    """Rebuild the HybridRetriever from current state."""
    state.retriever = HybridRetriever(
        embedder=state.embedder,
        faiss_index=state.faiss_index,
        bm25=state.bm25,
        chunks=state.chunks,
    )


def load_indexes() -> None:
    """Load FAISS, BM25, and chunks from disk. Raises if files are missing."""
    missing = [
        p for p in (settings.FAISS_INDEX_PATH, settings.BM25_PATH, settings.CHUNKS_PATH)
        if not p.exists()
    ]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise RuntimeError(
            f"Index files missing ({names}). Run `python ingest.py` first."
        )

    print("[server] Loading chunk metadata …")
    state.chunks = json.loads(settings.CHUNKS_PATH.read_text(encoding="utf-8"))

    print("[server] Loading FAISS index …")
    import faiss
    state.faiss_index = faiss.read_index(str(settings.FAISS_INDEX_PATH))

    print("[server] Loading BM25 index …")
    import pickle
    with open(settings.BM25_PATH, "rb") as f:
        state.bm25 = pickle.load(f)


def tokenize_bm25(text: str) -> list[str]:
    """Simple BM25 tokenizer (kept for legacy compatibility)."""
    return re.findall(r"[a-z0-9][a-z0-9_\-]*", text.lower())


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def load_everything() -> None:
    load_indexes()

    print(f"[server] Loading embedding model from {settings.EMBED_MODEL_PATH} …")
    state.embedder = get_embedder()

    print(f"[server] Loading LLM from {settings.LLM_MODEL_PATH} …")
    from llama_cpp import Llama
    state.llm = Llama(
        model_path=settings.LLM_MODEL_PATH,
        n_ctx=settings.N_CTX,
        n_gpu_layers=settings.N_GPU_LAYERS,
        verbose=False,
    )

    # Pre-warm reranker (optional — loads in background, degrades gracefully)
    def _warm_reranker():
        try:
            get_reranker()
            print("[server] Reranker ready.")
        except Exception as exc:
            print(f"[server] Reranker unavailable (will use fallback): {exc}")

    threading.Thread(target=_warm_reranker, daemon=True).start()

    _build_retriever()

    files_count = len(set(c["file"] for c in state.chunks))
    print(f"[server] Ready. {len(state.chunks)} chunks from {files_count} file(s) indexed.")

    # Auto-open browser once
    if not os.environ.get("BROWSER_OPENED"):
        os.environ["BROWSER_OPENED"] = "1"
        threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8000/")).start()


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    metadata_filter: dict | None = None  # optional: {"document_type": "pdf", "service": "auth"}


# ---------------------------------------------------------------------------
# Routes — preserved for frontend compatibility
# ---------------------------------------------------------------------------

@app.get("/api/stats")
def stats():
    if not state.chunks:
        raise HTTPException(503, "Index not loaded")
    files = sorted(set(c["file"] for c in state.chunks))
    reranker_available = False
    try:
        r = get_reranker()
        reranker_available = r is not None and getattr(r, "available", False)
    except Exception:
        pass
    return {
        "chunks_indexed": len(state.chunks),
        "files_indexed": len(files),
        "file_names": files,
        "embed_model": Path(settings.EMBED_MODEL_PATH).name,
        "llm_model": Path(settings.LLM_MODEL_PATH).name,
        "retrieval_mode": "hybrid (FAISS dense + BM25 sparse, RRF fusion, cross-encoder rerank)",
        "reranker_available": reranker_available,
        "cache_stats": query_cache.cache_stats(),
    }


@app.post("/api/upload_runbook")
async def upload_runbook(file: UploadFile = File(...)):
    """Accept a runbook upload, save it, and incrementally update the index
    without triggering a full rebuild."""
    allowed = (".md", ".pdf", ".docx", ".pptx")
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Supported: {', '.join(allowed)}")

    content = await file.read()

    # Save to runbooks dir
    settings.RUNBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = settings.RUNBOOKS_DIR / file.filename
    out_path.write_bytes(content)

    # Incremental index update (O(new chunks), not O(total corpus))
    try:
        new_chunks, new_faiss, new_bm25 = incremental_add(
            file_path=out_path,
            existing_chunks=state.chunks,
            existing_faiss_index=state.faiss_index,
            existing_bm25=state.bm25,
        )
        state.chunks = new_chunks
        state.faiss_index = new_faiss
        state.bm25 = new_bm25
        save_state(state.chunks, state.faiss_index, state.bm25)
        _build_retriever()

        # Invalidate caches since index has changed
        query_cache.invalidate_retrieval_caches()

    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Indexing failed: {exc}")

    return {"status": "success", "filename": file.filename,
            "total_chunks": len(state.chunks)}


@app.post("/api/admin/rebuild")
async def admin_rebuild():
    """Trigger a full index rebuild from the runbooks/ directory.
    Use only when explicitly needed (e.g., after deleting files from runbooks/).
    Normal uploads use incremental indexing."""
    try:
        chunks, faiss_index, bm25 = build_full_index(settings.RUNBOOKS_DIR)
        state.chunks = chunks
        state.faiss_index = faiss_index
        state.bm25 = bm25
        save_state(state.chunks, state.faiss_index, state.bm25)
        _build_retriever()
        query_cache.invalidate_retrieval_caches()
        files_count = len(set(c["file"] for c in state.chunks))
        return {"status": "success", "chunks": len(state.chunks), "files": files_count}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Rebuild failed: {exc}")


@app.get("/api/metrics")
def metrics():
    """Return per-stage timing logs and aggregate stats. Does not affect
    existing /api/query or /api/stats response shapes."""
    return get_metrics()


@app.post("/api/query")
def query(req: QueryRequest):
    """Main query endpoint. Returns SSE stream. Response shape is preserved
    for frontend compatibility."""
    q = req.query.strip()
    if not q:
        raise HTTPException(400, "Empty query")

    request_id = str(uuid.uuid4())[:8]
    print(f"[server] Query [{request_id}]: {q[:80]}")

    def event_stream():
        log = RequestLog(
            request_id=request_id,
            query_raw=q,
        )
        t_total_start = time.time()

        # ── 1. Normalize query ──────────────────────────────────────────────
        with RequestTimer("normalization") as t_norm:
            q_normalized = normalize(q)
        log.query_normalized = q_normalized
        log.timings_ms["normalization"] = t_norm.ms

        # ── 2. Cache check (full response) ──────────────────────────────────
        filter_hash = json.dumps(req.metadata_filter or {}, sort_keys=True)
        ck = query_cache.cache_key(q_normalized, filter_hash)

        cached_response = query_cache.get_response(ck)
        cached_sources = query_cache.get_rerank(ck)
        if cached_response and cached_sources:
            log.cache_hit = True
            log.timings_ms["total"] = int((time.time() - t_total_start) * 1000)
            log_request(log)
            # Stream cached response to keep SSE shape identical
            yield f"data: {json.dumps({'type': 'sources', 'sources': cached_sources, 'retrieval_ms': 0, 'cache_hit': True})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'text': cached_response})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'generation_ms': 0, 'cache_hit': True})}\n\n"
            return

        # ── 3. Parallel hybrid retrieval ────────────────────────────────────
        with RequestTimer("retrieval") as t_retrieval:
            try:
                loop = asyncio.new_event_loop()
                candidates, top1_cosine, _ = loop.run_until_complete(
                    state.retriever.retrieve(q_normalized, req.metadata_filter)
                )
                loop.close()
            except Exception as exc:
                print(f"[server] Retrieval error: {exc}")
                candidates, top1_cosine = [], 0.0
        log.timings_ms["retrieval"] = t_retrieval.ms
        log.chunk_counts["rrf_output"] = len(candidates)
        log.confidence["top1_cosine"] = round(top1_cosine, 4)

        retrieval_ms = t_retrieval.ms

        # ── 4. Confidence gate ──────────────────────────────────────────────
        if not candidates or top1_cosine < settings.CONFIDENCE_THRESHOLD:
            payload = {
                "type": "no_match",
                "message": (
                    "No runbook in the index matches this issue with enough "
                    "confidence to answer safely. Recommend escalating to the "
                    "on-call lead or checking the internal wiki for undocumented "
                    "procedures."
                ),
                "top1_cosine": round(top1_cosine, 3),
                "retrieval_ms": retrieval_ms,
            }
            log.timings_ms["total"] = int((time.time() - t_total_start) * 1000)
            log_request(log)
            yield f"data: {json.dumps(payload)}\n\n"
            return

        # ── 5. Rerank + adaptive depth + dedup ─────────────────────────────
        with RequestTimer("reranking") as t_rerank:
            try:
                reranked = rerank(q_normalized, candidates, state.embedder)
            except Exception as exc:
                print(f"[server] Reranker error (fallback): {exc}")
                reranked = candidates[:settings.TOP_K_FINAL_MAX]
        log.timings_ms["reranking"] = t_rerank.ms
        log.chunk_counts["after_rerank"] = len(reranked)

        top_reranker_score = reranked[0].get("reranker_score", 0.0) if reranked else 0.0
        log.confidence["reranker_top_score"] = round(top_reranker_score, 4)

        # Cache reranked results
        query_cache.set_rerank(ck, reranked)

        # ── 6. Context compression ──────────────────────────────────────────
        with RequestTimer("compression") as t_compress:
            compressed = compress_chunks(reranked)
        log.timings_ms["compression"] = t_compress.ms

        # ── 7. Build prompt ─────────────────────────────────────────────────
        with RequestTimer("prompt_build") as t_prompt:
            cached_prompt = query_cache.get_prompt(ck)
            if cached_prompt:
                prompt = cached_prompt
            else:
                prompt = build_prompt(q_normalized, compressed)
                query_cache.set_prompt(ck, prompt)
        log.timings_ms["prompt_build"] = t_prompt.ms
        log.chunk_counts["sent_to_llm"] = len(compressed)

        # ── 8. Stream SSE: sources first ────────────────────────────────────
        yield f"data: {json.dumps({'type': 'sources', 'sources': reranked, 'retrieval_ms': retrieval_ms})}\n\n"

        # ── 9. LLM generation ───────────────────────────────────────────────
        raw_text = ""
        t_llm_start = time.time()
        try:
            stream = state.llm(
                prompt,
                max_tokens=settings.MAX_NEW_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
                top_p=settings.LLM_TOP_P,
                stop=["<|im_end|>"],
                stream=True,
            )
            for chunk in stream:
                text = chunk["choices"][0]["text"]
                if text:
                    raw_text += text
                    yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"
        except Exception as exc:
            print(f"[server] LLM error: {exc}")
            err_msg = f"LLM generation failed: {exc}"
            yield f"data: {json.dumps({'type': 'token', 'text': err_msg})}\n\n"
            raw_text = err_msg

        generation_ms = int((time.time() - t_llm_start) * 1000)
        log.timings_ms["llm_inference"] = generation_ms
        log.timings_ms["total"] = int((time.time() - t_total_start) * 1000)
        log.memory_mb = get_memory_mb()

        # Cache the full response
        query_cache.set_response(ck, raw_text)

        yield f"data: {json.dumps({'type': 'done', 'generation_ms': generation_ms})}\n\n"

        log_request(log)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Static routes (kept identical)
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(
        settings.STATIC_DIR / "index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")