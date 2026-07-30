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
      → conflict detection between top sources
      → deduplicate near-identical chunks
      → context compression (pipeline.compressor)
      → structured prompt with multi-turn history (pipeline.prompt_builder)
      → Qwen2.5-3B-Instruct GGUF streamed via llama-cpp-python
      → answer + citations + confidence (SSE stream)

New endpoints (v2):
    POST /api/feedback              — thumbs up/down feedback
    GET  /api/runbook/{filename}    — raw runbook file content
    GET  /api/system_metrics        — live CPU/RAM/Disk via psutil
    POST /api/patch_gap             — web search + AI runbook creation
    WS   /ws/shell                  — live shell wrapper (cmd.exe)

Run with:
    python -m uvicorn server:app --port 8000
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import subprocess
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# Bootstrap sys.path
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
from pipeline.abbrev_miner import update_from_text               # noqa: E402
from pipeline import cache as query_cache                        # noqa: E402
from observability.logger import (                               # noqa: E402
    RequestLog, RequestTimer, log_request, get_metrics, get_memory_mb,
)


# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------

app = FastAPI(title="Runbook RAG Chatbot")

FEEDBACK_PATH = BASE_DIR / "logs" / "feedback.jsonl"
FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)


class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


app.add_middleware(NoCacheMiddleware)


# ---------------------------------------------------------------------------
# Global state
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
    state.retriever = HybridRetriever(
        embedder=state.embedder,
        faiss_index=state.faiss_index,
        bm25=state.bm25,
        chunks=state.chunks,
    )


def load_indexes() -> None:
    missing = [
        p for p in (settings.FAISS_INDEX_PATH, settings.BM25_PATH, settings.CHUNKS_PATH)
        if not p.exists()
    ]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise RuntimeError(f"Index files missing ({names}). Run `python ingest.py` first.")

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
    return re.findall(r"[a-z0-9][a-z0-9_\-]*", text.lower())


# ---------------------------------------------------------------------------
# Conflict detection helper
# ---------------------------------------------------------------------------

def _detect_conflicts(sources: list[dict]) -> list[dict]:
    """
    Heuristic conflict detection between source chunks.
    Marks sources with conflict=True if contradictory signals are found
    between any two chunks from different files.
    """
    if len(sources) < 2:
        return sources

    # Patterns that suggest contradictory advice
    ENABLE_RE  = re.compile(r'\b(enable|set to true|turn on|activate|use)\b', re.I)
    DISABLE_RE = re.compile(r'\b(disable|set to false|turn off|deactivate|avoid)\b', re.I)

    # Extract numeric values from text for comparison
    NUM_RE = re.compile(r'\b(\d+)\s*(ms|s|seconds|minutes|mb|gb|%|connections?|threads?|retries?)\b', re.I)

    def get_nums(text: str) -> dict[str, set]:
        nums: dict[str, set] = {}
        for m in NUM_RE.finditer(text):
            unit = m.group(2).lower().rstrip('s')
            nums.setdefault(unit, set()).add(int(m.group(1)))
        return nums

    conflicted = set()

    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            si, sj = sources[i], sources[j]
            ti, tj = si.get("text", ""), sj.get("text", "")
            # Different files only
            if si.get("file") == sj.get("file"):
                continue
            # Check enable/disable flip
            i_enable  = bool(ENABLE_RE.search(ti))
            i_disable = bool(DISABLE_RE.search(ti))
            j_enable  = bool(ENABLE_RE.search(tj))
            j_disable = bool(DISABLE_RE.search(tj))
            if (i_enable and j_disable) or (i_disable and j_enable):
                conflicted.add(i)
                conflicted.add(j)
            # Check numeric value conflicts (same unit, different values)
            ni, nj = get_nums(ti), get_nums(tj)
            for unit in ni:
                if unit in nj and ni[unit] != nj[unit]:
                    conflicted.add(i)
                    conflicted.add(j)

    result = []
    for idx, src in enumerate(sources):
        s = dict(src)
        s["conflict"] = idx in conflicted
        result.append(s)
    return result


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

    if not os.environ.get("BROWSER_OPENED"):
        os.environ["BROWSER_OPENED"] = "1"
        threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8000/")).start()


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    metadata_filter: dict | None = None
    conversation_history: list | None = None   # multi-turn: [{role, content}, …]
    shell_context: str | None = None           # last N lines from the shell panel
    mode: str = "descriptive"                  # "descriptive" or "stepwise"


class FeedbackRequest(BaseModel):
    request_id: str
    query: str
    rating: int          # 1 = thumbs up, -1 = thumbs down
    comment: str | None = None


class PatchGapRequest(BaseModel):
    query: str


# ---------------------------------------------------------------------------
# Existing routes
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
    allowed = (".md", ".pdf", ".docx", ".pptx")
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Supported: {', '.join(allowed)}")

    content = await file.read()
    settings.RUNBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = settings.RUNBOOKS_DIR / file.filename
    out_path.write_bytes(content)

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
        query_cache.invalidate_retrieval_caches()
        # Mine abbreviations from uploaded file
        try:
            update_from_text(content.decode("utf-8", errors="ignore"))
        except Exception:
            pass
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Indexing failed: {exc}")

    return {"status": "success", "filename": file.filename, "total_chunks": len(state.chunks)}


@app.post("/api/admin/rebuild")
async def admin_rebuild():
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
    return get_metrics()


# ---------------------------------------------------------------------------
# NEW: Feedback endpoint
# ---------------------------------------------------------------------------

@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    """Store thumbs up/down feedback to logs/feedback.jsonl."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": req.request_id,
        "query": req.query,
        "rating": req.rating,
        "comment": req.comment or "",
    }
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# NEW: Runbook file content endpoint
# ---------------------------------------------------------------------------

@app.get("/api/runbook/{filename:path}")
def get_runbook_content(filename: str):
    """Serve the raw text content of a runbook file for the split-screen viewer."""
    # Security: only allow files that are actually in the runbooks directory
    safe_name = Path(filename).name
    candidate = settings.RUNBOOKS_DIR / safe_name
    if not candidate.exists():
        raise HTTPException(404, f"Runbook '{safe_name}' not found")
    try:
        # For text-based files return as-is; for binary files return placeholder
        ext = candidate.suffix.lower()
        if ext in (".md", ".txt"):
            text = candidate.read_text(encoding="utf-8", errors="replace")
        elif ext in (".pdf", ".docx", ".pptx"):
            # Extract plain text using the chunker's existing parsers
            from pipeline.chunker import chunk_file
            chunks = chunk_file(candidate, start_id=0)
            text = "\n\n---\n\n".join(
                f"## {c.section}\n\n{c.text}" for c in chunks
            )
        else:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        return JSONResponse({"filename": safe_name, "content": text, "type": ext.lstrip(".")})
    except Exception as exc:
        raise HTTPException(500, f"Could not read file: {exc}")


# ---------------------------------------------------------------------------
# NEW: Live system metrics endpoint
# ---------------------------------------------------------------------------

@app.get("/api/system_metrics")
def system_metrics():
    """Return real-time CPU, RAM, and Disk usage via psutil."""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.2)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage(os.path.abspath(os.sep))
        
        # Windows GPU usage is hard to fetch without heavy libs; using CPU as a proxy/placeholder
        gpu_percent = round(cpu, 1)

        return {
            "gpu_percent": gpu_percent,
            "cpu_percent": round(cpu, 1),
            "ram_used_gb": round(mem.used / 1e9, 2),
            "ram_total_gb": round(mem.total / 1e9, 2),
            "ram_percent": round(mem.percent, 1),
            "disk_used_gb": round(disk.used / 1e9, 2),
            "disk_total_gb": round(disk.total / 1e9, 2),
            "disk_percent": round(disk.percent, 1),
        }
    except ImportError:
        return {"error": "psutil not installed"}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# NEW: Patch-Gap — web search + auto runbook creation
# ---------------------------------------------------------------------------

@app.post("/api/patch_gap")
async def patch_gap(req: PatchGapRequest):
    """
    When no confident match exists:
    1. Search DuckDuckGo for the query.
    2. Scrape top results with trafilatura.
    3. Create an AI-GENERATED / UNVERIFIED runbook .md in runbooks/.
    4. Incrementally add it to the index.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "Empty query")

    # Slugify for filename
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower())[:60].strip("-")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"ai_generated_{slug}_{ts}.md"

    results_text = []

    # ── Step 1: DuckDuckGo search ────────────────────────────────────────────
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=5))
    except Exception as exc:
        hits = []
        print(f"[patch_gap] DDGS search failed: {exc}")

    # ── Step 2: Scrape with trafilatura ──────────────────────────────────────
    try:
        import trafilatura
        for hit in hits[:4]:
            url = hit.get("href") or hit.get("url", "")
            if not url:
                continue
            try:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    extracted = trafilatura.extract(
                        downloaded,
                        include_comments=False,
                        include_tables=True,
                        no_fallback=False,
                    )
                    if extracted and len(extracted) > 100:
                        results_text.append(f"### Source: {url}\n\n{extracted[:3000]}")
            except Exception:
                pass
    except ImportError:
        # trafilatura not installed — use raw body snippets from search results
        for hit in hits:
            body = hit.get("body", "")
            if body:
                results_text.append(f"### Source: {hit.get('href','')}\n\n{body}")

    if not results_text:
        raise HTTPException(422, "Could not retrieve any web content for this query.")

    # ── Step 3: Build runbook markdown ───────────────────────────────────────
    combined_body = "\n\n".join(results_text)
    runbook_content = f"""# {query.title()}

> [!WARNING]
> **AI-GENERATED · UNVERIFIED** — This runbook was auto-created from web sources on {ts}.
> Review and validate before using in production.

## Overview

This runbook was automatically generated by Stratum AI after no confident match was found
in the indexed runbooks. The information below was gathered from public web sources.

## Web-Sourced Information

{combined_body}

## Next Steps

1. Verify all commands and configurations against your environment.
2. Have a senior SRE review this runbook before promoting it.
3. Remove the `AI-GENERATED` tag once validated.
"""

    # ── Step 4: Save + index ─────────────────────────────────────────────────
    settings.RUNBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = settings.RUNBOOKS_DIR / filename
    out_path.write_text(runbook_content, encoding="utf-8")

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
        query_cache.invalidate_retrieval_caches()
        # Mine abbreviations from new content
        try:
            update_from_text(runbook_content)
        except Exception:
            pass
    except Exception as exc:
        out_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Failed to index generated runbook: {exc}")

    return {
        "status": "ok",
        "filename": filename,
        "sources_scraped": len(results_text),
        "message": f"Created and indexed '{filename}'. Re-ask your question to use it.",
    }


# ---------------------------------------------------------------------------
# Main query endpoint (updated with multi-turn, conflict, near-misses)
# ---------------------------------------------------------------------------

@app.post("/api/query")
def query(req: QueryRequest):
    q = req.query.strip()
    if not q:
        raise HTTPException(400, "Empty query")

    request_id = str(uuid.uuid4())[:8]
    print(f"[server] Query [{request_id}]: {q[:80]}")

    def event_stream():
        log = RequestLog(request_id=request_id, query_raw=q)
        t_total_start = time.time()

        # ── 1. Normalize ─────────────────────────────────────────────────────
        with RequestTimer("normalization") as t_norm:
            # Prepend shell context to query if provided
            full_q = q
            if req.shell_context:
                full_q = f"{q}\n\n[Shell output for context]:\n{req.shell_context[-2000:]}"
            q_normalized = normalize(full_q)
        log.query_normalized = q_normalized
        log.timings_ms["normalization"] = t_norm.ms

        # ── 2. Cache check ───────────────────────────────────────────────────
        filter_hash = json.dumps(req.metadata_filter or {}, sort_keys=True)
        ck = query_cache.cache_key(q_normalized, filter_hash)

        cached_response = query_cache.get_response(ck)
        cached_sources = query_cache.get_rerank(ck)
        if cached_response and cached_sources and not req.conversation_history:
            log.cache_hit = True
            log.timings_ms["total"] = int((time.time() - t_total_start) * 1000)
            log_request(log)
            yield f"data: {json.dumps({'type': 'sources', 'sources': cached_sources, 'retrieval_ms': 0, 'cache_hit': True})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'text': cached_response})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'generation_ms': 0, 'cache_hit': True})}\n\n"
            return

        # ── 3. Hybrid retrieval ──────────────────────────────────────────────
        with RequestTimer("retrieval") as t_retrieval:
            try:
                candidates, top1_cosine, _ = state.retriever.retrieve(q_normalized, req.metadata_filter)
            except Exception as exc:
                print(f"[server] Retrieval error: {exc}")
                candidates, top1_cosine = [], 0.0
        log.timings_ms["retrieval"] = t_retrieval.ms
        log.chunk_counts["rrf_output"] = len(candidates)
        log.confidence["top1_cosine"] = round(top1_cosine, 4)
        retrieval_ms = t_retrieval.ms

        # ── 4. Confidence gate with near-miss ────────────────────────────────
        if not candidates or top1_cosine < settings.CONFIDENCE_THRESHOLD:
            near_misses = []
            for c in candidates[:3]:
                near_misses.append({
                    "file": c.get("file", ""),
                    "section": c.get("section", ""),
                    "score": round(c.get("rrf_score", top1_cosine), 4),
                    "text_preview": c.get("text", "")[:200],
                })
            payload = {
                "type": "no_match",
                "message": (
                    "No runbook in the index matches this issue with enough "
                    "confidence to answer safely. Recommend escalating to the "
                    "on-call lead or checking the internal wiki for undocumented "
                    "procedures."
                ),
                "top1_cosine": round(top1_cosine, 3),
                "threshold": settings.CONFIDENCE_THRESHOLD,
                "near_misses": near_misses,
                "retrieval_ms": retrieval_ms,
            }
            log.timings_ms["total"] = int((time.time() - t_total_start) * 1000)
            log_request(log)
            yield f"data: {json.dumps(payload)}\n\n"
            return

        # ── 5. Rerank ────────────────────────────────────────────────────────
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

        # ── 5b. Conflict detection ───────────────────────────────────────────
        reranked = _detect_conflicts(reranked)

        query_cache.set_rerank(ck, reranked)

        # ── 6. Compression ───────────────────────────────────────────────────
        with RequestTimer("compression") as t_compress:
            compressed = compress_chunks(reranked)
        log.timings_ms["compression"] = t_compress.ms

        # ── 7. Build prompt (with multi-turn history) ─────────────────────────
        with RequestTimer("prompt_build") as t_prompt:
            cached_prompt = query_cache.get_prompt(ck)
            if cached_prompt and not req.conversation_history:
                prompt = cached_prompt
            else:
                prompt = build_prompt(
                    q_normalized,
                    compressed,
                    conversation_history=req.conversation_history,
                    mode=req.mode,
                )
                if not req.conversation_history:
                    query_cache.set_prompt(ck, prompt)
        log.timings_ms["prompt_build"] = t_prompt.ms
        log.chunk_counts["sent_to_llm"] = len(compressed)

        # ── 8. SSE: sources first ─────────────────────────────────────────────
        yield f"data: {json.dumps({'type': 'sources', 'sources': reranked, 'retrieval_ms': retrieval_ms, 'request_id': request_id})}\n\n"

        # ── 9. LLM generation ─────────────────────────────────────────────────
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
        query_cache.set_response(ck, raw_text)

        yield f"data: {json.dumps({'type': 'done', 'generation_ms': generation_ms, 'request_id': request_id})}\n\n"
        log_request(log)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# NEW: WebSocket shell (cmd.exe wrapper)
# ---------------------------------------------------------------------------

@app.websocket("/ws/shell")
async def shell_ws(websocket: WebSocket):
    """
    WebSocket terminal — spawns a cmd.exe subprocess and bridges
    stdin/stdout between the browser terminal and the process.
    """
    await websocket.accept()
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            "cmd.exe",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        async def read_output():
            """Read process stdout and send to websocket."""
            try:
                while True:
                    data = await proc.stdout.read(1024)
                    if not data:
                        break
                    await websocket.send_text(data.decode("cp850", errors="replace"))
            except Exception:
                pass

        # Start reading task
        read_task = asyncio.create_task(read_output())

        # Receive commands from browser
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=300)
            except asyncio.TimeoutError:
                break
            except WebSocketDisconnect:
                break
            if proc.stdin and not proc.stdin.is_closing():
                proc.stdin.write((msg + "\r\n").encode("utf-8", errors="replace"))
                await proc.stdin.drain()

        read_task.cancel()

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_text(f"\r\n[Shell error: {exc}]\r\n")
        except Exception:
            pass
    finally:
        if proc:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Static routes
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