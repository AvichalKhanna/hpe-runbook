"""
config/settings.py
------------------
Central configuration for the Runbook RAG pipeline.
All values can be overridden via environment variables.
"""
from __future__ import annotations
import os
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
INDEX_DIR  = BASE_DIR / "index_store"
LOGS_DIR   = BASE_DIR / "logs"
CACHE_DIR  = BASE_DIR / "cache"
STATIC_DIR = BASE_DIR / "static"
RUNBOOKS_DIR = BASE_DIR / "runbooks"

# ── Model paths ──────────────────────────────────────────────────────────────
EMBED_MODEL_PATH   = str(MODELS_DIR / "MiniLM-L6-v2")
RERANKER_MODEL_PATH = str(MODELS_DIR / "bge-reranker-base")
LLM_MODEL_PATH     = str(MODELS_DIR / "Qwen2.5-3B-Instruct-Q3_K_M.gguf")

# ── Index paths ───────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
BM25_PATH        = INDEX_DIR / "bm25.pkl"
CHUNKS_PATH      = INDEX_DIR / "chunks.json"

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_MAX_WORDS     = int(os.getenv("CHUNK_MAX_WORDS", "180"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "40"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_DENSE      = int(os.getenv("TOP_K_DENSE",  "50"))
TOP_K_SPARSE     = int(os.getenv("TOP_K_SPARSE", "50"))
TOP_K_RRF        = int(os.getenv("TOP_K_RRF",   "30"))
TOP_K_FINAL_MIN  = int(os.getenv("TOP_K_FINAL_MIN", "3"))
TOP_K_FINAL_MAX  = int(os.getenv("TOP_K_FINAL_MAX", "8"))
RRF_K            = int(os.getenv("RRF_K", "60"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.32"))

# ── Reranker ─────────────────────────────────────────────────────────────────
RERANKER_HIGH_THRESHOLD  = float(os.getenv("RERANKER_HIGH_THRESHOLD", "0.7"))
RERANKER_LOW_THRESHOLD   = float(os.getenv("RERANKER_LOW_THRESHOLD",  "0.35"))
DEDUP_SIM_THRESHOLD      = float(os.getenv("DEDUP_SIM_THRESHOLD", "0.92"))

# ── LLM ──────────────────────────────────────────────────────────────────────
N_CTX          = int(os.getenv("N_CTX", "8192"))
N_GPU_LAYERS   = int(os.getenv("N_GPU_LAYERS", "0"))   # 0 = CPU-only; set to 99 for GPU/Vulkan builds
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TOP_P       = float(os.getenv("LLM_TOP_P", "0.9"))

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_MAX_SIZE   = int(os.getenv("CACHE_MAX_SIZE", "256"))
CACHE_ENABLED    = os.getenv("CACHE_ENABLED", "1") == "1"

# ── Observability ─────────────────────────────────────────────────────────────
LOG_ENABLED      = os.getenv("LOG_ENABLED", "1") == "1"
METRICS_HISTORY  = int(os.getenv("METRICS_HISTORY", "100"))  # keep last N request logs

# ── Embedding batch ───────────────────────────────────────────────────────────
EMBED_BATCH_SIZE  = int(os.getenv("EMBED_BATCH_SIZE", "32"))
EMBED_MAX_LENGTH  = int(os.getenv("EMBED_MAX_LENGTH", "256"))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── Groq / Speech Mode ───────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
