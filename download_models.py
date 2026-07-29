"""
download_models.py
------------------
Downloads all required models from HuggingFace Hub to the local models/ directory.
Run this once during installation. No network calls are made at runtime.

Usage:
    python download_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def download_embedding_model() -> bool:
    dest = MODELS_DIR / "MiniLM-L6-v2"
    if (dest / "model.safetensors").exists() or (dest / "pytorch_model.bin").exists():
        print("[download] Embedding model already present — skipping.")
        return True
    print("[download] Downloading embedding model (sentence-transformers/all-MiniLM-L6-v2, ~90 MB)...")
    try:
        snapshot_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        print("[download] Embedding model downloaded OK.")
        return True
    except Exception as e:
        print(f"[download] ERROR downloading embedding model: {e}", file=sys.stderr)
        return False


def download_reranker_model() -> bool:
    dest = MODELS_DIR / "bge-reranker-base"
    if (dest / "model.safetensors").exists() or (dest / "pytorch_model.bin").exists():
        print("[download] Reranker model already present — skipping.")
        return True
    print("[download] Downloading reranker model (BAAI/bge-reranker-base, ~280 MB)...")
    try:
        snapshot_download(
            repo_id="BAAI/bge-reranker-base",
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        print("[download] Reranker model downloaded OK.")
        return True
    except Exception as e:
        print(f"[download] WARNING: Could not download reranker model: {e}", file=sys.stderr)
        print("[download] The system will run without cross-encoder reranking (graceful degradation).")
        return False  # non-fatal — reranker degrades gracefully


def download_llm() -> bool:
    dest = MODELS_DIR / "Qwen2.5-3B-Instruct-Q3_K_M.gguf"
    if dest.exists() and dest.stat().st_size > 100_000_000:
        print("[download] LLM (Qwen2.5-3B GGUF) already present — skipping.")
        return True
    print("[download] Downloading LLM (Qwen2.5-3B-Instruct-Q3_K_M.gguf, ~1.5 GB)...")
    try:
        hf_hub_download(
            repo_id="bartowski/Qwen2.5-3B-Instruct-GGUF",
            filename="Qwen2.5-3B-Instruct-Q3_K_M.gguf",
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )
        print("[download] LLM downloaded OK.")
        return True
    except Exception as e:
        print(f"[download] ERROR downloading LLM: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    results = {
        "embedding_model": download_embedding_model(),
        "reranker_model":  download_reranker_model(),   # non-fatal if fails
        "llm":             download_llm(),
    }

    print("\n[download] Summary:")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name:<20} {status}")

    # Only fail hard if embedding model or LLM is missing (reranker is optional)
    if not results["embedding_model"] or not results["llm"]:
        print("\n[download] FATAL: Required models missing. Fix the errors above and re-run.")
        sys.exit(1)

    print("\n[download] All required models ready.")
