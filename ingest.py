"""
ingest.py
---------
CLI entry point for a FULL index rebuild from the runbooks/ directory.

For normal uploads, the server uses pipeline/indexer.incremental_add() instead
so no full rebuild is triggered.

Usage:
    python ingest.py                    # rebuild from runbooks/
    python ingest.py --dir /other/dir   # rebuild from a custom directory
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: make sure the project root is on sys.path so `config` and
# `pipeline` packages are importable when this script is run directly.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import settings                          # noqa: E402
from pipeline.indexer import build_full_index, save_state  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full FAISS+BM25 index from runbooks directory.")
    parser.add_argument(
        "--dir", type=Path, default=settings.RUNBOOKS_DIR,
        help="Directory containing runbook files (.md, .pdf, .docx, .pptx)",
    )
    args = parser.parse_args()

    runbooks_dir: Path = args.dir.resolve()

    if not runbooks_dir.exists():
        print(f"[ingest] ERROR: directory not found: {runbooks_dir}", file=sys.stderr)
        sys.exit(1)

    all_files = []
    for ext in ("*.md", "*.pdf", "*.docx", "*.pptx"):
        all_files.extend(runbooks_dir.glob(ext))
    all_files = sorted(all_files)

    if not all_files:
        print(f"[ingest] ERROR: no supported files (.md, .pdf, .docx, .pptx) found in {runbooks_dir}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[ingest] Found {len(all_files)} file(s) in {runbooks_dir}")
    for f in all_files:
        print(f"  - {f.name}")

    t0 = time.time()
    print("[ingest] Building full index …")

    try:
        chunks, faiss_index, bm25 = build_full_index(runbooks_dir)
    except Exception as exc:
        print(f"[ingest] ERROR during indexing: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if not chunks:
        print("[ingest] ERROR: produced 0 chunks — check file contents.", file=sys.stderr)
        sys.exit(1)

    save_state(chunks, faiss_index, bm25)

    elapsed = time.time() - t0
    files_indexed = len(set(c["file"] for c in chunks))
    print(f"[ingest] Done. {len(chunks)} chunks from {files_indexed} file(s) indexed in {elapsed:.1f}s.")
    print(f"[ingest] Index written to: {settings.INDEX_DIR}")
    print("[ingest] Start the server with: python -m uvicorn server:app --port 8000")


if __name__ == "__main__":
    main()