"""
pipeline/abbrev_miner.py
------------------------
Auto-extract domain abbreviations from runbooks and merge into
config/abbreviations.json. Called during ingest and upload.

Patterns detected:
  - "full phrase (ABBR)"  e.g. "out-of-memory (OOM)"
  - "ABBR (full phrase)"  e.g. "OOM (out-of-memory)"
  - "ABBR — full phrase"  e.g. "CPU — central processing unit"
  - Explicit glossary lines: "ABBR: definition"

New entries are merged; existing manual entries are NEVER overwritten.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
ABBREV_PATH = BASE_DIR / "config" / "abbreviations.json"

# Patterns: (ABBR, definition) pairs
_PATTERNS = [
    # "full phrase (ABBR)"  — e.g. "out-of-memory (OOM)"
    re.compile(r'\b([a-z][a-z0-9\-\s]{2,40})\s+\(([A-Z]{2,10})\)', re.UNICODE),
    # "ABBR (full phrase)"  — e.g. "OOM (out-of-memory)"
    re.compile(r'\b([A-Z]{2,10})\s+\(([a-z][a-z0-9\-\s]{2,40})\)', re.UNICODE),
    # "ABBR — definition" or "ABBR: definition"
    re.compile(r'\b([A-Z]{2,10})\s*(?:—|-|:)\s+([a-z][a-z0-9\-\s]{3,50})', re.UNICODE),
]

# Words that look like abbreviations but aren't useful
_STOPWORDS = {
    "THE", "AND", "OR", "FOR", "NOT", "CAN", "ARE", "HAS", "THIS",
    "USE", "SET", "GET", "RUN", "LOG", "TRY", "SEE", "ADD", "MAY",
    "ALL", "ANY", "NEW", "OLD", "TWO", "ONE", "ITS", "OFF", "ON",
}


def _load_existing() -> dict:
    try:
        return json.loads(ABBREV_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(data: dict) -> None:
    ABBREV_PATH.parent.mkdir(parents=True, exist_ok=True)
    ABBREV_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def mine_text(text: str) -> dict[str, str]:
    """Extract abbreviation→definition pairs from a block of text."""
    found: dict[str, str] = {}
    for pat in _PATTERNS:
        for m in pat.finditer(text):
            g1, g2 = m.group(1).strip(), m.group(2).strip()
            # Identify which is abbr and which is definition
            if re.fullmatch(r'[A-Z]{2,10}', g1) and g1 not in _STOPWORDS:
                abbr, defn = g1.lower(), g2.lower()
            elif re.fullmatch(r'[A-Z]{2,10}', g2) and g2 not in _STOPWORDS:
                abbr, defn = g2.lower(), g1.lower()
            else:
                continue
            # Only keep if definition looks meaningful (>2 chars, has a space or is a real word)
            if len(defn) > 3:
                found[abbr] = defn
    return found


def mine_file(path: Path) -> dict[str, str]:
    """Mine a single file for abbreviations."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return mine_text(text)
    except Exception:
        return {}


def update_abbreviations(runbooks_dir: Path) -> int:
    """
    Mine all runbooks in runbooks_dir and merge new abbreviations into
    config/abbreviations.json. Returns count of NEW entries added.
    """
    existing = _load_existing()
    all_new: dict[str, str] = {}

    for ext in ("*.md", "*.pdf", "*.docx", "*.txt"):
        for f in runbooks_dir.glob(ext):
            try:
                mined = mine_file(f)
                all_new.update(mined)
            except Exception:
                pass

    added = 0
    for abbr, defn in all_new.items():
        if abbr not in existing:
            existing[abbr] = defn
            added += 1

    if added:
        _save(existing)
        print(f"[abbrev_miner] Added {added} new abbreviation(s) to config/abbreviations.json")

    return added


def update_from_text(text: str) -> int:
    """Mine a block of text and merge new abbreviations. Returns count added."""
    existing = _load_existing()
    mined = mine_text(text)
    added = 0
    for abbr, defn in mined.items():
        if abbr not in existing:
            existing[abbr] = defn
            added += 1
    if added:
        _save(existing)
    return added
