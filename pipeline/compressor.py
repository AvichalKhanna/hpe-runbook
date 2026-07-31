import re
from typing import List

# Boilerplate filler patterns — these add no information
_FILLER_RE = re.compile(
    r'^(This section (describes|covers|explains|provides)|'
    r'The following (steps|procedure|commands) (describe|explain|show)|'
    r'For more information(,| see| refer)|'
    r'Please (note|be aware|ensure) that|'
    r'Click (here|ok|next|finish) to|'
    r'This document (describes|covers|explains))[\s,.]',
    re.I
)

def compress_chunk(text: str) -> str:
    """
    Light compression: only remove duplicate blank lines and pure boilerplate filler.
    Preserve ALL commands, steps, prose explanations, table rows, and code blocks.
    The LLM needs the full context to give a relevant answer.
    """
    lines = text.split('\n')
    compressed = []
    prev_blank = False

    for line in lines:
        line_s = line.strip()

        # Collapse multiple consecutive blank lines into one
        if not line_s:
            if not prev_blank:
                compressed.append('')
            prev_blank = True
            continue
        prev_blank = False

        # Skip pure boilerplate filler sentences (no content value)
        if _FILLER_RE.match(line_s):
            continue

        # Keep everything else: commands, prose, headings, tables, code, steps
        compressed.append(line)

    return '\n'.join(compressed).strip()


def compress_chunks(chunks: List[dict]) -> List[dict]:
    compressed_chunks = []
    for c in chunks:
        c_new = c.copy()
        c_new['text'] = compress_chunk(c['text'])
        compressed_chunks.append(c_new)
    return compressed_chunks
