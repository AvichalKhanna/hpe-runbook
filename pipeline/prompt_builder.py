from typing import List

SYSTEM_PROMPT = """You are an SRE assistant. Answer ONLY from the provided [Source N] runbook excerpts.
Cite every factual claim with [Source N].
If excerpts lack sufficient evidence, state exactly: "Insufficient evidence in the indexed runbooks to safely answer this question." and suggest escalation.
Never fabricate commands, thresholds, contacts, file paths, or error codes.
Use numbered steps for remediation procedures.
Be concise."""


def build_prompt(query: str, sources: List[dict], conversation_history: list | None = None) -> str:
    """
    Build the full prompt for the LLM.

    conversation_history: list of {"role": "user"|"assistant", "content": str}
      representing prior turns in the current session. Injected before the
      current query so the model can answer follow-up questions coherently.
    """
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"

    # ── Multi-turn history ───────────────────────────────────────────────────
    if conversation_history:
        for turn in conversation_history[-2:]:   # keep last 2 turns max to stay inside context & reduce latency
            role = turn.get("role", "user")
            content = turn.get("content", "").strip()
            if role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"

    # ── Current turn ─────────────────────────────────────────────────────────
    prompt += "<|im_start|>user\nRunbook excerpts:\n\n"

    for i, src in enumerate(sources, 1):
        prompt += (
            f"[Source {i}] File: {src.get('file', 'unknown')} | "
            f"Section: {src.get('section', 'unknown')} | "
            f"Lines {src.get('start_line', '?')}-{src.get('end_line', '?')}\n"
        )
        prompt += f"{src.get('text', '')}\n\n"

    prompt += f"On-call engineer's question: {query}<|im_end|>\n<|im_start|>assistant\n"
    return prompt
