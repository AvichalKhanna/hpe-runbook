from typing import List

SYSTEM_PROMPT = """You are an SRE runbook assistant. Your ONLY knowledge source is the [Source N] excerpts provided below.
Rules:
1. Answer using ONLY information present verbatim in the excerpts. Copy exact commands, thresholds, paths, and error codes from the sources.
2. Cite every fact with [Source N] inline.
3. If the excerpts do not contain enough information, say exactly: "Insufficient evidence in the indexed runbooks." Do NOT guess or use general knowledge.
4. Use numbered steps for remediation procedures.
5. Be precise and concise."""

STEPWISE_SYSTEM_PROMPT = """You are an SRE runbook assistant providing STEP-BY-STEP remediation guidance.
Your ONLY knowledge source is the [Source N] excerpts provided below.
Rules:
1. Output EXACTLY ONE step at a time, copied from the source excerpts.
2. Cite the source with [Source N].
3. After outputting the step, stop and wait for the user's confirmation or error report.
4. If the user reports an error, recalibrate using the excerpts and provide the corrected action.
5. If the excerpts lack evidence, say: "Insufficient evidence to proceed. Escalate."
"""

def build_prompt(query: str, sources: List[dict], conversation_history: list | None = None, mode: str = "descriptive") -> str:
    """
    Build the full prompt for the LLM.

    conversation_history: list of {"role": "user"|"assistant", "content": str}
    mode: "descriptive" (default) or "stepwise"
    """
    sys_prompt = STEPWISE_SYSTEM_PROMPT if mode == "stepwise" else SYSTEM_PROMPT
    prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"

    # ── Multi-turn history ───────────────────────────────────────────────────
    if conversation_history:
        for turn in conversation_history[-3:]:
            role = turn.get("role", "user")
            content = turn.get("content", "").strip()
            if role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"

    # ── Current turn ─────────────────────────────────────────────────────────
    prompt += "<|im_start|>user\n"
    prompt += "=== RUNBOOK EXCERPTS (answer only from these) ===\n\n"

    for i, src in enumerate(sources, 1):
        prompt += (
            f"[Source {i}] File: {src.get('file', 'unknown')} | "
            f"Section: {src.get('section', 'unknown')} | "
            f"Lines {src.get('start_line', '?')}-{src.get('end_line', '?')}\n"
            f"---\n"
        )
        prompt += f"{src.get('text', '').strip()}\n\n"

    prompt += f"=== QUESTION ===\n{query}<|im_end|>\n<|im_start|>assistant\n"
    return prompt
