from typing import List

SYSTEM_PROMPT = """You are an SRE assistant. Answer ONLY from the provided [Source N] runbook excerpts.
Cite every factual claim with [Source N].
If excerpts lack sufficient evidence, state exactly: "Insufficient evidence in the indexed runbooks to safely answer this question." and suggest escalation.
Never fabricate commands, thresholds, contacts, file paths, or error codes.
Use numbered steps for remediation procedures.
Be concise."""

STEPWISE_SYSTEM_PROMPT = """You are an SRE assistant providing STEP-BY-STEP remediation guidance.
Answer ONLY from the provided [Source N] runbook excerpts.
CRITICAL INSTRUCTION: Output EXACTLY ONE step at a time.
Wait for the user's feedback after each step before providing the next one.
If the user reports an error, recalibrate and provide the corrected next step.
If the excerpts lack sufficient evidence to continue, state exactly: "Insufficient evidence to proceed. Escalate."
Keep your response to a single step."""

def build_prompt(query: str, sources: List[dict], conversation_history: list | None = None, mode: str = "descriptive") -> str:
    """
    Build the full prompt for the LLM.

    conversation_history: list of {"role": "user"|"assistant", "content": str}
      representing prior turns in the current session. Injected before the
      current query so the model can answer follow-up questions coherently.
    mode: "descriptive" (default) or "stepwise" for step-by-step output.
    """
    if mode == "stepwise":
        sys_prompt = STEPWISE_SYSTEM_PROMPT
    else:
        sys_prompt = SYSTEM_PROMPT

    prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"

    # ── Multi-turn history ───────────────────────────────────────────────────
    if conversation_history:
        for turn in conversation_history[-3:]:   # keep last 3 turns for stepwise context
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

    prompt += f"On-call engineer's question (provide ONLY the immediate next step if in stepwise mode): {query}<|im_end|>\n<|im_start|>assistant\n"
    return prompt
