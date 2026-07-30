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

INTERACTIVE_SYSTEM_PROMPT = """You are an SRE voice assistant operating in Interactive Speech Mode.
Your goal is to guide the user through remediation step-by-step.
CRITICAL RULES:
1. Speak ONLY ONE step at a time. Never dump multiple steps.
2. Wait for the user to confirm they have completed the step before providing the next one.
3. Be conversational, extremely concise, and do not use complex formatting (no asterisks, hashes, or long URLs) since your response will be read aloud by a Text-to-Speech engine.
4. If the user indicates they are stuck in a loop, or a step fails 3 times, immediately suggest an ESCALATION protocol (e.g., escalating to L3 support).
5. Ground your steps ONLY in the provided runbook excerpts."""

def build_interactive_prompt(query: str, sources: List[dict], conversation_history: list | None = None) -> str:
    """Builds a prompt tailored for interactive voice mode."""
    prompt = f"<|im_start|>system\n{INTERACTIVE_SYSTEM_PROMPT}<|im_end|>\n"

    if conversation_history:
        for turn in conversation_history[-4:]: # Keep a bit more history for step context
            role = turn.get("role", "user")
            content = turn.get("content", "").strip()
            if role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"

    prompt += "<|im_start|>user\nRunbook excerpts:\n\n"
    for src in sources:
        prompt += f"{src.get('text', '')}\n\n"

    prompt += f"User's spoken input: {query}\n\nRespond with exactly ONE concise step or question.<|im_end|>\n<|im_start|>assistant\n"
    return prompt
