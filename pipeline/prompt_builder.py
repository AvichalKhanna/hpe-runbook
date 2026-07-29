from typing import List

SYSTEM_PROMPT = """You are an SRE assistant. Answer ONLY from the provided [Source N] runbook excerpts.
Cite every factual claim with [Source N].
If excerpts lack sufficient evidence, state exactly: "Insufficient evidence in the indexed runbooks to safely answer this question." and suggest escalation.
Never fabricate commands, thresholds, contacts, file paths, or error codes.
Use numbered steps for remediation procedures.
Be concise."""

def build_prompt(query: str, sources: List[dict]) -> str:
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nRunbook excerpts:\n\n"
    
    for i, src in enumerate(sources, 1):
        prompt += f"[Source {i}] File: {src.get('file', 'unknown')} | Section: {src.get('section', 'unknown')} | Lines {src.get('start_line', '?')}-{src.get('end_line', '?')}\n"
        prompt += f"{src.get('text', '')}\n\n"
        
    prompt += f"On-call engineer's question: {query}<|im_end|>\n<|im_start|>assistant\n"
    return prompt
