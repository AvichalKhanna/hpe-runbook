import logging
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config.settings import RERANKER_MODEL_PATH, RERANKER_HIGH_THRESHOLD, RERANKER_LOW_THRESHOLD, DEDUP_SIM_THRESHOLD, TOP_K_FINAL_MIN, TOP_K_FINAL_MAX, TOP_K_RRF
from pipeline.embedder import get_embedder

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    def __init__(self):
        self.available = True
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_PATH, local_files_only=True)
            self.model.eval()
        except Exception as e:
            logger.warning(f"Reranker failed to load: {e}")
            self.available = False

    def score(self, query: str, texts: List[str]) -> List[float]:
        if not self.available or not texts:
            return []
        pairs = [[query, text] for text in texts]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
            scores = torch.sigmoid(scores).cpu().numpy().tolist()
        return scores

_reranker_instance = None

def get_reranker() -> CrossEncoderReranker | None:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance if _reranker_instance.available else None

def rerank(query: str, candidates: List[dict], embedder) -> List[dict]:
    reranker = get_reranker()
    if not reranker or not candidates:
        return candidates[:TOP_K_FINAL_MAX]
    
    texts = [c['text'] for c in candidates[:TOP_K_RRF]]
    scores = reranker.score(query, texts)
    
    for i, c in enumerate(candidates[:TOP_K_RRF]):
        c['reranker_score'] = scores[i]
        
    scored_candidates = sorted(candidates[:TOP_K_RRF], key=lambda x: x.get('reranker_score', 0.0), reverse=True)
    
    top_score = scored_candidates[0].get('reranker_score', 0.0)
    if top_score >= RERANKER_HIGH_THRESHOLD:
        keep = TOP_K_FINAL_MIN
    elif top_score >= RERANKER_LOW_THRESHOLD:
        keep = 5
    else:
        keep = TOP_K_FINAL_MAX
        
    initial_kept = scored_candidates[:keep]
    if len(initial_kept) <= 1:
        return initial_kept
        
    kept_texts = [c['text'] for c in initial_kept]
    embs = embedder.encode(kept_texts)
    
    final_kept = [initial_kept[0]]
    for i in range(1, len(initial_kept)):
        sims = np.dot(embs[i], embs[:len(final_kept)].T)
        if np.max(sims) <= DEDUP_SIM_THRESHOLD:
            final_kept.append(initial_kept[i])
            
    return final_kept
