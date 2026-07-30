import asyncio
import time
from typing import List, Tuple, Optional, Dict
import numpy as np

from config.settings import TOP_K_DENSE, TOP_K_SPARSE, TOP_K_RRF, RRF_K

class HybridRetriever:
    def __init__(self, embedder, faiss_index, bm25, chunks: List[dict]):
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.bm25 = bm25
        self.chunks = chunks

    def retrieve(self, query: str, metadata_filter: Optional[Dict] = None) -> Tuple[List[dict], float, int]:
        start_time = time.time()
        
        candidates = []
        top1_cosine = 0.0
        
        # FAISS retrieval
        faiss_results = []
        if self.faiss_index:
            q_emb = self.embedder.encode([query])
            D, I = self.faiss_index.search(q_emb, TOP_K_DENSE)
            faiss_results = [(I[0][i], D[0][i]) for i in range(len(I[0])) if I[0][i] != -1]
            
        # BM25 retrieval
        bm25_results = []
        if self.bm25:
            import re
            tokenized_query = re.findall(r'\w+', query.lower())
            scores = self.bm25.get_scores(tokenized_query)
            top_n = np.argsort(scores)[::-1][:TOP_K_SPARSE]
            bm25_results = [(idx, scores[idx]) for idx in top_n if scores[idx] > 0]
        
        if faiss_results:
            top1_cosine = float(faiss_results[0][1])
            
        rrf_scores = {}
        for rank, (idx, _) in enumerate(faiss_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
            
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_RRF]
        
        candidates = [self.chunks[idx].copy() for idx, _ in sorted_indices]
        
        if metadata_filter:
            candidates = [c for c in candidates if all(c.get(k) == v for k, v in metadata_filter.items())]
            
        retrieval_ms = int((time.time() - start_time) * 1000)
        return candidates, top1_cosine, retrieval_ms
