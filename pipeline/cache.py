import hashlib
from collections import OrderedDict
import logging

from config.settings import CACHE_MAX_SIZE, CACHE_ENABLED

logger = logging.getLogger(__name__)

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: str, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
            
    def clear(self):
        self.cache.clear()
        
    def __len__(self):
        return len(self.cache)

_query_norm_cache = LRUCache(CACHE_MAX_SIZE)
_retrieval_cache = LRUCache(CACHE_MAX_SIZE)
_rerank_cache = LRUCache(CACHE_MAX_SIZE)
_prompt_cache = LRUCache(CACHE_MAX_SIZE)
_response_cache = LRUCache(CACHE_MAX_SIZE)

def cache_key(normalized_query: str, filter_hash: str = "") -> str:
    return hashlib.sha256(f"{normalized_query}|{filter_hash}".encode('utf-8')).hexdigest()

def get_norm(raw_query: str) -> str | None:
    return _query_norm_cache.get(raw_query) if CACHE_ENABLED else None

def set_norm(raw_query: str, normalized: str) -> None:
    if CACHE_ENABLED:
        _query_norm_cache.set(raw_query, normalized)

def get_retrieval(key: str) -> list | None:
    return _retrieval_cache.get(key) if CACHE_ENABLED else None

def set_retrieval(key: str, candidates: list) -> None:
    if CACHE_ENABLED:
        _retrieval_cache.set(key, candidates)

def get_rerank(key: str) -> list | None:
    return _rerank_cache.get(key) if CACHE_ENABLED else None

def set_rerank(key: str, reranked: list) -> None:
    if CACHE_ENABLED:
        _rerank_cache.set(key, reranked)

def get_prompt(key: str) -> str | None:
    return _prompt_cache.get(key) if CACHE_ENABLED else None

def set_prompt(key: str, prompt: str) -> None:
    if CACHE_ENABLED:
        _prompt_cache.set(key, prompt)

def get_response(key: str) -> str | None:
    return _response_cache.get(key) if CACHE_ENABLED else None

def set_response(key: str, response: str) -> None:
    if CACHE_ENABLED:
        _response_cache.set(key, response)

def invalidate_retrieval_caches() -> None:
    _retrieval_cache.clear()
    _rerank_cache.clear()
    _prompt_cache.clear()
    _response_cache.clear()
    logger.info("Invalidated retrieval caches")

def cache_stats() -> dict:
    return {
        "query_norm": len(_query_norm_cache),
        "retrieval": len(_retrieval_cache),
        "rerank": len(_rerank_cache),
        "prompt": len(_prompt_cache),
        "response": len(_response_cache)
    }
