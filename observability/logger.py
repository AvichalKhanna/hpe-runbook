import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

from config.settings import LOGS_DIR, LOG_ENABLED, METRICS_HISTORY

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

LOGS_DIR.mkdir(parents=True, exist_ok=True)
_recent_requests = deque(maxlen=METRICS_HISTORY)

class RequestTimer:
    def __init__(self, name: str):
        self.name = name
        self.start = 0.0
        self.ms = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ms = int((time.time() - self.start) * 1000)

from dataclasses import dataclass, field
import datetime

@dataclass
class RequestLog:
    request_id: str
    query_raw: str
    ts: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")
    query_normalized: str = ""
    cache_hit: bool = False
    timings_ms: Dict[str, float] = field(default_factory=dict)
    chunk_counts: Dict[str, int] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    memory_mb: float = 0.0
    error: Optional[str] = None

def log_request(log: RequestLog) -> None:
    log_dict = log.__dict__
    _recent_requests.append(log_dict)
    if LOG_ENABLED:
        log_file = LOGS_DIR / "runbook_rag.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_dict) + "\n")

def get_metrics(n: int = METRICS_HISTORY) -> dict:
    recent = list(_recent_requests)[-n:]
    total = len(recent)
    if total == 0:
        return {"recent_requests": [], "summary": {}}
        
    avg_total_ms = sum(r.get("timings_ms", {}).get("total", 0) for r in recent) / total
    avg_llm_ms = sum(r.get("timings_ms", {}).get("llm", 0) for r in recent) / total
    cache_hits = sum(1 for r in recent if r.get("cache_hit"))
    cache_hit_rate = cache_hits / total if total > 0 else 0
    avg_chunks = sum(r.get("chunk_counts", {}).get("sent_to_llm", 0) for r in recent) / total
    
    return {
        "recent_requests": recent,
        "summary": {
            "total_requests": total,
            "avg_total_ms": avg_total_ms,
            "avg_llm_ms": avg_llm_ms,
            "cache_hit_rate": cache_hit_rate,
            "avg_chunks_sent_to_llm": avg_chunks
        }
    }

def get_memory_mb() -> float:
    if HAS_PSUTIL:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    return 0.0
