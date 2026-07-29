import threading
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from config.settings import EMBED_MODEL_PATH, EMBED_BATCH_SIZE, EMBED_MAX_LENGTH

class Embedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH, local_files_only=True)
        self.model = AutoModel.from_pretrained(EMBED_MODEL_PATH, local_files_only=True)
        self.model.eval()

    def encode(self, texts: List[str], batch_size=EMBED_BATCH_SIZE, max_length=EMBED_MAX_LENGTH, normalize_embeddings=True, convert_to_numpy=True) -> np.ndarray:
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                if normalize_embeddings:
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                all_embeddings.append(batch_embeddings.cpu().numpy())
        if convert_to_numpy:
            return np.vstack(all_embeddings)
        return all_embeddings

_embedder_instance = None
_embedder_lock = threading.Lock()

def get_embedder() -> Embedder:
    global _embedder_instance
    if _embedder_instance is None:
        with _embedder_lock:
            if _embedder_instance is None:
                _embedder_instance = Embedder()
    return _embedder_instance
