import json
import pickle
import faiss
import numpy as np
import re
from pathlib import Path
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from config.settings import INDEX_DIR, FAISS_INDEX_PATH, BM25_PATH, CHUNKS_PATH
from pipeline.chunker import chunk_file
from pipeline.embedder import get_embedder

def load_state() -> Dict:
    state = {'chunks': None, 'faiss_index': None, 'bm25': None}
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, 'r') as f:
            state['chunks'] = json.load(f)
    if FAISS_INDEX_PATH.exists():
        state['faiss_index'] = faiss.read_index(str(FAISS_INDEX_PATH))
    if BM25_PATH.exists():
        with open(BM25_PATH, 'rb') as f:
            state['bm25'] = pickle.load(f)
    return state

def save_state(chunks: List[dict], faiss_index, bm25) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, 'w') as f:
        json.dump(chunks, f)
    faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))
    with open(BM25_PATH, 'wb') as f:
        pickle.dump(bm25, f)

def build_full_index(runbooks_dir: Path) -> Tuple[List[dict], faiss.Index, BM25Okapi]:
    all_chunks = []
    chunk_id = 0
    for p in runbooks_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in ['.md', '.pdf', '.docx']:
            file_chunks = chunk_file(p, chunk_id)
            all_chunks.extend(file_chunks)
            chunk_id += len(file_chunks)
    
    chunks_dicts = [vars(c) for c in all_chunks]
    
    embedder = get_embedder()
    texts = [c['text'] for c in chunks_dicts]
    embeddings = embedder.encode(texts)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    tokenized_corpus = [re.findall(r'\w+', t.lower()) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    return chunks_dicts, index, bm25

def incremental_add(file_path: Path, existing_chunks: List[dict], existing_faiss_index, existing_bm25) -> Tuple[List[dict], faiss.Index, BM25Okapi]:
    start_id = len(existing_chunks)
    new_chunks = chunk_file(file_path, start_id)
    new_dicts = [vars(c) for c in new_chunks]
    
    if not new_dicts:
        return existing_chunks, existing_faiss_index, existing_bm25
        
    embedder = get_embedder()
    new_texts = [c['text'] for c in new_dicts]
    new_embeddings = embedder.encode(new_texts)
    
    existing_faiss_index.add(new_embeddings)
    all_chunks = existing_chunks + new_dicts
    
    tokenized_corpus = [re.findall(r'\w+', c['text'].lower()) for c in all_chunks]
    updated_bm25 = BM25Okapi(tokenized_corpus)
    
    return all_chunks, existing_faiss_index, updated_bm25
