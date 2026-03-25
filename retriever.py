import os
import faiss
import numpy as np
import sqlite3
import json
from sentence_transformers import SentenceTransformer
import torch

# Resolve the model path relative to this file so it works both locally and on Render
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL_MODEL = os.path.join(_HERE, "data", "model_all_minilm_l6_v2")
_MODEL_NAME = "all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, index_path="data/ayurveda.index", db_path="data/ayurveda_ai.db", model_name=_MODEL_NAME):
        self.index = faiss.read_index(index_path)
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = 20
        # Load from local path if available, otherwise download from HuggingFace
        local_model = _LOCAL_MODEL if os.path.isdir(_LOCAL_MODEL) else model_name
        self.model = SentenceTransformer(local_model)
        self.db_path = db_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        
        # Persistent Connection
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.weights = self._load_weights()
        
        # Embedding Cache
        self.embedding_cache = {}

    def _load_weights(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT symptom, count FROM symptom_freqs")
        freqs = dict(cursor.fetchall())
        return {s: 1.0 / np.log1p(count) for s, count in freqs.items()}

    def get_query_embedding(self, symptoms):
        """Compute weighted mean of symptom embeddings with caching."""
        symptoms = [s.strip().lower() for s in symptoms]
        
        vecs = []
        ws = []
        
        # Check cache for individual symptoms
        for s in symptoms:
            if s in self.embedding_cache:
                emb = self.embedding_cache[s]
            else:
                emb = self.model.encode([s], convert_to_numpy=True)[0]
                self.embedding_cache[s] = emb
            
            w = self.weights.get(s, 1.0)
            vecs.append(emb)
            ws.append(w)
            
        if not vecs:
            return np.zeros(self.model.get_sentence_embedding_dimension())
            
        vecs = np.array(vecs)
        ws = np.array(ws).reshape(-1, 1)
        weighted_vec = np.sum(vecs * ws, axis=0) / np.sum(ws)
        
        # Normalize
        norm = np.linalg.norm(weighted_vec)
        if norm > 0:
            weighted_vec /= norm
        return weighted_vec.astype('float32')

    def retrieve(self, symptoms, k=50):
        query_vec = self.get_query_embedding(symptoms).reshape(1, -1)
        similarities, indices = self.index.search(query_vec, k)
        
        # indices to record IDs (1-based in SQLite)
        results = []
        cursor = self.conn.cursor()
        
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1 or sim < 0.3: # Post-processing: Remove low similarity
                continue
            
            # Fetch from SQLite - persistent connection used here
            cursor.execute("SELECT data FROM records WHERE id = ?", (int(idx) + 1,))
            row = cursor.fetchone()
            if row:
                record = json.loads(row[0])
                results.append({
                    "record": record,
                    "similarity": float(sim)
                })
        
        return results

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    # Example usage (standalone)
    retr = Retriever()
    print(retr.retrieve(["fever", "headache"]))
