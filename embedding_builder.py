import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sqlite3
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import json

class EmbeddingBuilder:
    def __init__(self, db_path="data/ayurveda_ai.db", model_name="all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def get_weights(self):
        """Fetch symptom weights from DB: 1 / log(1 + freq)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT symptom, count FROM symptom_freqs")
        freqs = dict(cursor.fetchall())
        conn.close()
        
        weights = {s: 1.0 / np.log1p(count) for s, count in freqs.items()}
        return weights

    def precalculate_symptom_embeddings(self, symptoms):
        """Embed unique symptoms to save computation."""
        print(f"Precalculating embeddings for {len(symptoms)} unique symptoms...")
        embeddings = self.model.encode(symptoms, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
        return dict(zip(symptoms, embeddings))

    def build_embeddings(self, output_dir="data/embeddings", chunk_size=100000):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        weights = self.get_weights()
        unique_symptoms = list(weights.keys())
        symptom_map = self.precalculate_symptom_embeddings(unique_symptoms)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, data FROM records")
        
        all_embeddings = []
        chunk_idx = 0
        
        # We'll use a generator to skip loading everything into memory
        rows = cursor.execute("SELECT id, data FROM records")
        
        for i, (row_id, data_json) in enumerate(tqdm(rows, desc="Building Record Embeddings")):
            record = json.loads(data_json)
            symptoms = [s.strip().lower() for s in record.get("input", {}).get("symptoms", [])]
            
            vecs = []
            ws = []
            for s in symptoms:
                if s in symptom_map:
                    vecs.append(symptom_map[s])
                    ws.append(weights.get(s, 1.0))
            
            if vecs:
                # Weighted average
                vecs = np.array(vecs)
                ws = np.array(ws).reshape(-1, 1)
                weighted_vec = np.sum(vecs * ws, axis=0) / np.sum(ws)
                # Normalize (important for cosine similarity)
                norm = np.linalg.norm(weighted_vec)
                if norm > 0:
                    weighted_vec /= norm
                all_embeddings.append(weighted_vec)
            else:
                # Fallback to zero vector if no symptoms found
                all_embeddings.append(np.zeros(self.model.get_sentence_embedding_dimension()))

            if (i + 1) % chunk_size == 0:
                self.save_chunk(all_embeddings, output_dir, chunk_idx)
                all_embeddings = []
                chunk_idx += 1

        if all_embeddings:
            self.save_chunk(all_embeddings, output_dir, chunk_idx)
            
        conn.close()
        print("Embedding build complete.")

    def save_chunk(self, embeddings, output_dir, idx):
        path = os.path.join(output_dir, f"chunk_{idx}.npy")
        np.save(path, np.array(embeddings, dtype=np.float32))
        print(f"Saved {path}")

if __name__ == "__main__":
    builder = EmbeddingBuilder()
    builder.build_embeddings()
