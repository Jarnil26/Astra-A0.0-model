import faiss
import numpy as np
import os
import glob
from tqdm import tqdm

class FAISSIndexBuilder:
    def __init__(self, embedding_dir="data/embeddings", index_path="data/ayurveda.index", dim=384):
        self.embedding_dir = embedding_dir
        self.index_path = index_path
        self.dim = dim

    def build_index(self, nlist=4096, m=16, nbits=8):
        print(f"Building IVFPQ index: nlist={nlist}, m={m}, nbits={nbits}")
        
        # 1. Initialize Index
        quantizer = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
        
        # 2. Training (using a sample)
        chunk_files = glob.glob(os.path.join(self.embedding_dir, "chunk_*.npy"))
        if not chunk_files:
            raise ValueError("No embedding chunks found.")
            
        print("Loading sample for training...")
        # Train on up to 500k vectors (random sample)
        sample_chunk = np.load(chunk_files[0])
        train_size = min(len(sample_chunk), 500000)
        train_vecs = sample_chunk[:train_size].astype('float32')
        
        print("Training index...")
        index.train(train_vecs)
        
        # 3. Adding vectors in chunks
        print("Adding vectors to index...")
        for chunk_file in tqdm(chunk_files, desc="Processing chunks"):
            vecs = np.load(chunk_file).astype('float32')
            index.add(vecs)
            
        # 4. Save index
        print(f"Saving index to {self.index_path}...")
        faiss.write_index(index, self.index_path)
        print("Indexing complete.")

if __name__ == "__main__":
    builder = FAISSIndexBuilder()
    builder.build_index()
