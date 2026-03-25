import ijson
import sqlite3
import json
import os
from tqdm import tqdm
from collections import Counter
from decimal import Decimal

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

class DataStreamer:
    def __init__(self, dataset_path, db_path="data/ayurveda_ai.db"):
        self.dataset_path = dataset_path
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()
        # Metadata storage
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY,
                data TEXT
            )
        """)
        # Specificity and Priors
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS symptom_freqs (
                symptom TEXT PRIMARY KEY,
                count INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS disease_priors (
                disease TEXT PRIMARY KEY,
                count INTEGER
            )
        """)
        self.conn.commit()

    def normalize_symptom(self, symptom):
        if not symptom:
            return ""
        return str(symptom).strip().lower()

    def stream_and_index(self, batch_size=10000):
        """Pass 1: Populate DB and calculate frequencies."""
        print(f"Starting Pass 1: Indexing to {self.db_path} and calculating frequencies...")
        
        symptom_counter = Counter()
        disease_counter = Counter()
        
        cursor = self.conn.cursor()
        
        with open(self.dataset_path, 'rb') as f:
            objects = ijson.items(f, 'item')
            
            batch_records = []
            count = 0
            
            for record in tqdm(objects, desc="Streaming Records"):
                # 1. Normalize and count symptoms
                symptoms = [self.normalize_symptom(s) for s in record.get("input", {}).get("symptoms", [])]
                symptom_counter.update(symptoms)
                
                # 2. Count diseases for priors
                # We check probabilities and take the top disease or just all mentioned
                probs = record.get("prediction", {}).get("disease_probabilities", {})
                disease_counter.update(probs.keys())
                
                # 3. Store in batch
                batch_records.append((json.dumps(record, cls=DecimalEncoder),))
                count += 1
                
                if len(batch_records) >= batch_size:
                    cursor.executemany("INSERT INTO records (data) VALUES (?)", batch_records)
                    batch_records = []
                    # Commit every 100k for safety
                    if count % 100000 == 0:
                        self.conn.commit()

            if batch_records:
                cursor.executemany("INSERT INTO records (data) VALUES (?)", batch_records)
            
            self.conn.commit()

        # Save frequencies
        print("Saving symptom frequencies...")
        cursor.executemany(
            "INSERT OR REPLACE INTO symptom_freqs (symptom, count) VALUES (?, ?)",
            list(symptom_counter.items())
        )
        
        print("Saving disease priors...")
        cursor.executemany(
            "INSERT OR REPLACE INTO disease_priors (disease, count) VALUES (?, ?)",
            list(disease_counter.items())
        )
        
        self.conn.commit()
        print(f"Pass 1 complete. Indexed {count} records.")

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    dataset = "data/AyurGenixAI_Dataset.json"
    if not os.path.exists(dataset):
        dataset = "Dataset/AyurGenixAI_Dataset.json"
        
    streamer = DataStreamer(dataset)
    try:
        streamer.stream_and_index()
    finally:
        streamer.close()
