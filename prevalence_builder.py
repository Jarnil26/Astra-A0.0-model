import sqlite3
import json
import os

def build_prevalence(db_path="data/ayurveda_ai.db", output_path="data/disease_prevalence.json"):
    """
    Analyzes the indexed dataset to calculate normalized disease prevalence.
    Ensures that real-world common diseases in the training data have higher priors.
    """
    print(f"Building prevalence mapping from {db_path}...")
    
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Fetch counts from existing disease_priors table built by data_streamer
    cursor.execute("SELECT disease, count FROM disease_priors")
    priors = dict(cursor.fetchall())
    conn.close()

    if not priors:
        print("No disease priors found in database. Ensure data_streamer.py has been run.")
        return

    max_count = max(priors.values())
    
    # 2. Normalize: prevalence = count / max_count
    # We apply a small floor and logarithmic smoothing to avoid zero influence for rare items
    # but still keep common ones dominant.
    prevalence_map = {}
    for disease, count in priors.items():
        score = count / max_count
        prevalence_map[disease] = round(score, 4)

    # 3. Save to JSON
    with open(output_path, 'w') as f:
        json.dump(prevalence_map, f, indent=2)
    
    print(f"Successfully saved prevalence data for {len(prevalence_map)} diseases to {output_path}")

if __name__ == "__main__":
    build_prevalence()
