import os
import sys
import subprocess
import time

def run_step(name, command):
    print(f"\n>>> [ASTRA BUILD] {name}...")
    start_time = time.time()
    try:
        # Use sys.executable to ensure we use the same python environment
        full_command = f"{sys.executable} {command}"
        subprocess.run(full_command, shell=True, check=True)
        elapsed = time.time() - start_time
        print(f"--- [DONE] {name} in {elapsed:.2f}s ---")
    except subprocess.CalledProcessError as e:
        print(f"!!! [ERROR] {name} failed with exit code {e.returncode} !!!")
        sys.exit(1)

def main():
    print("==================================================")
    print("      Astra A0: Automated Data Bootstrapper       ")
    print("==================================================")

    # Ensure data dir exists
    if not os.path.exists("data"):
        os.makedirs("data")

    # 1. Data Streaming & Indexing (SQL)
    if not os.path.exists("data/ayurveda_ai.db"):
        run_step("1. SQL Indexing", "data_streamer.py")
    else:
        print("[SKIP] SQLite database exists.")

    # 2. Embedding Generation
    if not os.path.exists("data/embeddings"):
        run_step("2. Embedding Vectors", "embedding_builder.py")
    else:
        print("[SKIP] Embeddings exist.")

    # 3. FAISS Index Building
    if not os.path.exists("data/ayurveda.index"):
        run_step("3. FAISS Core Build", "faiss_index_builder.py")
    else:
        print("[SKIP] FAISS index exists.")

    # 4. Prevalence Matrix Building
    if not os.path.exists("data/disease_prevalence.json"):
        run_step("4. Clinical Prevalence", "prevalence_builder.py")
    else:
        print("[SKIP] Prevalence matrix exists.")

    print("\n==================================================")
    print("  BOOTSTRAP COMPLETE! ENGINE IS NOW READY.        ")
    print("==================================================")

if __name__ == "__main__":
    main()
