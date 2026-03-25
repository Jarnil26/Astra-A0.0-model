"""
build_astra.py — Render build-time bootstrap for Astra A0.0 Backend

The model files (ayurveda_ai.db, ayurveda.index) are pre-built and committed to Git.
This script simply verifies they are present and regenerates only what's missing.
"""
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))


def check(label: str, path: str) -> bool:
    full = os.path.join(BASE, path)
    exists = os.path.exists(full)
    size = os.path.getsize(full) if exists else 0
    status = f"OK  ({size/1024:.1f} KB)" if exists else "MISSING"
    print(f"  {'OK' if exists else '!!'}  {label:<30} {status}")
    return exists


def main():
    print("=" * 52)
    print("    Astra A0.0 — Build Verification")
    print("=" * 52)
    t0 = time.time()

    required = {
        "SQLite DB (ayurveda_ai.db)":  "data/ayurveda_ai.db",
        "FAISS Index (ayurveda.index)": "data/ayurveda.index",
        "Prevalence JSON":             "data/disease_prevalence.json",
    }

    missing = []
    for label, path in required.items():
        if not check(label, path):
            missing.append(path)

    print()
    if missing:
        print(f"[FATAL] {len(missing)} required model file(s) are missing:")
        for m in missing:
            print(f"  - {m}")
        print()
        print("These files must be committed to GitHub.")
        print("They are pre-built and should not be in .gitignore.")
        sys.exit(1)

    print(f"All required files present. Build verified in {time.time()-t0:.2f}s")
    print("=" * 52)


if __name__ == "__main__":
    main()
