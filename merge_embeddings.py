"""
Merge old (local) and new (Kaggle) embeddings into a single set.

This script:
  1. Loads existing embeddings.npy + chunk_ids.npy  (your ~51k chunks)
  2. Loads new_embeddings.npy + new_chunk_ids.npy  (from Kaggle batch)
  3. Deduplicates by chunk_id
  4. Saves the merged result back as embeddings.npy + chunk_ids.npy
  5. Backs up the originals first

Run from project root:
    python merge_embeddings.py
"""

import json
import shutil
import numpy as np
from pathlib import Path

EMBEDDINGS_DIR = Path("data/embeddings")

# File paths
old_emb_file  = EMBEDDINGS_DIR / "embeddings.npy"
old_ids_file  = EMBEDDINGS_DIR / "chunk_ids.npy"
new_emb_file  = EMBEDDINGS_DIR / "new_embeddings.npy"
new_ids_file  = EMBEDDINGS_DIR / "new_chunk_ids.npy"

# Backup dir
backup_dir = EMBEDDINGS_DIR / "backup_before_merge"
backup_dir.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("  RESEARCHPILOT — EMBEDDING MERGE TOOL")
    print("=" * 60)

    # ── Step 1: Validate files exist ──
    for f in [old_emb_file, old_ids_file, new_emb_file, new_ids_file]:
        if not f.exists():
            print(f"❌ Missing file: {f}")
            return
    print("✅ All required files found.\n")

    # ── Step 2: Load old embeddings ──
    print("Loading OLD embeddings...")
    old_embeddings = np.load(str(old_emb_file))
    old_ids = list(np.load(str(old_ids_file), allow_pickle=True))
    print(f"   Old: {old_embeddings.shape[0]:,} chunks, dim={old_embeddings.shape[1]}")

    # ── Step 3: Load new embeddings  ──
    print("Loading NEW embeddings (from Kaggle)...")
    new_embeddings = np.load(str(new_emb_file))
    new_ids = list(np.load(str(new_ids_file), allow_pickle=True))
    print(f"   New: {new_embeddings.shape[0]:,} chunks, dim={new_embeddings.shape[1]}")

    # ── Step 4: Deduplicate ──
    print("\nDeduplicating...")
    old_id_set = set(old_ids)
    keep_indices = []
    for i, cid in enumerate(new_ids):
        if cid not in old_id_set:
            keep_indices.append(i)

    unique_new_embeddings = new_embeddings[keep_indices]
    unique_new_ids = [new_ids[i] for i in keep_indices]
    duplicates_removed = len(new_ids) - len(unique_new_ids)
    print(f"   Duplicates skipped: {duplicates_removed}")
    print(f"   Unique new chunks:  {len(unique_new_ids):,}")

    # ── Step 5: Merge ──
    print("\nMerging...")
    merged_embeddings = np.vstack([old_embeddings, unique_new_embeddings])
    merged_ids = old_ids + unique_new_ids
    print(f"   MERGED TOTAL: {merged_embeddings.shape[0]:,} chunks")

    # ── Step 6: Backup originals ──
    print("\nBacking up originals...")
    shutil.copy2(old_emb_file, backup_dir / "embeddings_old.npy")
    shutil.copy2(old_ids_file, backup_dir / "chunk_ids_old.npy")
    print(f"   Backed up to: {backup_dir}")

    # ── Step 7: Save merged files ──
    print("\nSaving merged embeddings...")
    np.save(str(old_emb_file), merged_embeddings)
    np.save(str(old_ids_file), np.array(merged_ids, dtype=object))

    # Update the index file
    index = {
        "total_embeddings": len(merged_ids),
        "embedding_dimension": int(merged_embeddings.shape[1]),
        "model_name": "BAAI/bge-base-en-v1.5",
        "chunk_id_sample": merged_ids[:5],
    }
    with open(EMBEDDINGS_DIR / "embedding_index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"   ✅ embeddings.npy  → {merged_embeddings.shape}")
    print(f"   ✅ chunk_ids.npy   → {len(merged_ids):,} IDs")
    print(f"   ✅ embedding_index.json updated")

    # ── Summary ──
    size_mb = (EMBEDDINGS_DIR / "embeddings.npy").stat().st_size / 1e6
    print(f"\n{'=' * 60}")
    print(f"  MERGE COMPLETE!")
    print(f"  Old:    {len(old_ids):,} chunks")
    print(f"  + New:  {len(unique_new_ids):,} chunks")
    print(f"  = Total: {len(merged_ids):,} chunks")
    print(f"  File size: {size_mb:.0f} MB")
    print(f"{'=' * 60}")
    print(f"\n👉 Now run:  python run_indexing.py --recreate")


if __name__ == "__main__":
    main()
