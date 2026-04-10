"""
Re-index Qdrant with a LIMITED number of chunks (for low-RAM machines).

Your full 358k embeddings stay on disk untouched.
This only controls how many get loaded into the Qdrant search index.

Usage:
    python reindex_light.py              (default: 100,000 chunks)
    python reindex_light.py --limit 50000
"""

import sys
import json
import numpy as np
from pathlib import Path

from src.utils.logger import setup_logger, get_logger
from src.vectorstore.qdrant_store import QdrantStore
from src.embeddings.embedding_cache import EmbeddingCache
from config.settings import CHUNKS_DIR

setup_logger()
logger = get_logger(__name__)


def main():
    # Parse limit from args
    limit = 100_000
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        limit = int(sys.argv[idx + 1])

    print(f"{'=' * 60}")
    print(f"  LIGHTWEIGHT RE-INDEXER (RAM-safe)")
    print(f"  Chunk limit: {limit:,}")
    print(f"{'=' * 60}\n")

    # Step 1: Load chunk files from disk (only up to limit)
    print("Step 1: Loading chunk files...")
    chunk_ids = []
    texts = []
    metadata = []

    chunk_files = sorted(CHUNKS_DIR.glob("*_semantic.json"))
    print(f"   Found {len(chunk_files)} chunk files on disk")

    for cf in chunk_files:
        if len(chunk_ids) >= limit:
            break

        with open(cf, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # Handle both formats
        if isinstance(raw, dict) and "chunks" in raw:
            chunk_list = raw["chunks"]
        elif isinstance(raw, list):
            chunk_list = raw
        else:
            continue

        for chunk in chunk_list:
            if len(chunk_ids) >= limit:
                break
            chunk_ids.append(chunk['chunk_id'])
            texts.append(chunk['text'])
            metadata.append({k: v for k, v in chunk.items() if k != 'text'})

    print(f"   Loaded {len(chunk_ids):,} chunks (limit: {limit:,})\n")

    # Step 2: Load embeddings and match to chunks
    print("Step 2: Loading embedding cache...")
    cache = EmbeddingCache()
    cache.load()
    embeddings_matrix, cached_ids = cache.get_all()
    print(f"   Cache has {len(cached_ids):,} embeddings")

    # Build lookup
    id_to_row = {cid: i for i, cid in enumerate(cached_ids)}

    # Match chunks to embeddings
    valid = [(i, id_to_row[cid]) for i, cid in enumerate(chunk_ids) if cid in id_to_row]
    print(f"   Matched {len(valid):,} chunks with embeddings\n")

    chunk_indices = [v[0] for v in valid]
    embed_indices = [v[1] for v in valid]

    final_chunk_ids = [chunk_ids[i] for i in chunk_indices]
    final_texts = [texts[i] for i in chunk_indices]
    final_metadata = [metadata[i] for i in chunk_indices]
    final_embeddings = embeddings_matrix[embed_indices]

    # Step 3: Recreate Qdrant collection
    print("Step 3: Rebuilding Qdrant collection...")
    store = QdrantStore()
    store.create_collection(recreate=True)

    # Step 4: Index
    print(f"Step 4: Indexing {len(final_chunk_ids):,} chunks into Qdrant...")
    total = store.index_chunks(
        embeddings=final_embeddings,
        chunk_ids=final_chunk_ids,
        metadata=final_metadata,
        texts=final_texts,
    )

    print(f"\n{'=' * 60}")
    print(f"  ✅ INDEXING COMPLETE")
    print(f"  Chunks indexed: {total:,}")
    print(f"  Collection:     {store.get_collection_info()}")
    print(f"  RAM usage:      ~{total * 768 * 4 / 1e6:.0f} MB (vectors only)")
    print(f"{'=' * 60}")
    print(f"\n  👉 Now run: python run_api.py")


if __name__ == "__main__":
    main()
