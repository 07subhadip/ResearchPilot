"""Check what's actually stored in Qdrant payload."""

from src.utils.logger import setup_logger, get_logger
from src.vectorstore.qdrant_store import QdrantStore

setup_logger()
logger = get_logger(__name__)

def main():
    store = QdrantStore()

    # Fetch 3 points directly by scrolling the collection
    # scroll() returns points without needing a query vector
    results, _ = store.client.scroll(
        collection_name = store.collection_name,
        limit           = 3,
        with_payload    = True,
        with_vectors    = False,
    )

    for i, point in enumerate(results):
        print(f"\n{'='*55}")
        print(f"Point {i+1} — ID: {point.id}")
        print(f"Payload keys: {list(point.payload.keys())}")
        print()
        for k, v in point.payload.items():
            # Truncate long values for readability
            val_str = str(v)[:80] if v else "EMPTY/NONE"
            print(f"  {k:<22}: {val_str}")

if __name__ == "__main__":
    main()