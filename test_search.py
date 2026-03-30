"""
Test Qdrant search with real queries.
This is your first end-to-end retrieval test.
"""


from src.utils.logger import get_logger, setup_logger
from src.vectorstore.qdrant_store import QdrantStore
from src.embeddings.embedding_model import EmbeddingModel


setup_logger()
logger = get_logger(__name__)



def search_and_display(store: QdrantStore, model: EmbeddingModel, query: str, top_k: int = 3):
    """Run a search query and display results clearly."""
    print(f"\n{'=' * 60}")
    print(f"QUERY: {query}")
    print(f"{'=' * 60}")


    # Embed the query (with BGE prefix)
    query_vector = model.embed_query(query)

    # Search Qdrant
    results = store.search(query_vector, top_k = top_k)

    if not results:
        print(f"No results found.")
        return

    for i, r in enumerate(results):
        print(f"\n[{i+1}] Score: {r['score']:.4f}")
        print(f"     Paper: {r.get('paper_id', 'N/A')}")
        print(f"     Title: {r.get('title', 'N/A')[:65]}...")
        print(f"     Date:  {r.get('published_date', 'N/A')}")
        print(f"     Category: {r.get('primary_category', 'N/A')}")
        print(f"     Chunk {r.get('chunk_index','?')}/{r.get('total_chunks','?')}")
        print(f"     Text preview: {r.get('text','')[:150].replace(chr(10),' ')}...")




def main():
    logger.info("Loading model and connecting to Qdrant...")

    store = QdrantStore()
    model = EmbeddingModel()


    # Verify collection exists
    info = store.get_collection_info()
    logger.info(f"Collection info: {info}")


    if info.get("points_count", 0) == 0:
        logger.error("Collection is empty. Run run_indexing.py first.")
        return

    # --- Test queries covering different retrieval scenarios ---

    # Test 1: Conceptual Query
    search_and_display(store, model,
        "how does self-attention mechanism work in transformers",
        top_k=3
    )

    # Test 2: Task-specific query
    search_and_display(store, model,
        "reinforcement learning for multi-agent systems",
        top_k=3
    )

    # Test 3: Method comparison query
    search_and_display(store, model,
        "comparison of fine-tuning methods for large language models",
        top_k=3
    )


    # Test 4: with metadata filter - only cs.LG papers
    print(f"\n{'='*60}")
    print("FILTERED QUERY: 'neural network optimization' (cs.LG only)")
    print(f"{'='*60}")
    query_vector = model.embed_query("neural network optimization methods")
    results = store.search(
        query_vector,
        top_k = 3,
        filter_category = "cs.LG"
    )
    for i, r in enumerate(results):
        print(f"[{i+1}] {r['score']:.4f} | {r.get('primary_category')} | {r.get('title','')[:55]}...")

    logger.info("\n✅ Search test complete.")



if __name__ == "__main__":
    main()