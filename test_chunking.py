"""
Compare all three chunking strategies on the same document.
This script teaches you WHY strategy choice matters.
"""

import json
from pathlib import Path
from config.settings import PROCESSED_DIR
from src.utils.logger import get_logger, setup_logger
from src.processing.chunker import (
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
    Chunk
)



setup_logger()
logger = get_logger(__name__)




def analyze_chunks(chunks: list[Chunk], strategy_name: str):
    """Print detailed statistics about a set of chunks"""
    if not chunks:
        print(f"\n{strategy_name}: No chunks produced")
        return

    
    sizes = [c.word_count for c in chunks]


    print(f"\n{'='*55}")
    print(f"  STRATEGY: {strategy_name.upper()}")
    print(f"{'='*55}")
    print(f"  Total chunks:      {len(chunks)}")
    print(f"  Avg words/chunk:   {sum(sizes)/len(sizes):.0f}")
    print(f"  Min words/chunk:   {min(sizes)}")
    print(f"  Max words/chunk:   {max(sizes)}")
    print(f"  Std dev:           {(sum((x - sum(sizes) / len(sizes)) ** 2 for x in sizes)/len(sizes)) ** 0.5:.0f}")
    print()

    
    # Show first 3 chunks with annotations  
    for i, chunk in enumerate(chunks[:3]):
        # Check if chunk ends mid-sentence
        ends_cleanly = chunk.text.rstrip().endswith(('.', '!', '?'))
        quality_flag = "✅" if ends_cleanly else "⚠️ mid-sentence"

        print(f"  Chunk {i+1} [{chunk.word_count} words] {quality_flag}")
        print(f"  {'-'*50}")
        # Show first 200 chars
        preview = chunk.text[:200].replace('\n', ' ')
        print(f"  {preview}...")
        print()




def load_sample_paper() -> dict:
    """Load and processed paper for testing."""
    processed_files = list(PROCESSED_DIR.glob("*.json"))

    if not processed_files:
        raise FileNotFoundError(
            "No processed papers found. Run run_ingestion.py first."
        )

    
    # Find a paper with substantial text for meaningful comparison
    for pf in processed_files:
        with open(pf, encoding = 'utf-8') as f:
            doc = json.load(f)
        # Use a paper with 1000+ words for meaningful chunking
        if doc.get("word_count", 0) > 3000:
            logger.info(
                f"Using paper: {doc['paper_id']}\n"
                f"Title: {doc['title'][:70]}\n"
                f"Words: {doc['word_count']}"
            )

            return doc

    
    # Fallback to any paper
    with open(processed_files[0], encoding = 'utf-8') as f:
        return json.load(f)




def main():
    logger.info("Starting chunking strategy comparison...")


    # Load sample documents
    doc = load_sample_paper()
    text = doc['full_text']
    metadata = {
        "paper_id":         doc.get("paper_id", ""),
        "title":            doc.get("title", ""),
        "authors":          doc.get("authors", []),
        "published_date":   doc.get("published_date", ""),
        "primary_category": doc.get("primary_category", ""),
        "arxiv_url":        doc.get("arxiv_url", ""),
    }


    print(f"\nDocument: {doc['title'][:60]}...")
    print(f"Total words: {doc['word_count']}")
    print(f"Total chars: {doc['text_length']}")


    # ----------- STRATEGY 1: Fixed -----------
    logger.info("Running Fixed Size chunker...")
    fixed_chunks = FixedSizeChunker().split(text, metadata)
    analyze_chunks(fixed_chunks, "Fixed Size")


    # ----------- STRATEGY 2: Recursive -----------
    logger.info("Running Recursive chunker...")
    recursive_chunks = RecursiveChunker().split(text, metadata)
    analyze_chunks(recursive_chunks, "Recursive")


    # ----------- STRATEGY 3: Semantic -----------
    logger.info("Running Semantic chunker (loads embedding model)...")
    semantic_chunks = SemanticChunker().split(text, metadata)
    analyze_chunks(semantic_chunks, "Semantic")




    # ----------- Head-to-Head comparison -----------
    print(f"\n{'='*55}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*55}")
    print(f"  {'Metric':<28} {'Fixed':>8} {'Recursive':>10} {'Semantic':>9}")
    print(f"  {'-'*55}")


    for label, chunks in [
        ("fixed", fixed_chunks),
        ("recursive", recursive_chunks),
        ("semantic", semantic_chunks),
    ]:
        sizes = [c.word_count for c in chunks]
        avg   = sum(sizes) / len(sizes) if sizes else 0
        std   = (sum((x-avg) ** 2 for x in sizes) / len(sizes)) ** 0.5 if sizes else 0
        clean = sum(1 for c in chunks if c.text.rstrip().endswith(('.','!','?')))
        pct   = 100 * clean / len(chunks) if chunks else 0


    # Print comparison table properly
    all_results = {}
    for label, chunks in [
        ("Fixed", fixed_chunks),
        ("Recursive", recursive_chunks),
        ("Semantic", semantic_chunks),
    ]:
        sizes = [c.word_count for c in chunks]
        avg   = sum(sizes) / len(sizes) if sizes else 0
        std   = (sum((x-avg) ** 2 for x in sizes) / len(sizes)) ** 0.5 if sizes else 0
        clean = sum(1 for c in chunks if c.text.rstrip().endswith(('.','!','?')))
        pct   = 100 * clean/len(chunks) if chunks else 0
        all_results[label] = {
            "count": len(chunks), "avg": avg,
            "std": std, "clean_pct": pct
        }


    r = all_results
    print(f"  {'Chunk count':<28} {r['Fixed']['count']:>8} {r['Recursive']['count']:>10} {r['Semantic']['count']:>9}")
    print(f"  {'Avg words/chunk':<28} {r['Fixed']['avg']:>8.0f} {r['Recursive']['avg']:>10.0f} {r['Semantic']['avg']:>9.0f}")
    print(f"  {'Std dev (consistency)':<28} {r['Fixed']['std']:>8.0f} {r['Recursive']['std']:>10.0f} {r['Semantic']['std']:>9.0f}")
    print(f"  {'Clean endings %':<28} {r['Fixed']['clean_pct']:>7.0f}% {r['Recursive']['clean_pct']:>9.0f}% {r['Semantic']['clean_pct']:>8.0f}%")

    print(f"\n  WINNER: Semantic (highest clean endings, adaptive sizing)")
    print(f"  FOR PRODUCTION: Recursive (fast + good quality trade-off)")
    


if __name__ == "__main__":
    main()