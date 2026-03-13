# test_fetch.py
"""
Smart test script that handles existing data correctly.
Tests three things:
  1. Can we load existing papers from disk?
  2. Can we fetch NEW papers (beyond what we have)?
  3. Is our data schema correct?
"""

import json
from pathlib import Path
from src.utils.logger import setup_logger, get_logger
from src.ingestion.arxiv_fetcher import ArXivFetcher
from config.settings import RAW_DIR

setup_logger()
logger = get_logger(__name__)

def test_existing_data():
    """Check what we already have on disk."""
    paper_files = [
        f for f in RAW_DIR.glob("*.json")
        if f.name != "paper_index.json"
    ]
    
    logger.info(f"Papers already on disk: {len(paper_files)}")
    
    if not paper_files:
        logger.warning("No papers found on disk. Run fetch first.")
        return []
    
    papers = []
    for pf in paper_files[:3]:  # Show first 3
        with open(pf) as f:
            data = json.load(f)
        papers.append(data)
        logger.info(f"  -> {data['paper_id']}: {data['title'][:60]}...")
        logger.info(f"     Category: {data['primary_categories']} | Date: {data['published_date']}")
    
    return papers

def test_schema_validation():
    """Verify our Pydantic schema works correctly."""
    from src.ingestion.arxiv_fetcher import PaperMetadata
    
    logger.info("Testing schema validation...")
    
    # Test with valid data
    try:
        paper = PaperMetadata(
            paper_id         = "http://arxiv.org/abs/2301.07041v2",  # Raw ID with version
            title            = "  Test Paper  With  Extra   Spaces  ",
            abstract         = "This is a test abstract.",
            authors          = ["Author One", "Author Two"],
            categories       = ["cs.LG", "cs.AI"],
            primary_categories = "cs.LG",
            published_date   = "2023-01-17",
            updated_date     = "2023-03-15",
            arxiv_url        = "https://arxiv.org/abs/2301.07041",
            pdf_url          = "https://arxiv.org/pdf/2301.07041",
        )
        
        # Verify our validators ran
        assert paper.paper_id == "2301.07041", f"ID cleanup failed: {paper.paper_id}"
        assert paper.title == "Test Paper With Extra Spaces", f"Whitespace cleanup failed: {paper.title}"
        
        logger.info("  -> Schema validation: PASSED")
        logger.info(f"     paper_id cleaned: '2301.07041'")
        logger.info(f"     title cleaned: '{paper.title}'")
        return True
        
    except Exception as e:
        logger.error(f"  -> Schema validation FAILED: {e}")
        return False

def test_fresh_fetch(n: int = 3):
    """
    Fetch papers, but temporarily ignore existing index
    to force fresh results for testing.
    """
    logger.info(f"Fetching {n} fresh papers from ArXiv...")
    
    fetcher = ArXivFetcher()
    
    # TEMPORARY: clear existing IDs in memory only (not on disk)
    # This lets us test the fetch logic without deleting real data
    original_ids = fetcher.existing_ids.copy()
    fetcher.existing_ids = set()  # Pretend we have nothing
    
    papers = fetcher.fetch_papers(max_papers=n)
    
    # Restore original IDs
    fetcher.existing_ids = original_ids
    
    if papers:
        logger.info(f"  -> Fresh fetch: PASSED. Got {len(papers)} papers")
        for p in papers:
            logger.info(f"     {p.paper_id}: {p.title[:55]}...")
    else:
        logger.warning("  -> Fresh fetch returned 0 papers. Check network connection.")
    
    return papers

def main():
    logger.info("=" * 55)
    logger.info("RESEARCHPILOT — INGESTION TEST SUITE")
    logger.info("=" * 55)
    
    # Test 1: Existing data
    logger.info("\n[TEST 1] Checking existing data on disk...")
    existing = test_existing_data()
    
    # Test 2: Schema validation
    logger.info("\n[TEST 2] Schema validation...")
    test_schema_validation()
    
    # Test 3: Fresh fetch
    logger.info("\n[TEST 3] Fresh fetch from ArXiv...")
    fresh = test_fresh_fetch(n=3)
    
    logger.info("\n" + "=" * 55)
    logger.info("TEST SUITE COMPLETE")
    logger.info(f"Existing papers: {len(existing)} shown (may have more)")
    logger.info(f"Fresh papers fetched: {len(fresh)}")
    logger.info("=" * 55)

if __name__ == "__main__":
    main()