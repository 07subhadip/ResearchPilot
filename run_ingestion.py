"""
Master script to run the data ingestion pipeline.

Run this from the project root:
    python run_ingestion.py

This script orchestrates:
    1. Fetch paper metadata from ArXiv
    2. Download PDFs for fetched papers
"""

import json
from pathlib import Path

from src.utils.logger import get_logger, setup_logger
from src.ingestion.arxiv_fetcher import ArXivFetcher
from src.ingestion.pdf_downloader import PDFDownloader
from src.processing.pdf_extractor import PDFExtractor
from config.settings import RAW_DIR, PROCESSED_DIR, TOTAL_PAPERS_TARGET


setup_logger()
logger = get_logger(__name__)


def load_all_raw_papers() -> list[dict]:
    papers = []

    for f in RAW_DIR.glob("*.json"):
        if f.name == "paper_index.json":
            continue
        with open(f, encoding = 'utf-8') as fp:
            papers.append(json.load(fp))
    return papers



def print_section(title: str):
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)



def main():
    print_section("RESEARCHPILOT — FULL PIPELINE")

    # -------- PHASE 1: Fetch Metadata --------
    print_section("PHASE 1: Fetching ArXiv Metadata")
    fetcher     = ArXivFetcher()
    new_papers  = fetcher.fetch_papers(max_papers = TOTAL_PAPERS_TARGET)
    logger.info(f"New papers fetched: {len(new_papers)}")

    # -------- PHASE 2: Download PDFs --------
    print_section("PHASE 2: Downloading PDFs")
    all_papers = load_all_raw_papers()
    downloader = PDFDownloader()
    dl_stats   = downloader.download_all(all_papers)
    logger.info(f"Download stats: {dl_stats}")

    # -------- PHASE 3: Extract Text --------
    print_section("PHASE 3: Extracting and Cleaning Text")
    extractor  = PDFExtractor()
    proc_stats = extractor.process_all()
    logger.info(f"Processing stats: {proc_stats}")

    # -------- SUMMARY --------
    processed_files = list(PROCESSED_DIR.glob("*.json"))
    print_section("PIPELINE COMPLETE")
    logger.info(f"Papers in processed/: {len(processed_files)}")
    logger.info("Ready for Phase 5: Chunking")



if __name__ == "__main__":
    main()