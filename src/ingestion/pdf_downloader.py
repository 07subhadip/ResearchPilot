"""
Downloads PDF files for papers that have been fetched from ArXiv.

SEPARATION OF CONCERNS:
  arxiv_fetcher.py  → Gets metadata (fast, no large files)
  pdf_downloader.py → Downloads PDFs (slow, large files)

This separation means if PDF download fails, metadata is safe.
We can retry ONLY the failed PDFs without re-fetching metadata.
"""

import time
import json
import requests
from pathlib import Path

from tqdm import tqdm # Progress bar

from src.utils.logger import get_logger
from config.settings import (
    RAW_DIR,
    PDF_DOWNLOAD_TIMEOUT,
    MAX_DOWNLOAD_RETRIES,
    ARXIV_API_DELAY_SECONDS
)

logger = get_logger(__name__)



class PDFDownloader:
    """
    Download PDFs from ArXiv with retry logic and progress tracking
    """

    def __init__(self):
        # Configure requests session
        # WHY SESSION: Reuses TCP Connection across requests
        # (faster than creating new connection per download)
        self.session = requests.Session()
        self.session.headers.update(
            {
                # Identify themselves to ArXiv - polite and avoids blocks
                "User-Agent": "ResearchPilot/1.0 (educational research project)"
            }
        ) 


        # Directory for downloaded PDFs
        self.pdf_dir = RAW_DIR / "pdfs"
        self.pdf_dir.mkdir(exist_ok = True)

    
    def download_pdf(self, paper_id: str, pdf_url: str) -> bool:
        """
        Download a single PDF with retry logic.

        Args:
            paper_id: ArXiv paper ID (used for filename)
            pdf_url:  Direct URL to the PDF

        Returns:
            True if downloaded successfully, False otherwise

        RETRY PATTERN (Exponential Backoff):
            Attempt 1: fail → wait 2 seconds
            Attempt 2: fail → wait 4 seconds
            Attempt 3: fail → wait 8 seconds
            → give up, log error, continue to next paper

        WHY EXPONENTIAL BACKOFF:
        If a server is overloaded, hammering it with immediate retries
        makes things worse. Waiting longer between retries gives the
        server time to recover. This is standard practice for all
        production systems that call external services.
        """
        output_path = self.pdf_dir / f"{paper_id}.pdf"


        # Skip if already downloaded (idempotent)
        if output_path.exists() and output_path.stat().st_size > 1000:
            logger.debug(f"PDF already exists: {paper_id}")
            return True

        for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
            try:
                logger.debug(f"Downloading {paper_id} (attempt {attempt})")

                # stream = True means we download in chunks, not all at once
                # This prevents running out of memory on large PDFs
                response = self.session.get(
                    pdf_url,
                    timeout = PDF_DOWNLOAD_TIMEOUT,
                    stream = True
                )

                # Raise exception for 4xx or 5xx status codes
                response.raise_for_status()

                # Write PDF to disk in chunks of 8KB
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size = 8192):
                        if chunk:   # Filter out keep-alive empty chunks
                            f.write(chunk)

                # Verify file is not empty or suspiciously small
                file_size = output_path.stat().st_size
                if file_size < 1000:
                    logger.warning(f"Suspiciously small PDF: {paper_id} ({file_size} bytes)")
                    output_path.unlink()    # Delete Bad File
                    return False

                logger.debug(f"Downloaded {paper_id}: {file_size / 1024:.1f} KB")
                return True


            except requests.exceptions.RequestException as e:
                logger.warning(f"Download attempt {attempt} failed for {paper_id}: {e}")

                if attempt < MAX_DOWNLOAD_RETRIES:
                    # Exponential backoff: 2^attempt seconds
                    wait_time = 2 ** attempt
                    logger.debug(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {MAX_DOWNLOAD_RETRIES} attemps failed for {paper_id}")
                    return False

            
        return False



    def download_all(self, papers: list[str]) -> dict:
        """
        Download PDFs for a list of papers with progress tracking.

        Args:
            papers: List of paper metadata dicts (loaded from JSON files)

        Returns:
            Summary statistics dict
        """  

        successful = 0
        failed     = 0
        skipped    = 0


        # tqdm wraps our list to show a progress bar
        # desc= sets the label on the progress bar
        for paper in tqdm(papers, desc = "Downloading PDFs"):
            paper_id = paper['paper_id']
            pdf_url  = paper['pdf_url']


            # Skip already downloaded papers
            if paper.get("pdf_downloaded"):
                skipped += 1
                continue

            # Download with delay to respect rate limits
            success = self.download_pdf(paper_id, pdf_url)


            if success:
                successful += 1
                # Update the paper's JSON file to mark pdf_downloaded = True
                self._mark_downloaded(paper_id)
                time.sleep(ARXIV_API_DELAY_SECONDS)
            else:
                failed += 1

            
        
        summary = {
            "successful": successful,
            "failed":     failed,
            "skipped":    skipped,
            "total":      len(papers)
        }

        logger.info(f"PDF download complete: {summary}")
        return summary

    
    def _mark_downloaded(self, paper_id: str):
        """
        Update the paper's JSON metadata to mark pdf_downloaded = True.
        This updates our pipeline state flag.
        """
        json_path = RAW_DIR / f"{paper_id}.json"

        if not json_path.exists():
            return

        with open(json_path, 'r', encoding = 'utf-8') as f:
            data = json.load(f)


        data["pdf_downloaded"] = True

        with open(json_path, "w") as f:
            json.dump(data, f, indent = 2)