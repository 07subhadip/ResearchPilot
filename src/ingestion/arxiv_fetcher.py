"""
ArXiv API client for fetching ML paper metadata.

RESPONSIBILITY: This module has ONE job - fetch paper metadata from ArXiv
and return validated, structured data. It does NOT download PDFs (that's
the pdf extractor's job). Single Responsibility  Principle.

Why ArXiv LIBRARY:
    The arxiv Python library wraps the raw XML API response into clean
    Python objects. We could parse XML ourselves with BeautifulSoup,
    but using the official library means we benefit from their bug fixes
    and API changes without rewriting our code.
"""

import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


import arxiv
from pydantic import BaseModel, field_validator

from src.utils.logger import get_logger
from config.settings import (
    RAW_DIR,
    ARXIV_CATEGORIES,
    MAX_PAPERS_PER_FETCH,
    TOTAL_PAPERS_TARGET,
    ARXIV_API_DELAY_SECONDS
)


# Get a named logger for this module
# Every module gets its own named logger - makes debugging trivial
logger = get_logger(__name__)


# -------------------------------------------
# DATA MODEL
# -------------------------------------------

class PaperMetadata(BaseModel):
    """
    Pydantic model defininf the exact schema for a paper's metadata.

    WHY PYDANTIC:
    Pydantic enforces data types at runtime. If ArXiv returns a date
    in an unexpected format, Pydantic raises a clear error immediately
    instead of silently storing bad data that breaks things 3 steps later.

    This is called "fail fast" - catch bad data as early as possible.
    """
    paper_id:           str
    title:              str
    abstract:           str
    authors:            list[str]
    categories:         list[str]
    primary_categories: str
    published_date:     str         # ISO Format: "2023-01-17"
    updated_date:       str
    arxiv_url:          str
    pdf_url:            str

    # Pipeline stage flags - track what processing has been done
    pdf_downloaded:     bool = False
    text_extracted:     bool = False
    chunked:            bool = False
    embedded:           bool = False


    @field_validator("title", "abstract")
    @classmethod
    def clean_whitespace(cls, value: str) -> str:
        """
        Strip excess whitespace from text fields
        ArXiv abstracts often contain \n and multiple spaces
        """
        return " ".join(value.split())

    
    @field_validator("paper_id")
    @classmethod
    def extract_short_id(cls, value: str) -> str:
        """
        ArXiv returns IDs like 'http://arxiv.org/abs/2301.07041v1'
        We want just '2301.07041'
        """
        
        # Split on "/" and take the last part, then remove version suffix
        short_id = value.split("/")[-1]

        if "v" in short_id:
            short_id = short_id.split("v")[0]
        
        return short_id




# -------------------------------------------
# FETCHER CLASS
# -------------------------------------------

class ArXivFetcher:
    """
    Fetches and persists paper metadata from the ArXiv API.

    DESIGN PATTERN: This class is stateless — it doesn't store any
    papers in memory. It fetches, validates, and immediately saves
    to disk. This means if the process crashes at paper #347,
    papers 1-346 are already saved and we can resume.
    """

    def __init__(self):
        # arxiv.Client lets us configure rate limiting behavior
        self.client = arxiv.Client(
            page_size = MAX_PAPERS_PER_FETCH,
            # Delay between API page requests (ArXiv policy: >= 3 Seconds)
            delay_seconds = ARXIV_API_DELAY_SECONDS,
            num_retries = 3     # Retry failed requests automatically
        )


        # File to track which paper IDs we've already downloaded
        # This enables idempotent runs - safe to run pipeline multiple times
        self.index_file = RAW_DIR / "paper_index.json"
        self.existing_ids = self._load_existing_ids()


        logger.info(
            f"ArXivFetcher initialized. "
            f"Already have {len(self.existing_ids)} papers indexed."
        )


    def _load_existing_ids(self) -> set[str]:
        """
        Load set of already-fetched paper IDs from disk

        WHY A SET: Checking 'if paper_id in existing_ids' is O(1) with a set
        versus O(n) with a list. At 10,000 papers, this matters.
        """
        if self.index_file.exists():
            with open(self.index_file, "r") as f:
                data = json.load(f)
                return set(data.get("paper_ids", []))

        return set()

    

    def _save_paper_metadata(self, paper: PaperMetadata) -> Path:
        """
        Save a single paper's metadata as JSON to disk.

        Each paper gets its own JSON file named by its ID.
        WHY NOT A DATABASE: For a pipeline this size, flat JSON files
        are simpler, portable, and Git-friendly. We add a database
        later when we need querying capabilities.
        """

        # e.g., data/raw/2301.07041.json
        file_path = RAW_DIR / f"{paper.paper_id}.json"

        with open(file_path, "w", encoding = 'utf-8') as f:
            # model.dump() converts Pydantic model to dict
            # indent = 2 makes the JSON human-readable
            json.dump(paper.model_dump(), f, indent = 2, ensure_ascii = False)

        
        return file_path


    def _update_index(self, paper_id: str):
        """
        Add paper_id to our index file and memory set.
        Called after every successful save
        """
        self.existing_ids.add(paper_id)

        with open(self.index_file, "w") as f:
            json.dump(
                {
                    "paper_ids": list(self.existing_ids),
                    "last_updated": datetime.now().isoformat(),
                    "total_count": len(self.existing_ids)
                },
                f, indent = 2
            )

    

    def _parse_arxiv_result(self, result: arxiv.Result) -> Optional[PaperMetadata]:
        """
        Convert a raw arxiv.Result object into our PaperMetadata model.

        WHY THIS WRAPPER EXISTS:
        The arxiv library's Result object has its own structure that
        may change across library versions. By converting to our own
        PaperMetadata model here, the rest of our codebase never
        depends on the arxiv library directly. If arxiv changes its
        API tomorrow, we only fix this one function.

        This is called the ADAPTER PATTERN.
        """
        try:
            metadata = PaperMetadata(
                paper_id            = result.entry_id,
                title               = result.title,
                abstract            = result.summary,
                authors             = [str(a) for a in result.authors],
                categories          = result.categories,
                primary_categories  = result.primary_category,
                published_date      = result.published.strftime("%Y-%m-%d"),
                updated_date        = result.updated.strftime("%Y-%m-%d"),
                arxiv_url           = result.entry_id,
                pdf_url             = result.pdf_url, 
            )

            return metadata

        except Exception as e:
            # Log warning but don't crash - one bad paper shouldn't
            # stop the entire pipeline
            logger.warning(f"Failed to parse paper: {result.entry_id}: {e}")
            return None



    def fetch_papers(
        self,
        categories: list[str] = None,
        max_papers: int = None,
        date_filter_year: Optional[int] = None
    ) -> list[PaperMetadata]:
        """
        Main method: fetch papers from ArXiv for given categories.

        Args:
            categories:         ArXiv category codes e.g. ["cs.LG", "cs.AI"]
            max_papers:         Maximum papers to fetch
            date_filter_year:   Only fetch papers from this years onwards

        Returns:
            List of validated PaperMetaData objects

        HOW THE QUERY WORKS:
        ArXiv search syntax uses boolean operators.
        'cat:cs.LG' OR 'cat:cs.AI' means "Papers in cs.LG OR cs.AI category"
        We sort by submission date (newest first) to get fresh papers.
        """

        if categories is None:
            categories = ARXIV_CATEGORIES
        if max_papers is None:
            max_papers = TOTAL_PAPERS_TARGET

        # Build search query: "cat:cs.LG OR cat:cs.AI"
        category_query = " OR ".join([f"cat:{cat}" for cat in categories])
        logger.info(f"Search query: '{category_query}'")
        logger.info(f"Target: '{max_papers} papers'")


        # Configure ArXiv search
        search = arxiv.Search(
            query       = category_query,
            max_results = max_papers * 2,    # Fetch extra account for skips
            sort_by     = arxiv.SortCriterion.SubmittedDate,
            sort_order  = arxiv.SortOrder.Descending,
        )

        fetched_papers = []
        skipped_duplicate = 0
        skipped_invalid = 0

        logger.info("Starting ArXiv fetch...")

        # self.client.results() is a GENERATOR
        # WHY GENERATOR: It fetches pages lazily - doesn't load all 500
        # papers into memory at once. Memory efficient.
        for result in self.client.results(search):

            # Stop if we've reached our target
            if len(fetched_papers) >= max_papers:
                break

            # Skip papers we already have
            raw_id = result.entry_id.split("/")[-1].split("v")[0]
            if raw_id in self.existing_ids:
                skipped_duplicate += 1
                continue

            # Apply year filter if specified
            if date_filter_year and result.published.year < date_filter_year:
                continue

            # Parse and validate
            paper = self._parse_arxiv_result(result)
            if paper is None:
                skipped_invalid += 1
                continue

            # Save to disk immediately
            self._save_paper_metadata(paper)
            self._update_index(paper.paper_id)
            fetched_papers.append(paper)

            # Progress logging every 10 papers
            if len(fetched_papers) % 10 == 0:
                logger.info(
                    f"Progress: {len(fetched_papers)}/{max_papers} papers fetched"
                )

        
        logger.info(
            f"Fetch complete."
            f"Fetched: {len(fetched_papers)} | "
            f"Skipped (duplicate): {skipped_duplicate} | "
            f"Skipped (invalid): {skipped_invalid}"
        )

        return fetched_papers