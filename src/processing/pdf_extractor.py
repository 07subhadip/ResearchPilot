"""
Extracts and cleans text from downloaded PDF files.

WHY PYMUPDF (fitz) over alternatives:
    Library        | Speed  |  Quality   | Handles columns?
    ---------------|--------|------------|-----------------
    PyMuPDF        | Fast   |  ★★★★★   | Yes (sort=True)
    pdfplumber     | Medium |  ★★★★☆   | Partial
    pypdf2         | Medium |  ★★★☆☆   | No
    pdfminer       | Slow   |  ★★★★☆   | Partial

PyMuPDF's sort=True parameter reads text in natural reading order
(top-to-bottom, left-to-right) which is critical for multi-column
academic papers.
"""

import json
from pathlib import Path

import fitz  # PyMuPDF - imported as 'fitz' (legacy name from founder)
from tqdm import tqdm

from src.processing.text_cleaner import clean_text
from src.utils.logger import get_logger
from config.settings import (
    RAW_DIR,
    PROCESSED_DIR,
    MIN_TEXT_LENGTH,
    MAX_TEXT_LENGTH
)

logger = get_logger(__name__)



class PDFExtractor:
    """
    Extracts clean text from PDF files and saves to processed directory.
    
    Output structure for each paper:
    data/processed/2301.07041.json  ← cleaned text + original metadata
    """

    def __init__(self):
        self.pdf_dir = RAW_DIR / 'pdfs'

    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract raw text from a PDF using PyMuPDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Raw extracted text string (not yet cleaned)

        HOW PYMUPDF READS PDFS:
        PDF is a page-based format. We iterate each page,
        extract text with sort=True (reading order), then
        join all pages. The 'text' flag tells PyMuPDF to
        extract plain text (vs HTML or dict formats).
        """
        try:
            # Open PDF - fitz.open() handles file reading
            doc = fitz.open(str(pdf_path))


            pages_text = []

            for page_num, page in enumerate(doc):
                # get_text("text", sort = True)
                #   "text" -> plain text extraction mode
                #   sort = True -> respect reading order (critical for columns)
                page_text = page.get_text("text", sort = True)

                if page_text.strip():
                    pages_text.append(page_text)

            # Close the document to free memory
            doc.close()


            # Join all pages with double newline (paragraph seperator)
            full_text = '\n\n'.join(pages_text)
            return full_text


        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path.name}: {e}")
            return ""


    
    def validate_extracted_text(self, text: str, paper_id: str) -> tuple[bool, str]:
        """
        Validate that extracted text is usable.

        Returns:
            (is_valid: bool, reason: str)

        VALIDATION RULES:
        1. Not empty
        2. Long enough to be a real paper (not a 1-page erratum)
        3. Not too long (might indicate extraction corruption)
        4. Contains alphabetic characters (not just symbols/numbers)
        5. Is primarily English (our embedding model is English-optimized)
        """
        if not text:
            return False, "Empty text"

        if len(text) < MIN_TEXT_LENGTH:
            return False, f"Too short: {len(text)} chars < {MIN_TEXT_LENGTH}"

        if len(text) > MAX_TEXT_LENGTH:
            return False, f"Too long: {len(text)} chars > {MAX_TEXT_LENGTH}"

        
        # Check that text contains substantial alphabetic content
        # (not just numbers, equations, or garbled encoding)
        alpha_chars  = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_chars / len(text)


        if alpha_ratio < 0.4:
            return False, f"Low alphanumeric ration: {alpha_ratio:.2f} (likely encoding issue)"

        return True, "Valid"



    def process_paper(self, paper_metadata: dict) -> bool:
        """
        Full pipeline for one paper: extract -> clean -> validate -> save.

        Args:
            paper_metadata: dict loaded from data/raw/{paper_id}.json

        Returns:
            True if processed successfully, False otherwise
        """
        paper_id = paper_metadata['paper_id']

        # Skip if already processed (idempotent)
        output_path = PROCESSED_DIR / f'{paper_id}.json'
        if output_path.exists():
            logger.debug(f"Already processed: {paper_id}")
            return True

        # Check PDF exists
        pdf_path = self.pdf_dir / f"{paper_id}.pdf"
        if not pdf_path.exists():
            logger.warning(f"PDF not found for {paper_id}, using abstract only")
            # FALLBACK: Use abstract as the text source
            # Abstract is short but better than nothing
            # This handles cases where PDF download failed
            text = paper_metadata.get("abstract", "")
            if not text:
                return False

        else:
            # Extract from PDF
            raw_text = self.extract_text_from_pdf(pdf_path)


            # Clean the text
            text = clean_text(raw_text)

            
        # Validate
        is_valid, reason = self.validate_extracted_text(text, paper_id)
        if not is_valid:
            logger.warning(f"Validation failed for {paper_id}: {reason}")
            return False

        # Build processed document
        #---------------------------------------------------------------------------
        # processed_doc = {
        #     # Copy all original metadata
        #     **paper_metadata,

        #     # Add processed text
        #     "full_text": text,
        #     "text_length": len(text),
        #     "word_count": len(text.split()),

        #     # Update pipeline state
        #     "text_extracted": True,
        #     "pdf_downloaded": paper_metadata.get("pdf_downloaded", False),
        # }
        #---------------------------------------------------------------------------

        primary_cat = paper_metadata.get("primary_category")

        if not primary_cat:
            cats = paper_metadata.get("categories", [])
            primary_cat = cats[0] if cats else "cs.LG" 

        processed_doc = {
            **paper_metadata,
            "primary_category": primary_cat,   # Override with rescued value
            "full_text": text,
            "text_length": len(text),
            "word_count": len(text.split()),
            "text_extracted": True,
            "pdf_downloaded": paper_metadata.get("pdf_downloaded", False),
        }


        # Save to processed directory
        with open(output_path, "w", encoding = 'utf-8') as f:
            json.dump(processed_doc, f, indent = 2, ensure_ascii = False)

        logger.debug(
            f"Processed {paper_id}: "
            f"{processed_doc['word_count']} words, "
            f"{len(text)} chars"
        )

        return True



    def process_all(self) -> dict:
        """
        Process all papers that have been fetched.

        Loads metadata from data/raw/, extracts text,
        saves results to data/processed/.
        """
        # Load all paper metadata from raw directory
        raw_files = [
            f for f in RAW_DIR.glob("*.json")
            if f.name != "paper_index.json" 
        ]


        logger.info(f"Found {len(raw_files)} papers to process")

        successful = 0
        failed     = 0
        skipped    = 0



        for raw_file in tqdm(raw_files, desc = "Extracting text"):
            with open(raw_file, 'r', encoding = 'utf-8') as f:
                metadata = json.load(f)

            # Skip if already processed
            output_path = PROCESSED_DIR / f"{metadata['paper_id']}.json"
            if output_path.exists():
                skipped += 1
                continue

            success = self.process_paper(metadata)
            if success:
                successful += 1
            else:
                failed += 1

            
        stats = {
            "total":      len(raw_files),
            "successful": successful,
            "failed":     failed,
            "skipped":    skipped,
        }

        logger.info(f"Processing complete: {stats}")
        return stats