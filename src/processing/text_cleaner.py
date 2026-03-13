"""
Text normalization utilities for extracted PDF content.

These functions are PURE FUNCTIONS — they take a string,
return a string, have no side effects, and are independently
testable. This is the correct way to write data transformation
logic.
"""

import re
import unicodedata
import ftfy

from src.utils.logger import get_logger

logger = get_logger(__name__)


def fix_hyphenated_linebreaks(text: str) -> str:
    """
    Fix words broken across lines with hyphens.

    Research PDFs use justified text with hyphenation:
        "This is a demon-
         stration of the problem"
    
    Should become:
        "This is a demonstration of the problem"
    
    REGEX EXPLANATION:
        ([a-zA-Z])   -> capture a letter (end of line fragment)
        -            -> literal hyphen
        \n           -> newline
        \s*          -> optional whitespace on next line
        ([a-zA-Z])   -> capture a letter (start of continuation)
    """

    return re.sub(r'([a-zA-Z])-\n\s*([a-zA-Z])', r'\1\2', text)



def remove_page_artifacts(text: str) -> str:
    """
    Remove common PDF page artifacts that pollute extracted text.

    Handles:
    - Form feed characters (\x0c) that mark page boundaries
    - Standalone page numbers (lines containing only digits)
    - Running headers/footers (short lines that repeat)
    """

    # Remove form feed characters (page breaks)
    text = text.replace('\x0c', '\n')

    lines = text.split('\n')
    cleaned_lines = []


    for line in lines:
        stripped = line.strip()

        # Skip empty lines (we'll normalize spacing later)
        if not stripped:
            cleaned_lines.append('')
            continue

            
        # Skip standalone page numbers: lines that are ONLY digits
        # e.g., "12", "247"
        if re.match(r'^\d{1,4}$', stripped):
            continue

        # Skip lines that look like page header/footers
        # Pattern: short lines with mostly uppercase or digits
        # e.g., "NEURIPS 2023", "arXiv:2301.07041v2"
        # FIX: Check if the line CONTAINS these patterns anywhere,
        # not just at the start. Also expanded patterns.
        artifact_patterns = [
            r'arXiv:\d{4}\.\d+',           # arXiv:2301.07041v2
            r'^doi:\s*10\.',               # DOI lines
            r'Preprint\.\s*Under review',  # "Preprint. Under review"
            r'Under review',               # Review notice
            r'Proceedings of (ICML|NeurIPS|ICLR|CVPR|ACL|EMNLP)',
            r'(ICML|NeurIPS|ICLR|CVPR|ACL|EMNLP)\s+20\d{2}',  # "ICML 2023"
            r'Workshop on',                # Workshop lines
            r'^\*+Equal contribution',     # Footnotes
            r'^\dDepartment of',           # Affiliation footnotes
            r'^\d+University of',          # University affiliations
            r'Correspondence to:',         # Contact info
        ]

        is_artifacts = False
        for pattern in artifact_patterns:
            if re.search(pattern, stripped, re.IGNORECASE):
                is_artifacts = True
                break

        if is_artifacts:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def normalize_whitespace(text: str) -> str:
    """
    Normalize all forms of whitespace to standard single spaces.

    PDFs produce various whitespace characters:
    - Multiple consecutive spaces (from column alignment)
    - Tabs
    - Non-breaking spaces (\xa0)
    - Zero-width spaces
    
    STRATEGY:
    1. Replace all non-newline whitespace with single space
    2. Collapse multiple newlines into max double newline
       (preserving paragraph breaks)
    3. Strip leading/trailing whitespace
    """

    # Replace tabs and non-breaking spaces with regular space
    text = text.replace('\t', ' ')
    text = text.replace('\xa0', ' ')

    # Collapse multiple spaces into one
    # re.sub with pattern ' +' matches one or more spaces
    text = re.sub(r' +', ' ', text)

    # Collapse 3+ consecutive newlines into exactly 2
    # (preserves paragraph breaks without excessive gaps)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip each line individually, then rejoin
    lines = [line.strip() for line in text.split('\n')] 
    text = '\n'.join(lines)

    return text.strip()



def fix_unicode(text: str) -> str:
    """
    Fix broken Unicode encoding common in PDF text extraction.

    PDFs often have encoding issues:
    - "â€™" instead of "'" (UTF-8 read as Latin-1)
    - "Ã©" instead of "é"
    - Ligature characters: "ﬁ" (fi ligature) instead of "fi"

    ftfy (Fixes Text For You) handles all these cases automatically.
    It was created at Luminoso and is used in production at scale.
    """
    return ftfy.fix_text(text)



def remove_reference_section(text: str) -> str:
    """
    Remove the bibliography/references section from papers.

    WHY: References contain hundreds of author names, journal names,
    and years. These would pollute our vector index — if someone asks
    about "attention mechanisms", we don't want to retrieve a chunk
    that's just a list of citations like:
    "Vaswani, A., Shazeer, N., Parmar, N., ... (2017). Attention is all you need."

    APPROACH: Find the last occurrence of a "References" header and
    remove everything after it. We use LAST occurrence because some
    papers have "Related Work" sections that reference other sections
    before the actual bibliography.
    """
    # Patterns that signal start of references section
    # re.IGNORECASE to handle "References", "REFERENCES", "Bibliography"
    # FIX: More robust patterns that handle varied spacing
    referece_patterns = [
        r'\n\s*References\s*\n',
        r'\n\s*REFERENCES\s*\n',
        r'\n\s*Bibliography\s*\n',
        r'\n\s*BIBLIOGRAPHY\s*\n',
        r'\n\s*\d+\.\s*References\s*\n',
        r'\n\s*\d+\s+References\s*\n',
        # Handle case where References appears after a section number
        r'\nReferences$',           # At end of line 
    ]


    last_match_pos = -1

    for pattern in referece_patterns:
        # Find all matches, take the last one
        matches = list(re.finditer(pattern, text, re.MULTILINE))
        if matches:
            # Take position of the last match
            pos = matches[-1].start()
            if pos > last_match_pos:
                last_match_pos = pos

    
    if last_match_pos > 0:
        # Only remove if references is in the last 40% of document
        # Increased from 30% because some papers have long conclusions
        cutoff_threshold = len(text) * 0.60
        if last_match_pos > cutoff_threshold:
            text = text[:last_match_pos]
            logger.debug('References section removed')
        else:
            logger.debug(
                f"Reference found at {last_match_pos/len(text):.0%} "
                f"- too early to be bibliography, keeping"
            )

    return text




def remove_short_lines(text: str, min_length: int = 3) -> str:
    """
    Remove lines that are too short to be meaningful content.

    Very short lines in PDFs are usually:
    - Stray characters from column separators
    - Figure/table labels: "Fig.", "Table 1"
    - Single letter section markers
    
    We keep lines >= min_length characters.
    """
    lines = text.split('\n')
    cleaned = [
        line for line in lines
        if len(line.strip()) == 0 or len(line.strip()) >= min_length
    ]

    return '\n'.join(cleaned)


def clean_text(text: str) -> str:
    """
    Master cleaning function — applies all transformations in order.

    ORDER MATTERS:
    1. Fix encoding first (so subsequent regex works on clean chars)
    2. Fix hyphenation (before whitespace normalization)
    3. Remove page artifacts (before whitespace normalization)
    4. Remove references (on mostly clean text)
    5. Remove short lines
    6. Normalize whitespace LAST (cleans up after all other operations)
    """
    if not text or not text.strip():
        return ""

    text = fix_unicode(text)
    text = fix_hyphenated_linebreaks(text)
    text = remove_page_artifacts(text)
    text = remove_reference_section(text)
    text = remove_short_lines(text)
    text = normalize_whitespace(text)


    return text