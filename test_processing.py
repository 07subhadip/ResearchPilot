from src.utils.logger import setup_logger, get_logger
from src.processing.text_cleaner import clean_text

setup_logger()
logger = get_logger(__name__)

# Simulate dirty PDF text
dirty_text = """
arXiv:2301.07041v2  [cs.LG]  17 Jan 2023

We propose a novel at-
tention mechanism that re-
duces computational com-
plexity significantly.

This method achieves state-of-the-art results.

2

ICML 2023 Workshop

The key insight is that sparse attention patterns
can approximate full attention with minimal quality loss.

References

Vaswani, A., et al. (2017). Attention is all you need.
Brown, T., et al. (2020). Language models are few-shot learners.
"""

cleaned = clean_text(dirty_text)

logger.info("─── DIRTY TEXT ───")
print(dirty_text[:300])
logger.info("─── CLEANED TEXT ───")
print(cleaned)
logger.info(f"Original length: {len(dirty_text)}")
logger.info(f"Cleaned length:  {len(cleaned)}")