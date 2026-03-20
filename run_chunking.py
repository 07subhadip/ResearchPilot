"""
Run the chunking pipeline on all processed documents.

OPTIMIZATION: Checks existing chunks before loading model,
so if everything is already chunked, we exit immediately
without loading 110MB embedding model.
"""


import json
from pathlib import Path

from src.utils.logger import setup_logger, get_logger
from src.processing.chunker import ChunkingPipeline
from config.settings import PROCESSED_DIR, CHUNKS_DIR



setup_logger()
logger = get_logger(__name__)



def count_remaining(strategy: str) -> int:
    """Count how many papers still need chunking."""

    processed = list(PROCESSED_DIR.glob("*.json"))
    remaining = 0

    for f in processed:
        paper_id    = f.stem
        output_path = CHUNKS_DIR / f"{paper_id}_{strategy}.json"

        if not output_path.exists():
            remaining += 1

        return remaining



def main():
    strategy    = 'semantic'
    remaining   = count_remaining(strategy)


    logger.info(f"Papers remaining to chunk: {remaining}")


    if remaining == 0:
        logger.info("All papers already chunked. Nothing to do.")

        # Print summary of existing chunks
        chunk_files = list(CHUNKS_DIR.glob(f"*_{strategy}.json"))
        total = 0
        for cf in chunk_files:
            with open(cf) as f:
                chunks = json.load()

            total += len(chunks)

        logger.info(f"Existing chunks: {total} across {len(chunk_files)} papers")

    logger.info(f"Starting chunking pipeline for {remaining} papers...")
    pipeline = ChunkingPipeline(strategy = strategy)
    stats    = pipeline.run(PROCESSED_DIR)
    logger.info(f"Done: {stats}")


if __name__ == "__main__":
    main()