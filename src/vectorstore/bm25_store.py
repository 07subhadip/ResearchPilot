"""
BM25 sparse retrieval index for keyword-based search.

BM25 (Best Match 25) is the gold standard keyword search algorithm.
It powers Elasticsearch, Solr, and was the backbone of Google Search
before neural methods. It rewards:
    - Term frequency: how often the query word appears in the chunk
    - Inverse document frequency: rare words are more discriminative
    - Document length normalization: prevents long chunks from dominating

WHY WE NEED THIS ALONGSIDE VECTOR SEARCH:
    Query: "what is LoRA fine-tuning?"

    Vector search: finds chunks about "parameter-efficient training"
    (semantically related but may miss the exact acronym)

    BM25: finds chunks containing the EXACT token "LoRA"
    (exact match, regardless of semantic similarity)

    Hybrid: finds chunks that are BOTH semantically relevant
    AND contain the keyword - best of both worlds.
"""

from copyreg import pickle
import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.utils.logger import get_logger
from config.settings import CHUNKS_DIR, EMBEDDINGS_DIR

logger = get_logger(__name__)


# Where we persist the BM25 index
BM25_INDEX_PATH = EMBEDDINGS_DIR / "bm25_index.pkl"



def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25.

    BM25 works on token lists, not raw strings.
    We lowercase and split on non-alphanumeric characters.

    WHY NOT USE NLTK/SPACY:
        For BM25 in a RAG pipeline, simple whitespace+punctuation
        tokenization is sufficient and avoids heavy dependencies.
        The quality difference is minimal for retrieval tasks.
    """
    text = text.lower()

    # Split on anything that't not a letter, number, or hyphen
    tokens = re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', text)
    return tokens



class BM25Store:
    """
    Manages a BM25 index over all chunk texts.

    The index is built once and persisted to disk as a pickle file.
    Loading from pickle is near-instant vs rebuilding from scratch.
    """

    def __init__(self):
        self.bm25:          BM25Okapi = None
        self.chunk_ids:     list[str] = []
        self.texts:         list[str] = []


    def build_index(self) -> None:
        """
        Build BM25 index from all chunk files.

        Loads all chunk texts, tokenizes them, and creates the BM25 index.
        This takes ~30 seconds for 15,664 chunks.
        """
        logger.info("Building BM25 index from chunk files...")

        chunk_ids = []
        texts     = []

        for cf in CHUNKS_DIR.glob("*_semantic.json"):
            with open(cf, "r", encoding = 'utf-8') as f:
                chunks = json.load(f)
            
            for chunk in chunks:
                chunk_ids.append(chunk["chunk_id"])
                texts.append(chunk["text"])

        logger.info(f"Tokenizing {len(texts):,} chunks...")

        # Tokenize all texts
        # bm250kapi expects a list of token lists
        tokenized_corpus = [tokenize(text) for text in texts]

        # Build the BM25 index
        # BM250kapi is the standard 0kapi BM25 variant
        self.bm25       = BM25Okapi(tokenized_corpus)
        self.chunk_ids  = chunk_ids
        self.texts      = texts

        logger.info(f"BM25 index built: {len(chunk_ids):,} documents")

        # Persist to disk
        self._save()


    
    def _save(self) -> None:
        """Save index to disk using pickle."""
        data = {
            "bm25":      self.bm25,
            "chunk_ids": self.chunk_ids,
            "texts":     self.texts,
        }

        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(data, f)
        size_mb = BM25_INDEX_PATH.stat().st_size / 1024 / 1024
        logger.info(f"BM25 index saved: {BM25_INDEX_PATH} ({size_mb:.1f} MB)")


    
    def load(self) -> bool:
        """
        Look index from disk
        Return True if loaded, False if index doesn't exists
        """
        if not BM25_INDEX_PATH.exists():
            logger.info("No BM25 index found on disk")
            return False

        logger.info("Loading BM25 index from disk...")
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)

        self.bm25       = data["bm25"]
        self.chunk_ids  = data["chunk_ids"]
        self.texts      = data["texts"]

        logger.info(f"BM25 index loaded: {len(self.chunk_ids):,} documents")
        return True


    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Search BM25 index with a text query.

        Args:
            query: Raw query string (NOT embedded - BM25 uses tokens)
            top_k: Number of top results to return

        Returns:
            List of dicts with chunk_id, bm25_score, text

        HOW BM25 SCORING WORKS:
            Given query tokens ["lora", "fine-tuning"],
            BM25 scores each document based on how frequently
            these tokens appear, weighted by their rarity across
            all documents (IDF) and normalized by document length.
            Higher score = better keyword match.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not loaded. Call build_index() or load() first.")

        query_tokens = tokenize(query)

        if not query_tokens:
            return []


        # get_scores returns array of shape (n_documents,)
        # with BM25 score for each document
        scores = self.bm25.get_scores(query_tokens)


        # Get indices of top-k scores (argsort ascending, take last k, reverse)
        top_indices = np.argsort(scores)[-top_k:][::-1]


        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                # Skip zero-score results - no keywords overlap at all
                continue
            results.append(
                {
                    "chunk_id":     self.chunk_ids[idx],
                    "bm25_score":   round(score, 4),
                    "text":         self.texts[idx],
                }
            )

        return results