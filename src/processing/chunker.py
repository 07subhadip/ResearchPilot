"""
Document chunking strategies for ResearchPilot.

Three strategies implemented:
  1. FixedSizeChunker      — baseline, educational
  2. RecursiveChunker      — production standard
  3. SemanticChunker       — highest quality, used in final system

Each chunker produces a list of Chunk objects with identical
structure so the rest of the pipeline doesn't care which
strategy was used. This is the STRATEGY PATTERN in software design.
"""

import re
import uuid
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logger import get_logger
from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    CHUNKS_DIR,
    EMBEDDING_MODEL_NAME
)


logger = get_logger(__name__)



# ---------------------------------------------------------
# DATA MODEL
# ---------------------------------------------------------

@dataclass
class Chunk:
    """
    A single chunk of text with all metadata attached.

    WHY ATTACH METADATA TO EVERY CHUNK:
    When a user asks a question and we retrieve chunk #347,
    we need to know: which paper did this come from? What page?
    What section? Without metadata on the chunk itself, we'd
    have to do a separate lookup — slow and error-prone.

    Every chunk is self-contained and self-describing.
    """

    # Unique identifier for this chunk
    # uuid4() generates a random unique ID - no two chunks collide
    chunk_id:           str


    # The actual text content
    text:               str


    # Which paper this came from
    paper_id:           str
    title:              str
    authors:            list[str]
    published_date:     str
    primary_category:   str
    arxiv_url:          str

    # Position within the document
    chunk_index:        int         # 0, 1, 2, ...(position in paper)
    total_chunks:       int         # How many chunks this paper was split into

    # Text statistics
    char_count:         int
    word_count:         int

    # Chunking metadata
    chunking_strategy:  str         # 'fixed', 'recursive', 'semantic'



    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization"""
        return asdict(self)

    
    @property
    def is_valid(self) -> bool:
        """Check if chunk has enough content to be useful."""
        return (
            len(self.text.strip()) >= MIN_CHUNK_SIZE and
            self.word_count >= 10   # At least 10 words
        )



# ---------------------------------------------------------
# STRATEGY 1: FIXED SIZE CHUNKER
# ---------------------------------------------------------

class FixedSizeChunker:
    """
    Splits text every N characters with M character overlap.

    This is the WORST chunking strategy but we include it as:
    1. A baseline to compare against
    2. To demonstrate WHY better strategies exist
    3. Educational — see exactly what breaks

    OVERLAP EXPLAINED:
    Without overlap:
      Chunk 1: "The model achieves 94.2% accuracy on"
      Chunk 2: "GLUE benchmark, beating prior work by"

    The phrase "accuracy on GLUE" is split — neither chunk
    contains the complete concept.

    With 50-char overlap:
      Chunk 1: "The model achieves 94.2% accuracy on"
      Chunk 2: "accuracy on GLUE benchmark, beating prior work by"

    Now "accuracy on GLUE" exists in chunk 2. Retrieval works.
    Overlap is a band-aid for fixed-size chunking's core problem.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap


    def split(self, text: str, metadata: dict) -> list[Chunk]:
        chunks  = []
        start   = 0
        index   = 0


        while start < len(text):
            end         =  min(start + self.chunk_size, len(text))
            chunk_text  = text[start : end].strip()

            if len(chunk_text) > MIN_CHUNK_SIZE:
                chunks.append(self._make_chunk(chunk_text, index, metadata))
                index += 1

            
            # Move forward by (chunk_size - overlap)
            # This creates the sliding window effect
            start += self.chunk_size - self.overlap


        # Now that we know total_chunks, update all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks


    
    def _make_chunk(self, text: str, index: int, meta: dict) -> Chunk:
        return Chunk(
            chunk_id            = str(uuid.uuid4()),
            text                = text,
            paper_id            = meta['paper_id'],
            title               = meta['title'],
            authors              = meta['authors'],
            published_date      = meta['published_date'],
            primary_category    = meta['primary_category'],
            arxiv_url           = meta['arxiv_url'],
            chunk_index         = index,
            total_chunks        = 0,        # Updated after all chunks created
            char_count          = len(text),
            word_count          = len(text.split()),
            chunking_strategy   = 'fixed', 
        )



# ---------------------------------------------------------
# STRATEGY 2: RECURSIVE CHARACTER SPLITTER
# ---------------------------------------------------------

class RecursiveChunker:
    """
    Splits text by trying delimiters in order of preference.

    DELIMITER HIERARCHY:
        1. "\n\n"   -> paragraph break (best — complete thought)
        2. "\n"     -> line break (good)
        3. ". "     -> sentence end (acceptable)
        4. " "      -> word boundary (last resort)
        5. ""       -> character (never want this)

    The splitter tries to split at \n\n first. If a resulting piece
    is still too large, it tries \n. Still too large? Tries ". " etc.

    WHY THIS IS BETTER THAN FIXED:
    Fixed chunking: "...achieves 94.2% ac" + "curacy on GLUE..."
    Recursive:      "...achieves 94.2% accuracy on GLUE benchmark."

    Recursive splitting respects natural language boundaries.
    The resulting chunks contain complete sentences and paragraphs.

    THIS IS THE INDUSTRY STANDARD. Use this unless you have a
    specific reason to use semantic chunking.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP
    ):
        # LangChain's implementation is well-tested and handles
        # many edge cases we'd miss writing our own
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size          = chunk_size,
            chunk_overlap       = overlap,
            length_function     = len,
            # Separators tried in order - most preferred first
            separators          = ["\n\n", "\n", ". ", " ", ""],
            # Keep separator at the end of the chunk (preserves sentence endings)
            keep_separator      = True,
        )



    def split(self, text: str, metadata: dict) -> list[Chunk]:
        # LangChain splits the text into string pieces
        text_pieces = self.splitter.split_text(text)


        chunks = []

        for index, piece in enumerate(text_pieces):
            piece = piece.strip()


            if len(piece) < MIN_CHUNK_SIZE:
                continue

            chunk = Chunk(
                chunk_id          = str(uuid.uuid4()),
                text              = piece,
                paper_id          = metadata["paper_id"],
                title             = metadata["title"],
                authors           = metadata["authors"],
                published_date    = metadata["published_date"],
                primary_category  = metadata["primary_category"],
                arxiv_url         = metadata["arxiv_url"],
                chunk_index       = index,
                total_chunks      = 0,
                char_count        = len(piece),
                word_count        = len(piece.split()),
                chunking_strategy = "recursive",
            )

            chunks.append(chunk)
        

        for chunk in chunks:
            chunk.total_chunks = len(chunks)


        return chunks




# ---------------------------------------------------------
# STRATEGY 3: SEMANTIC CHUNKER
# ---------------------------------------------------------

class SemanticChunker:
    """
    Splits text at points where the semantic meaning changes.

    THE CORE INSIGHT:
    In a research paper, adjacent sentences that discuss the
    SAME idea have HIGH embedding similarity.
    When the topic shifts (e.g., from "method" to "results"),
    the similarity between adjacent sentences DROPS sharply.

    We find these DROP POINTS and split there.

    ALGORITHM:
    1. Split text into individual sentences
    2. Embed every sentence using BGE model
    3. Calculate cosine similarity between each adjacent pair:
       sim(sentence_1, sentence_2), sim(sentence_2, sentence_3), ...
    4. Find similarity values that drop below a threshold
       (these are semantic boundaries)
    5. Split the document at those boundary points
    6. Each resulting chunk contains sentences about ONE topic

    VISUAL EXAMPLE:
    Sentence similarities: [0.92, 0.89, 0.91, 0.45, 0.88, 0.90, 0.38, 0.85]
                                                      ↑ split here    ↑ split here
    The drops at 0.45 and 0.38 mark topic changes.

    WHY THIS MATTERS FOR RESEARCH PAPERS:
    Papers have clear sections: Introduction -> Method -> Experiments -> Conclusion
    Within each section, sentences are semantically close.
    At section transitions, similarity drops sharply.
    Semantic chunking naturally aligns with paper structure.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        # Similarity Threshold: splits happen where similarity < this value
        # 0.5 means "splits when adjacent sentences share < 50% semantic similarity"
        breakpoint_threshold: float = 0.5,
        # Minimum sentences per chunk (avoid 1-sentence chunks)
        min_sentences_per_chunk: int = 3,
        # Maximum sentences per chunk (avoid enormous chunks)
        max_sentences_per_chunk: int = 15,
    ):
        self.breakpoint_threshold    = breakpoint_threshold
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.max_sentences_per_chunk = max_sentences_per_chunk

        # Lazy load the model - only load when first needed
        # WHY: Loading a 110MB model takes ~3 seconds
        # We don't want that delay at import time
        self._model = None
        self._model_name = model_name

        logger.info(
            f"SemanticChunker initialized "
            f"(threshold={breakpoint_threshold})"
        )



    @property
    def model(self):
        """
        Lazy-load the embedding model.

        PROPERTY PATTERN: self.model looks like an attribute but
        actually runs this function the first time it's accessed.
        After first load, self._model is set and returned directly.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded")

        return self._model


    
    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into individual sentences.

        WHY NOT JUST split('.'):
        "Dr. Smith proposed..." -> split on period gives ["Dr", " Smith proposed"]
        "The accuracy was 94.2% on..." -> breaks at decimal
        "et al. showed..." -> breaks at abbreviation

        Our regex handles these cases:
        - Requires capital letter after period (new sentence starts with capital)
        - Handles "." followed by newline
        - Keeps sentences of reasonable length
        """
        # Split on: period/!/? followed by whitespace and capital letter
        # OR on double newlines (paragraph breaks are always sentence breaks)
        sentence_pattern = r'(?<=[.|?])\s+(?=[A-Z])|(?<=\n)\n+'

        sentences = re.split(sentence_pattern, text)


        # Filter out very short fragments (less than 20 chars)
        # These are usually artifacts, not real sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]


        return sentences



    
    def _compute_similarities(self, sentences: list[str]) -> np.ndarray:
        """
        Embed all sentences and compute cosine similarity
        between each adjacent pair.

        Returns array of shape (len(sentences) - 1,)
        where result[i] = cosine_similarity(sentence[i], sentence[i+1])

        BATCH PROCESSING:
        We embed ALL sentences at once (not one by one).
        Batch embedding is 10-50x faster than individual embedding
        because the model processes multiple inputs in parallel.
        """
        if len(sentences) < 2:
            return np.array([])
        
        
        # encode() returns numpy array of shape (n_sentences, embedding_dim)
        # show_progress_bar = False keeps output clean in pipeline
        embeddings = self.model.encode(
            sentences,
            show_progress_bar = False,
            batch_size = 64,            # Process 64 sentences at a time
            normalize_embeddings = True, # L2 Normalize for cosine similarity
            convert_to_numpy = True,    # Skip torch tensor, go direct to numpy
        )


        # Vectorized similarity — faster than Python loop
        # embeddings[:-1] = all sentences except last
        # embeddings[1:]  = all sentences except first
        # Row-wise dot product = cosine similarity for normalized vectors

        similarities = np.sum(
            embeddings[:-1] * embeddings[1:],
            axis = 1
        )

        return similarities




    def _find_split_points(self, similarities: np.ndarray) -> list[int]:
        """
        Find indices where the text should be split.

        ADAPTIVE THRESHOLD:
        Instead of a fixed threshold, we use the mean - std_dev.
        This adapts to each document's similarity distribution.

        WHY ADAPTIVE:
        A technical paper might have overall lower similarities
        than a narrative paper. A fixed threshold of 0.5 might
        split a technical paper every sentence (too granular)
        while never splitting a narrative paper (too coarse).

        Adaptive threshold = "split where similarity is
        significantly below average for THIS document."
        """ 
    
        if len(similarities) == 0:
            return []

        
        # Adaptive threshold: mean minus one standard deviation
        # This finds the "unusually low similarity" points
        mean_sim = np.mean(similarities)
        std_min  = np.std(similarities)
        threshold = max(
            self.breakpoint_threshold,      # Never go above configured max
            mean_sim - std_min              # Adaptive: 1 std below mean
        ) 


        logger.debug(
            f"Similarity stats: mean={mean_sim:.3f}"
            f"std={std_min:.3f}, threshold={threshold:.3f}"
        )



        # Find indices where similarity drops below threshold
        # These are the semantic breakpoints
        split_points = [
            i + 1   # +1 because we split AFTER sentence 1
            for i, sim in enumerate(similarities)
            if sim < threshold
        ]


        return split_points



    
    def _group_sentences_into_chunks(
        self,
        sentences: list[str],
        split_points: list[int]
    ) -> list[str]:
        """
        Group sentences into chunks based on split points,
        respecting min/max sentence constraints.
        """
        if not sentences:
            return []

        # Build groups using split_points as boundaries
        # split_points = [4, 9, 15] means:
        #   Group 1: sentences 0-3
        #   Group 2: sentences 4-8
        #   Group 3: sentences 9-14
        #   Group 4: sentences 15+

        boundaries  = [0] + split_points + [len(sentences)]
        groups       = []  


        for i in range(len(boundaries) - 1): 
            start = boundaries[i]
            end   = boundaries[i + 1]
            group = sentences[start : end]  


            if not group:  
                continue

            # ENFORCE MINIMUM: If group is too small, merge with next
            if len(group) < self.min_sentences_per_chunk and group and groups:  
                # Merge into previous group
                groups[-1].extend(group)  
            else:
                groups.append(group)  


        # ENFORCE MAXIMUM: If group is too large, subdivide it
        final_group = []

        for group in groups:  
            if len(group) <= self.max_sentences_per_chunk:
                final_group.append(group)
            else:
                # Split large groups into max_size pieces
                for i in range(0, len(group), self.max_sentences_per_chunk):
                    sub = group[i : i + self.max_sentences_per_chunk]
                    if sub:
                        final_group.append(sub)


        # Convert sentence lists to text strings
        return [" ".join(group) for group in final_group]



    def split(self, text: str, metadata: dict) -> list[Chunk]:
        """
        Main split method - full semantic chunking pipeline
        """ 
        # Step 1: Split into sentences
        sentences = self._split_into_sentences(text)


        if len(sentences) < 2:
            # Documents too short for semantic analysis - fall back to recursive 
            logger.debug(
                f"Too few sentences ({len(sentences)}) for semantic"
                f"chunking on {metadata['paper_id']}, using recursive"
            )

            return RecursiveChunker().split(text, metadata)


        # Step 2: Compute inter-sentence similarities
        similarities = self._compute_similarities(sentences)

        # Step 3: Find semantic breakpoints
        split_points = self._find_split_points(similarities)

        logger.debug(
            f"{metadata['paper_id']}: {len(sentences)} sentences, "
            f"{len(split_points)} splits points found"
        )

        # Step 4: Group sentences into chunks
        chunk_texts = self._group_sentences_into_chunks(sentences, split_points)

        # Step 5: Build Chunk objects
        chunks = []
        for index, chunk_text in enumerate(chunk_texts):
            chunk_text = chunk_text.strip()

            if len(chunk_text) < MIN_CHUNK_SIZE:
                continue

            chunk = Chunk(
                chunk_id          = str(uuid.uuid4()),
                text              = chunk_text,
                paper_id          = metadata["paper_id"],
                title             = metadata["title"],
                authors           = metadata["authors"],
                published_date    = metadata["published_date"],
                primary_category  = metadata["primary_category"],
                arxiv_url         = metadata["arxiv_url"],
                chunk_index       = index,
                total_chunks      = 0,
                char_count        = len(chunk_text),
                word_count        = len(chunk_text.split()),
                chunking_strategy = "semantic",
            )

            chunks.append(chunk)

        for chunk in chunks:
            chunk.total_chunks = len(chunks)


        logger.debug(
            f"{metadata['paper_id']}: produced {len(chunks)} semantic chunks"
        )

        return chunks




# ---------------------------------------------------------
# PIPELINE RUNNER
# ---------------------------------------------------------


class ChunkingPipeline:
    """
    Orchestrates chunking for all processed papers.

    Takes files from data/processed/ and produces
    chunk files in data/chunks/.
    """


    def __init__(self, strategy: str = 'recursive'):
        """
        Args:
            strategy: "fixed" | "recursive" | "semantic"
        """
        valid = {"fixed", "recursive", "semantic"}

        if strategy not in valid:
            raise ValueError(f"Strategy must be one of {valid}")

        self.strategy_name = strategy


        # Instantiate the correct chunker
        if strategy == "fixed":
            self.chunker = FixedSizeChunker()
        elif strategy == "recursive":
            self.chunker = RecursiveChunker()
        elif strategy == "semantic":
            self.chunker = SemanticChunker()

        
        logger.info(f"ChunkingPipeline initialized with strategy: {strategy}")



    def process_paper(self, processed_doc: dict) -> list[Chunk]:
        """Chunk a single processed document"""
        paper_id = processed_doc['paper_id']
        text     = processed_doc.get("full_text", "")


        if not text:
            logger.warning(f"No text found for {paper_id}")
            return []

        # Metadata dict passes to every chunk
        metadata = {
            "paper_id":         paper_id,
            "title":            processed_doc.get("title", ""),
            "authors":          processed_doc.get("authors", []),
            "published_date":   processed_doc.get("published_date", ""),
            "primary_category": processed_doc.get("primary_category", ""),
            "arxiv_url":        processed_doc.get("arxiv_url", ""),
        }

        return self.chunker.split(text, metadata)


    
    def save_chunks(self, chunks: list[Chunk], paper_id: str):
        """
        Save all chunks for a paper to data/chunks/.

        File format: data/chunks/{paper_id}_{strategy}.json
        Contains list of chunk dicts.
        """
        if not chunks:
            return

        output_path = (
            CHUNKS_DIR / f"{paper_id}_{self.strategy_name}.json"
        )


        with open(output_path, 'w', encoding = 'utf-8') as f:
            json.dump(
                [chunk.to_dict() for chunk in chunks],
                f, indent = 2, ensure_ascii = False
            )

        
    def run(self, process_dir: Path) -> dict:
        """
        Run chunking pipeline on all processed documents.

        Args:
            processed_dir: Path to data/processed/

        Returns:
            Summary statistics
        """
        from tqdm import tqdm

        processed_files = list(process_dir.glob("*.json"))

        logger.info(
            f"Chunking {len(processed_files)} documents "
            f"with '{self.strategy_name}' strategy"
        )


        total_chunks = 0
        successful   = 0
        failed       = 0
        skipped      = 0


        for proc_file in tqdm(processed_files, desc = f"Chunking ({self.strategy_name})"):
            with open(proc_file, 'r', encoding = 'utf-8') as f:
                doc = json.load(f)

            
            paper_id       = doc['paper_id']
            output_path = CHUNKS_DIR / f"{paper_id}_{self.strategy_name}.json"


            # Skip already chunked (idempotent)
            if output_path.exists():
                skipped += 1
                continue

            try:
                chunks = self.process_paper(doc)

                if not chunks:
                    failed += 1
                    continue

                self.save_chunks(chunks, paper_id)
                total_chunks += len(chunks)
                successful   += 1
 
            except Exception as e:
                logger.error(f"Failed to chunk {paper_id}: {e}")
                failed += 1

        stats = {
            "strategy":     self.strategy_name,
            "documents":    len(processed_files),
            "successful":   successful,
            "failed":       failed,
            "skipped":      skipped,
            "total_chunks": total_chunks,
            "avg_chunks_per_doc": (
                round(total_chunks / max(successful, 1), 1)
            ),
        }


        logger.info(f"Chunking complete: {stats}")
        return stats