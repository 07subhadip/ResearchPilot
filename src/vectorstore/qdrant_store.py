"""
Qdrant vector database interface for ResearchPilot.

RUNS LOCALLY - no server needed, no Docker, no cloud account.
Qdrant client in local mode stores everything in a directory
on disk, exactly like SQLite does for relational data.

Data lives in: data/qdrant_db/
"""

import json
import uuid
import numpy as np
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    SearchRequest,
)
from tqdm import tqdm


from src.utils.logger import get_logger
from config.settings import (
    QDRANT_COLLECTION_NAME,
    QDRANT_PATH,
    EMBEDDING_DIMENSION,
    TOP_K_RETRIEVAL,
)

logger = get_logger(__name__)

# How many points to upload to Qdrant at once
# Too large = memory spike. Too small = many round trips.
UPSERT_BATCH_SIZE = 256


class QdrantStore:
    """
    Manages the Qdrant vector database for chunk storage and retrieval.

    UPSERT PATTERN:
    We use 'upsert' (update + insert) instead of 'insert'.
    If a chunk already exists, upsert updates it.
    If it doesn't exist, upsert creates it.
    This makes our indexing pipeline idempotent - safe to re-run.
    """

    def __init__(self):
        # Local mode: pass path= instead of url=
        # Qdrant creates/opens a local database at this path
        # No server process needed - runs in-process
        logger.info(f"Connecting to local Qdrant at: {QDRANT_PATH}")
        self.client = QdrantClient(path = QDRANT_PATH)
        self.collection_name = QDRANT_COLLECTION_NAME

    
    def collection_exists(self) -> bool:
        """Check if our collection already exists in Qdrant."""
        collections = self.client.get_collections().collections
        names = [c.name for c in collections]
        return self.collection_name in names


    def get_collection_size(self) -> int:
        """Return number of points currently in the collections."""
        if not self.collection_exists():
            return 0
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    
    def create_collection(self, recreate: bool = False):
        """
        Create the Qdrant collection for research paper chunks.

        Args:
            recreate: If True, delete existing collection and rebuild.
                      Use this when you want a clean re-index.

        COLLECTION CONFIGURATION:
            size=768      -> matches BGE-base-en-v1.5 output dimension
            distance=COSINE -> similarity metric

        WHY COSINE DISTANCE:
            Our embeddings are L2-normalized (magnitude = 1.0).
            For normalized vectors: cosine_similarity = dot_product
            Qdrant's COSINE metric handles this correctly.
            Using DOT_PRODUCT would also work but COSINE is more explicit.
        """

        if self.collection_exists():
            if recreate:
                logger.warning(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                size = self.get_collection_size()
                logger.info(
                    f"Collection: '{self.collection_name}' already exists "
                    f"with {size:,} points. Skipping creation."
                )
                return

        logger.info(f"Creating collection: {self.collection_name}")
        self.client.create_collection(
            collection_name = self.collection_name,
            vectors_config = VectorParams(
                size        = EMBEDDING_DIMENSION,
                distance    = Distance.COSINE,
            ),
        )
        logger.info(f"Collection created: {self.collection_name}")


    def index_chunks(
        self,
        embeddings: np.ndarray,
        chunk_ids:  list[str],
        metadata:   list[dict],
        texts:      list[str]
    ) -> int:
        """
        Upload embeddings + metadata into Qdrant.

        Args:
            embeddings:  numpy array (N, 768)
            chunk_ids:   list of N chunk ID strings
            metadata:    list of N metadata dicts
            texts:       list of N chunk text strings

        Returns:
            Number of points successfully indexed

        QDRANT POINT STRUCTURE:
            Each point needs:
            - id:      unique identifier (we use the chunk_id UUID)
            - vector:  the embedding as a Python list of floats
            - payload: dict of any metadata we want to store/filter

        WHY INCLUDE TEXT IN PAYLOAD:
            When we retrieve a point, we need the text to show to the
            user and to send to the LLM. Storing it in the payload
            means ONE database query returns everything we need.
            Alternative would be a separate text lookup - slower and
            more complex.
        """
        assert len(embeddings) == len(chunk_ids) == len(metadata) == len(texts), \
            "All inputs must have the same length"

        total_indexed = 0

        # Process in batches to avoid memory spikes
        for batch_start in tqdm(
            range(0, len(embeddings), UPSERT_BATCH_SIZE),
            desc = "Indexing into Qdrant"
        ):
            batch_end = min(batch_start + UPSERT_BATCH_SIZE, len(embeddings))

            # Build PointStruct objects for this batch
            points = []
            for i in range(batch_start, batch_end):
                # Qdrant requires UUID format for point IDs
                # Our chunk_ids are already UUIDs from Phase 5
                point = PointStruct(
                    id      = chunk_ids[i],
                    vector  = embeddings[i].tolist(),    # Numpy -> Python List
                    payload = {
                        # Store ALL metadata in payload for retrieval
                        **metadata[i],
                        "text": texts[i],   # Include chunk text
                        "publication_year": int(metadata[i].get("published_date", "0000")[:4]),
                    }
                )
                points.append(point)

            # Upsert this batch
            self.client.upsert(
                collection_name = self.collection_name,
                points          = points,
            )
            total_indexed += len(points)


        logger.info(
            f"Indexing complete. "
            f"Total points in collection: {self.get_collection_size():,}"
        )
        return total_indexed


    def search(
        self,
        query_vector:    np.ndarray,
        top_k:           int = TOP_K_RETRIEVAL,
        filter_category: Optional[str] = None,
        filter_year_gte: Optional[int] = None,
    ) -> list[dict]:
        """
        Search for most similar chunks to a query vector.

        Args:
            query_vector:    768-dimensional query embedding
            top_k:           How many results to return
            filter_category: Only return chunks from this ArXiv category
            filter_year_gte: Only return chunks from this year or later

        Returns:
            List of result dicts, each containing:
            {
                "chunk_id":    str,
                "score":       float (cosine similarity, 0-1),
                "text":        str,
                "paper_id":    str,
                "title":       str,
                "authors":     list,
                "published_date": str,
                ...all other payload fields
            }

        FILTERING IN QDRANT:
            Qdrant applies metadata filters DURING vector search,
            not after. This means it only scores vectors that match
            the filter - much faster than post-filtering.

            Example: filter_year_gte=2024 means:
            "Find the top-20 most similar vectors, but ONLY consider
             vectors from papers published in 2024 or later"
        """
        # Build optional filter
        qdrant_filter = self._build_filter(filter_category, filter_year_gte)


        # Execute search
        results = self.client.query_points(
            collection_name = self.collection_name,
            query           = query_vector.tolist(),
            limit           = top_k,
            query_filter    = qdrant_filter,
            with_payload    = True,      # Return metadata with results
            with_vectors    = False      # Don't return the vectors (saves bandwidth)
        ).points

        # Convert Qdrant ScoredPoint objects to plain dicts
        return [
            {
                "chunk_id": str(r.id),
                "score"   : round(r.score, 4),
                **r.payload,    # Unpack all payload fields (text, title, etc.)
            }
            for r in results
        ]


    def _build_filter(
        self,
        category:   Optional[str],
        year_gte:   Optional[int],
    ) -> Optional[Filter]:
        """
        Build a Qdrant filter from optional parameters.

        Returns None if no filters specified (search everything).

        QDRANT FILTER SYNTAX:
            Filter(must=[condition1, condition2])
            means: results must satisfy condition1 AND condition2

            MatchValue -> exact match (equality check)
            Range      -> numeric range (gte, lte, gt, lt)
        """
        conditions = []

        if category:
            conditions.append(
                FieldCondition(
                    key   = "primary_category",
                    match = MatchValue(value = category) 
                )
            )

        if year_gte:
            # publication_year is stored as an integer (e.g. 2026)
            # Range(gte=year_gte) filters to papers from that year onwards
            conditions.append(
                FieldCondition(
                    key     = "publication_year",
                    range   = Range(gte = year_gte)
                )
            )

        if not conditions:
            return None

        return Filter(must = conditions)


    def get_collection_info(self) -> dict:
        """Return summary information about the collection."""
        if not self.collection_exists():
            return {"status": "collection_not_found"}

        info = self.client.get_collection(self.collection_name)

        return {
            "collection_name": self.collection_name,
            "points_count"   : info.points_count,
            "status"         : str(info.status),
            "vector_size"    : info.config.params.vectors.size,
            "distance"       : str(info.config.params.vectors.distance),
        }