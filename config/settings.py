"""
Central configuration for ResearchPilot

RULE: No hardcoded values anywhere else in this codebase.
Every constant lives here. This make the system to 
tune without hunting through multiple files.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# This must happen before anything else reads os.environ
load_dotenv()

# ------------------------------------------
# PROJECT PATHS
# ------------------------------------------
# Path(__file__) = config/setting.py 
# .parent =      = config/
# .parent.parent = researchpilot/ <- project root
ROOT_DIR = Path(__file__).parent.parent

DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
CHUNKS_DIR      = DATA_DIR / "chunks"
EMBEDDINGS_DIR  = DATA_DIR / "embeddings"
LOGS_DIR        = ROOT_DIR / "logs"    


# Create directories if they don't exist
# This ensures the app works on any machine without manual setup
for directory in [RAW_DIR, PROCESSED_DIR, CHUNKS_DIR, EMBEDDINGS_DIR, LOGS_DIR]:
    directory.mkdir(
        parents = True,
        exist_ok = True
    )

# ------------------------------------------
# DATA INGESTION SETTINGS
# ------------------------------------------
ARXIV_CATEGORIES = ["cs.LG", "cs.AI"]    # Machine Learning + AI
MAX_PAPERS_PER_FETCH = 100               # Papers per API call
TOTAL_PAPERS_TARGET  = 100               # Total papers to collect
ARXIV_API_DELAY_SECONDS = 3.0            # ArXiv rate limit: be respectful
PDF_DOWNLOAD_TIMEOUT = 30                # Seconds before giving up on a PDF
MAX_DOWNLOAD_RETRIES = 3                 # Retry failed downloads N times

# ------------------------------------------
# DOCUMENT PROCESSING SETTINGS
# ------------------------------------------
MIN_TEXT_LENGTH = 500       # Skip papers with less that 500 chars
MAX_TEXT_LENGTH = 500_000   # Skip papers larger than 100k chars (corrupted)

# ------------------------------------------
# CHUNKING SETTINGS
# ------------------------------------------
CHUNK_SIZE = 512        # Charaters per chunk
CHUNK_OVERLAP = 50      # Overlap between consecutive chunks
MIN_CHUNK_SIZE = 100    # Discard chunks smaller than this

# ------------------------------------------
# EMBEDDING SETTINGS
# ------------------------------------------
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_BATCH_SIZE = 32                       # Process N chunks at once
EMBEDDING_DIMENSION  = 768                      # BGE-base output dimension

# ------------------------------------------
# VECTOR STORE SETTINGS
# ------------------------------------------
QDRANT_COLLECTION_NAME = 'research_papers'
QDRANT_PATH = str(ROOT_DIR / 'data' / 'qdrant_db')  # Local Storage path
TOP_K_RETRIEVAL = 20                                # Retieve top 20 candidates
TOP_K_RERANK = 5                                    # Keep top 5 after reranking

# ------------------------------------------
# LLM SETTINGS
# ------------------------------------------
GROQ_API_KEY = os.getenv('GROQ_API_KEY')    # Loaded from .env
LLM_MODEL_NAME = 'llama3-8b-8192'           # Groq model ID
LLM_TEMPERATURE = 0.1                       # Low = More factual/consistent 
LLM_MAX_TOKENS = 1024                       # Max response tokens

# ------------------------------------------
# API SETTINGS
# ------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True   # Auto-reload on code change (dev-only)

# ------------------------------------------
# LOGGING SETTINGS
# ------------------------------------------
LOG_LEVEL     = "INFO"
LOG_FILE      = LOGS_DIR / "researchpilot.log"
LOG_ROTATION  = "10 MB"                         # Create new log file after 10MB
LOG_RETENTION = "7 days"                        # Keep logs for 7 days