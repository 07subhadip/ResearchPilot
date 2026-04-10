"""
Start the ResearchPilot FastAPI server.

Run from project root:
    python run_api.py

Then visit:
    http://localhost:8000/docs    <- Interactive API documentation
    http://localhost:8000/health  <- Health check
    http://localhost:8000/        <- API info
"""

import uvicorn
from config.settings import API_HOST, API_PORT, API_RELOAD

if __name__ == "__main__":
    print("Starting ResearchPilot API...")
    print(f"API docs: http://localhost:{API_PORT}/docs")
    print(f"Health:   http://localhost:{API_PORT}/health")

    uvicorn.run(
        "src.api.main:app",
        host    = API_HOST,
        port    = API_PORT,
        reload  = False,              # Disable auto-reload (saves ~10s scanning 3000+ data files)
        workers = 1,
    )