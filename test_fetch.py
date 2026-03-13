from src.utils.logger import setup_logger
from src.ingestion.arxiv_fetcher import ArXivFetcher

setup_logger()
fetcher = ArXivFetcher()
papers = fetcher.fetch_papers(max_papers = 5)


for p in papers:
    print(f"{p.paper_id}: {p.title[:60]}...")