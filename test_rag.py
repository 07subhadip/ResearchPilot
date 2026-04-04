"""
End-to-end test of the RAG pipeline.
This is the most important test in the project.
"""

from src.utils.logger import setup_logger, get_logger
from src.rag.pipeline import RAGPipeline


setup_logger()
logger = get_logger(__name__)



def ask(pipeline: RAGPipeline, question: str, **kwargs):
    print(f"\n{'='*65}")
    print(f"Q: {question}")
    print(f"{'='*65}")

    response = pipeline.query(question, **kwargs)

    print(f"\nANSWER:\n{response.answer}")

    print(f"\nSOURCES ({len(response.citations)}):")
    for i, cite in enumerate(response.citations, 1):
        print(f"  [{i}] {cite['paper_id']} — {cite['title'][:60]}...")
        print(f"       {cite['arxiv_url']}")

    print(f"\nTIMING:")
    print(f"  Retrieval:  {response.retrieval_time_ms:.0f}ms")
    print(f"  Generation: {response.generation_time_ms:.0f}ms")
    print(f"  Total:      {response.total_time_ms:.0f}ms")
    print(f"  Chunks used: {len(response.retrieved_chunks)}")


def main():
    logger.info("Initializing RAG pipeline...")
    pipeline = RAGPipeline()

    # Test 1: Specific technical question
    ask(
        pipeline,
        "What is LoRA and how does it reduce the number of trainable parameters?"
    )

    # Test 2: Comparison question
    ask(
        pipeline,
        "What are the main challenges in multi-agent reinforcement learning?"
    )

    # Test 3: Question that may not be in corpus
    ask(
        pipeline,
        "What is the history of the Python programming language?"
    )


if __name__ == "__main__":
    main()