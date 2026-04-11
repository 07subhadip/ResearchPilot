import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getcwd())

from src.rag.pipeline import RAGPipeline

try:
    pipeline = RAGPipeline()
    gen = pipeline.stream_query("What is LoRA?", top_k=2)
    for x in gen:
        print(x)
except Exception as e:
    print(f"Error: {e}")
