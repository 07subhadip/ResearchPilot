"""
Verify chunk quality across the full dataset.
Run this before embedding to catch any data issues early.
"""


import json
from pathlib import Path
from config.settings import CHUNKS_DIR
from src.utils.logger import setup_logger, get_logger


setup_logger()
logger = get_logger(__name__)



def main():
    chunk_files = list(CHUNKS_DIR.glob("*_semantic.json"))
    logger.info(f"Checking {len(chunk_files)} chunk files...")


    total_chunks    = 0
    total_words     = 0
    tiny_chunks     = 0     # < 50 words
    giant_chunks    = 0     # > 600 words
    clean_endings   = 0
    sample_chunks   = []    # Store a few for display


    for cf in chunk_files:
        with open(cf, encoding = 'utf-8') as f:
            chunks = json.load(f)

        
        for c in chunks:
            total_chunks += 1
            wc = c["word_count"]
            total_words += wc


            if wc < 50:
                tiny_chunks += 1
            if wc > 600:
                giant_chunks += 1
            if c["text"].rstrip().endswith(('.', '!', '?')):
                clean_endings += 1


            if len(sample_chunks) < 3:
                sample_chunks.append(c)


    avg_words = total_words / total_chunks if total_chunks else 0


    print(f"\n{'='*55}")
    print(f"  CHUNK QUALITY REPORT")
    print(f"{'='*55}")
    print(f"  Total chunk files:     {len(chunk_files)}")
    print(f"  Total chunks:          {total_chunks:,}")
    print(f"  Avg words per chunk:   {avg_words:.0f}")
    print(f"  Tiny chunks (<50w):    {tiny_chunks} ({100*tiny_chunks/total_chunks:.1f}%)")
    print(f"  Giant chunks (>600w):  {giant_chunks} ({100*giant_chunks/total_chunks:.1f}%)")
    print(f"  Clean endings:         {clean_endings} ({100*clean_endings/total_chunks:.1f}%)")
    print()


    print("  SAMPLE CHUNKS:")
    print(f"  {'-'*50}")
    for i, c in enumerate(sample_chunks):
        print(f"  [{i+1}] Paper: {c['paper_id']}")
        print(f"       Words: {c['word_count']} | Strategy: {c['chunking_strategy']}")
        print(f"       Text: {c['text'][:120].replace(chr(10), ' ')}...")
        print()

    # Quality gates - these thresholds indicate healthy chunking
    print(f"{'='*55}")
    print(f"  QUALITY GATES")
    print(f"{'='*55}")


    gates = [
        ("Total chunks > 10,000",         total_chunks > 10_000),
        ("Avg words 100-400",             100 <= avg_words <= 400),
        ("Tiny chunks < 10%",             tiny_chunks/total_chunks < 0.10),
        ("Clean endings > 70%",           clean_endings/total_chunks > 0.70),
    ]


    all_pass = True
    for name, passed in gates:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print()

    if all_pass:
        print("  ✅ All quality gates passed. Ready for Phase 6.")
    else:
        print("  ⚠️  Some gates failed. Review before proceeding.")

if __name__ == "__main__":
    main()