import json
from config.settings import CHUNKS_DIR

fixed_files = 0
fixed_chunks = 0

for f in CHUNKS_DIR.glob("*_semantic.json"):
    with open(f, "r", encoding = "utf-8") as fp:
        chunks = json.load(fp)
    
    changed = False
    for chunk in chunks:
        if not chunk.get("primary_category"):
            # Derive from paper_id if needed - use cs.LG as safe default
            chunk["primary_category"] = "cs.LG"
            fixed_chunks += 1
            changed = True
    
    if changed:
        with open(f, "w", encoding="utf-8") as fp:
            json.dump(chunks, fp, indent = 2, ensure_ascii = False)
        fixed_files += 1

print(f"Fixed {fixed_chunks} chunks across {fixed_files} files")