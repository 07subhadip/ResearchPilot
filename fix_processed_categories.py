import json
from config.settings import PROCESSED_DIR

fixed = 0

for f in PROCESSED_DIR.glob("*.json"):
    with open(f, "r", encoding = 'utf-8') as fp:
        data = json.load(fp)

    if not data.get("primary_category"):
        cats = data.get("categories", [])
        data["primary_category"] = cats[0] if cats else "cs.LG"
        with open(f, "w", encoding = "utf-8") as fp:
            json.dump(data, fp, indent = 2, ensure_ascii = False)
        fixed += 1

print(f"Fixed {fixed} processed files")