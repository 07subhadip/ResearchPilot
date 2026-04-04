# Run this in a SEPARATE terminal while run_api.py is running
import requests
import json

BASE_URL = "http://localhost:8000"

# Test 1: Health check
print("Testing /health...")
r = requests.get(f"{BASE_URL}/health")
print(json.dumps(r.json(), indent=2))

# Test 2: Query
print("\nTesting /query...")
payload = {
    "question": "What is LoRA and how does it work?",
    "top_k": 5
}
r = requests.post(f"{BASE_URL}/query", json=payload)
data = r.json()

print(f"Answer: {data['answer'][:300]}...")
print(f"\nCitations: {len(data['citations'])}")
for c in data['citations']:
    print(f"  - {c['paper_id']}: {c['title'][:50]}...")
print(f"\nTotal time: {data['total_time_ms']:.0f}ms")

# Test 3: Filtered query
print("\nTesting /query with filter...")
payload = {
    "question": "graph neural network applications",
    "top_k": 3,
    "filter_year_gte": 2026
}
r = requests.post(f"{BASE_URL}/query", json=payload)
data = r.json()
print(f"Answer: {data['answer'][:200]}...")
print(f"Citations: {len(data['citations'])}")