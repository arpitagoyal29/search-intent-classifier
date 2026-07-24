from src.classifier import load_artifacts, classify_query

model, vectorizer, known_sites = load_artifacts()

queries = [
    "youtube",
    "goggle",
    "flipkart",
    "gmial",
    "how does compound interest work",
    "best budget phones 2026",
    "myntra",
]

for query in queries:
    result = classify_query(query, model, vectorizer, known_sites)
    print(
        f"query={result['query']!r} "
        f"predicted={result['predicted']} "
        f"source={result['source']} "
        f"confidence={result['confidence']:.2f}"
    )
