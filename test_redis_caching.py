from src.classifier import load_artifacts, get_redis_connection, classify_query

model, vectorizer, known_sites = load_artifacts()
redis_conn = get_redis_connection()

query = "youtube"

redis_conn.delete(query)
print(f"Deleted any existing '{query}' key from Redis.")

first = classify_query(query, model, vectorizer, known_sites, redis_conn=redis_conn)
print("First call: ", first)

second = classify_query(query, model, vectorizer, known_sites, redis_conn=redis_conn)
print("Second call:", second)

passed = (
    first["cached"] is False
    and second["cached"] is True
    and first["predicted"] == second["predicted"]
)

print(f"Test passed: {passed}")
