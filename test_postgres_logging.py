from src.classifier import load_artifacts, get_db_connection, init_db, classify_query

model, vectorizer, known_sites = load_artifacts()
db_conn = get_db_connection()
init_db(db_conn)

queries = ["youtube", "how does compound interest work", "gogle"]

for query in queries:
    result = classify_query(query, model, vectorizer, known_sites, db_conn=db_conn)
    print(f"Classified: {result}")

with db_conn.cursor() as cur:
    cur.execute(
        "SELECT query, predicted, source, confidence, created_at FROM query_log ORDER BY id DESC LIMIT 3"
    )
    rows = cur.fetchall()

    cur.execute("SELECT COUNT(*) FROM query_log")
    total_rows = cur.fetchone()[0]

print("\nLast 3 rows in query_log:")
for row in rows:
    print(row)

print(f"\nTotal rows in query_log: {total_rows}")
