import os
import json

import joblib
import psycopg2
import redis
from dotenv import load_dotenv
from rapidfuzz import process, fuzz


def load_artifacts(models_dir="models"):
    model = joblib.load(f"{models_dir}/intent_model.pkl")
    vectorizer = joblib.load(f"{models_dir}/tfidf_vectorizer.pkl")
    known_sites = joblib.load(f"{models_dir}/known_sites.pkl")
    return model, vectorizer, known_sites


def get_redis_connection():
    load_dotenv()
    return redis.Redis(
        host=os.environ["REDIS_HOST"],
        port=int(os.environ["REDIS_PORT"]),
        username=os.environ["REDIS_USERNAME"],
        password=os.environ["REDIS_PASSWORD"],
        ssl=False,
    )


def get_db_connection():
    load_dotenv()
    return psycopg2.connect(os.environ["DATABASE_URL"])


def init_db(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS query_log (
                id SERIAL PRIMARY KEY,
                query TEXT,
                predicted TEXT,
                source TEXT,
                confidence FLOAT,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """
        )
    conn.commit()


def check_known_site(query, known_sites, threshold=80):
    match = process.extractOne(query, known_sites, scorer=fuzz.ratio)

    if match is None:
        return False, None, 0

    matched_to, score, _ = match

    if score >= threshold:
        return True, matched_to, score

    return False, None, score


def classify_query(query, model, vectorizer, known_sites, redis_conn=None, db_conn=None, threshold=80, ml_confidence_cutoff=0.90):
    query = query.lower().strip()

    result = None

    if redis_conn is not None:
        cached = redis_conn.get(query)
        if cached is not None:
            result = json.loads(cached)
            result["cached"] = True

    if result is None:
        is_known, matched_to, score = check_known_site(query, known_sites, threshold)

        if is_known:
            result = {
                "query": query,
                "predicted": "Navigational",
                "source": "lookup",
                "matched_to": matched_to,
                "confidence": score / 100,
            }
        else:
            features = vectorizer.transform([query])
            probabilities = model.predict_proba(features)[0]

            class_index = probabilities.argmax()
            predicted = model.classes_[class_index]
            confidence = float(probabilities[class_index])

            if predicted == "Navigational" and confidence < ml_confidence_cutoff:
                predicted = "Informational"

            result = {
                "query": query,
                "predicted": predicted,
                "source": "model",
                "matched_to": None,
                "confidence": confidence,
            }

        if redis_conn is not None:
            redis_conn.set(query, json.dumps(result))
            result["cached"] = False

    if db_conn is not None:
        try:
            with db_conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO query_log (query, predicted, source, confidence) VALUES (%s, %s, %s, %s)",
                    (result["query"], result["predicted"], result["source"], result["confidence"]),
                )
            db_conn.commit()
        except Exception as e:
            print(f"Warning: failed to log query to database: {e}")

    return result
