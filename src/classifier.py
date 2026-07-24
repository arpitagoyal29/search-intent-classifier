import joblib
from rapidfuzz import process, fuzz


def load_artifacts(models_dir="models"):
    model = joblib.load(f"{models_dir}/intent_model.pkl")
    vectorizer = joblib.load(f"{models_dir}/tfidf_vectorizer.pkl")
    known_sites = joblib.load(f"{models_dir}/known_sites.pkl")
    return model, vectorizer, known_sites


def check_known_site(query, known_sites, threshold=80):
    match = process.extractOne(query, known_sites, scorer=fuzz.ratio)

    if match is None:
        return False, None, 0

    matched_to, score, _ = match

    if score >= threshold:
        return True, matched_to, score

    return False, None, score


def classify_query(query, model, vectorizer, known_sites, threshold=80, ml_confidence_cutoff=0.90):
    query = query.lower().strip()

    is_known, matched_to, score = check_known_site(query, known_sites, threshold)

    if is_known:
        return {
            "query": query,
            "predicted": "Navigational",
            "source": "lookup",
            "matched_to": matched_to,
            "confidence": score / 100,
        }

    features = vectorizer.transform([query])
    probabilities = model.predict_proba(features)[0]

    class_index = probabilities.argmax()
    predicted = model.classes_[class_index]
    confidence = probabilities[class_index]

    if predicted == "Navigational" and confidence < ml_confidence_cutoff:
        predicted = "Informational"

    return {
        "query": query,
        "predicted": predicted,
        "source": "model",
        "matched_to": None,
        "confidence": confidence,
    }
