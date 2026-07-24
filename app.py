import streamlit as st

from src.classifier import load_artifacts, classify_query, get_redis_connection


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_resource
def get_cached_redis_connection():
    return get_redis_connection()


model, vectorizer, known_sites = get_artifacts()
redis_conn = get_cached_redis_connection()

st.title("Search Query Intent Classifier")
st.write(
    "Type a search query to see whether it needs a full AI-generated answer, "
    "or can go straight to a result."
)

query = st.text_input("Enter a search query")

if st.button("Classify"):
    if query.strip():
        result = classify_query(query, model, vectorizer, known_sites, redis_conn=redis_conn)

        st.header(result["predicted"])
        st.write(f"Confidence: {result['confidence'] * 100:.1f}%")
        st.caption(f"Source: {result['source']}")

        if result.get("cached"):
            st.caption("⚡ instant result (cached)")

        if result["predicted"] == "Navigational":
            st.success("This would skip the AI path and go straight to a result.")
        else:
            st.info("This would go through the full AI response.")
    else:
        st.warning("Please enter a query first.")
