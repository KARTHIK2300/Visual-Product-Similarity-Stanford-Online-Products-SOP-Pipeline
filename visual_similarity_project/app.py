"""
app.py
Streamlit frontend for the Visual Product Similarity & Image-Based
Recommendation System.

Lets a user upload a product photo (or pick a sample) and see the Top-K
most visually similar products retrieved via FAISS + cosine similarity,
with an optional category filter.

Run:
    streamlit run app.py
"""

import json
import os
import sys

import faiss
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

OUT_DIR = "outputs"
DATA_DIR = "data/images"

# Which extractor to use. "offline_demo" works with no internet (classic CV
# features). Switch to "cnn" once you've run feature_extraction.py in an
# environment with internet access to download the pretrained ResNet50 weights.
EXTRACTOR = "offline_demo"


@st.cache_resource
def load_index_and_metadata():
    index = faiss.read_index(os.path.join(OUT_DIR, "faiss.index"))
    with open(os.path.join(OUT_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    return index, metadata


@st.cache_resource
def load_extractor():
    if EXTRACTOR == "cnn":
        from feature_extraction import build_model, get_transform
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model(device)
        transform = get_transform()
        return ("cnn", model, transform, device)
    else:
        from feature_extraction_offline_demo import extract_embedding

        return ("offline_demo", extract_embedding, None, None)


def embed_image(image_path, extractor_bundle):
    kind, a, b, c = extractor_bundle
    if kind == "cnn":
        from feature_extraction import extract_embedding

        return extract_embedding(a, b, image_path, c)
    else:
        return a(image_path)


def run_search(query_embedding, index, metadata, k, category_filter):
    from search import search

    return search(query_embedding, index, metadata, k=k, category_filter=category_filter)


def main():
    st.set_page_config(page_title="Visual Product Similarity", layout="wide")
    st.title("🛍️ Visual Product Similarity & Recommendation System")
    st.caption(
        "Upload a product photo to find visually similar items — powered by "
        "deep visual embeddings + FAISS approximate nearest-neighbor search."
    )

    index, metadata = load_index_and_metadata()
    extractor_bundle = load_extractor()
    categories = sorted(set(m["category"] for m in metadata))

    with st.sidebar:
        st.header("Search settings")
        k = st.slider("Number of results (Top-K)", min_value=1, max_value=20, value=5)
        category_filter = st.selectbox("Filter by category (optional)", ["All"] + categories)
        category_filter = None if category_filter == "All" else category_filter
        st.markdown("---")
        st.write(f"Index size: **{index.ntotal}** products")
        st.write(f"Categories: {', '.join(categories)}")

    tab1, tab2 = st.tabs(["📤 Upload your own image", "🖼️ Try a sample product"])

    query_path = None

    with tab1:
        uploaded = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            os.makedirs("tmp_uploads", exist_ok=True)
            query_path = os.path.join("tmp_uploads", uploaded.name)
            with open(query_path, "wb") as f:
                f.write(uploaded.getbuffer())

    with tab2:
        sample_choice = st.selectbox(
            "Pick a sample image from the catalog", [m["filename"] for m in metadata]
        )
        if st.button("Use this sample as query"):
            query_path = next(m["path"] for m in metadata if m["filename"] == sample_choice)

    if query_path:
        col_q, col_r = st.columns([1, 3])
        with col_q:
            st.subheader("Query image")
            st.image(Image.open(query_path), use_container_width=True)

        with st.spinner("Extracting embedding and searching..."):
            query_emb = embed_image(query_path, extractor_bundle)
            results = run_search(query_emb, index, metadata, k, category_filter)

        with col_r:
            st.subheader(f"Top-{len(results)} visually similar products")
            if not results:
                st.info("No results found for this category filter.")
            cols = st.columns(min(5, len(results)) or 1)
            for i, r in enumerate(results):
                with cols[i % len(cols)]:
                    st.image(Image.open(r["path"]), use_container_width=True)
                    st.caption(
                        f"#{r['rank']} · {r['category']}\nscore: {r['score']:.3f}"
                    )
    else:
        st.info("Upload an image or pick a sample product to see similar recommendations.")


if __name__ == "__main__":
    main()
