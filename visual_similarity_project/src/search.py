"""
search.py
Step 3 of the pipeline: Similarity Matching

Given a query image, extract its embedding and retrieve the Top-K most
visually similar products from the FAISS index (cosine similarity).

Run:
    python src/search.py --query data/images/bottle_0_....jpg --k 5

Swap EXTRACT_FN below between the CNN extractor (feature_extraction.py) and
the offline demo extractor depending on which one built your index/embeddings.
"""

import argparse
import json
import os

import faiss
import numpy as np


def load_index_and_metadata(out_dir: str):
    index = faiss.read_index(os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "metadata.json")) as f:
        metadata = json.load(f)
    return index, metadata


def search(query_embedding: np.ndarray, index, metadata, k: int = 5, category_filter: str = None):
    """
    Returns Top-K results as a list of dicts: {rank, filename, category, path, score}
    Ranked & filtered per Step 4 (rank by score; optional category filter).
    """
    query_embedding = query_embedding.reshape(1, -1).astype("float32")
    # Over-fetch if filtering, then trim, so filtering doesn't starve results
    fetch_k = k * 5 if category_filter else k
    fetch_k = min(fetch_k, index.ntotal)

    scores, indices = index.search(query_embedding, fetch_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = metadata[idx]
        if category_filter and item["category"] != category_filter:
            continue
        results.append(
            {
                "filename": item["filename"],
                "category": item["category"],
                "path": item["path"],
                "score": float(score),
            }
        )
        if len(results) >= k:
            break

    for rank, r in enumerate(results, start=1):
        r["rank"] = rank
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Path to query image")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--category_filter", default=None)
    parser.add_argument(
        "--extractor",
        choices=["cnn", "offline_demo"],
        default="offline_demo",
        help="Which feature extractor matches the built index",
    )
    args = parser.parse_args()

    if args.extractor == "cnn":
        from feature_extraction import build_model, get_transform, extract_embedding
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model(device)
        transform = get_transform()
        query_emb = extract_embedding(model, transform, args.query, device)
    else:
        from feature_extraction_offline_demo import extract_embedding

        query_emb = extract_embedding(args.query)

    index, metadata = load_index_and_metadata(args.out_dir)
    results = search(query_emb, index, metadata, k=args.k, category_filter=args.category_filter)

    print(f"\nTop-{args.k} visually similar products for: {os.path.basename(args.query)}\n")
    for r in results:
        print(f"  #{r['rank']}  {r['filename']:35s}  category={r['category']:12s}  score={r['score']:.4f}")


if __name__ == "__main__":
    main()
