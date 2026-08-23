"""
evaluate.py
Step 5 of the pipeline: Evaluation

Since we don't have human-labeled "similar pairs", we use each product's
category as a proxy ground truth: a retrieved item is treated as "relevant"
if it shares the query's category (same style of product). This is the
standard approach used for datasets like Stanford Online Products / DeepFashion.

For every image in the dataset (used as a query, excluding itself from its
own results), we compute:
  - Precision@K = (# relevant items in top-K) / K
  - Recall@K    = (# relevant items in top-K) / (total relevant items in the dataset)
Averaged (macro) over all queries and reported overall + per category.

Run:
    python src/evaluate.py --k 5
"""

import argparse
import json
import os
from collections import defaultdict

import faiss
import numpy as np


def evaluate(out_dir: str, k: int = 5):
    embeddings = np.load(os.path.join(out_dir, "embeddings.npy")).astype("float32")
    with open(os.path.join(out_dir, "metadata.json")) as f:
        metadata = json.load(f)

    index = faiss.read_index(os.path.join(out_dir, "faiss.index"))

    categories = [m["category"] for m in metadata]
    category_counts = defaultdict(int)
    for c in categories:
        category_counts[c] += 1

    # search top-(k+1) since the query image itself will be the #1 hit (score=1.0)
    scores, indices = index.search(embeddings, k + 1)

    precisions = []
    recalls = []
    per_cat_precisions = defaultdict(list)

    for query_idx in range(len(metadata)):
        query_cat = categories[query_idx]
        total_relevant = category_counts[query_cat] - 1  # excluding the query itself

        neighbors = [idx for idx in indices[query_idx] if idx != query_idx][:k]
        relevant_hits = sum(1 for idx in neighbors if categories[idx] == query_cat)

        precision = relevant_hits / k
        recall = relevant_hits / total_relevant if total_relevant > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        per_cat_precisions[query_cat].append(precision)

    print(f"\n=== Evaluation @K={k} (category-based relevance) ===")
    print(f"Overall Precision@{k}: {np.mean(precisions):.4f}")
    print(f"Overall Recall@{k}:    {np.mean(recalls):.4f}\n")

    print("Per-category Precision@K:")
    for cat in sorted(per_cat_precisions):
        vals = per_cat_precisions[cat]
        print(f"  {cat:12s} n={len(vals):3d}  Precision@{k}={np.mean(vals):.4f}")

    return {
        "k": k,
        "overall_precision_at_k": float(np.mean(precisions)),
        "overall_recall_at_k": float(np.mean(recalls)),
        "per_category_precision_at_k": {
            cat: float(np.mean(vals)) for cat, vals in per_cat_precisions.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    results = evaluate(args.out_dir, args.k)

    with open(os.path.join(args.out_dir, f"evaluation_k{args.k}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics -> {args.out_dir}/evaluation_k{args.k}.json")


if __name__ == "__main__":
    main()
