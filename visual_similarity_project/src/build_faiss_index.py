"""
build_faiss_index.py
Step 2 of the pipeline: Embedding Indexing

Loads outputs/embeddings.npy and builds a FAISS index for millisecond-level
nearest-neighbor search. Since embeddings are L2-normalized, we use an
Inner Product index (IndexFlatIP) -- inner product of normalized vectors
is equivalent to cosine similarity.

Run:
    python src/build_faiss_index.py --out_dir outputs
"""

import argparse
import os

import faiss
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    emb_path = os.path.join(args.out_dir, "embeddings.npy")
    embeddings = np.load(emb_path).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized dot product
    index.add(embeddings)

    index_path = os.path.join(args.out_dir, "faiss.index")
    faiss.write_index(index, index_path)

    print(f"Indexed {index.ntotal} vectors of dim {dim}")
    print(f"Saved FAISS index -> {index_path}")


if __name__ == "__main__":
    main()
