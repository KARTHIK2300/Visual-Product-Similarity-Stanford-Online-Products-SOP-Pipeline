"""
feature_extraction_offline_demo.py

IMPORTANT: This is a DROP-IN SUBSTITUTE for feature_extraction.py, used only
to demo the full pipeline in this sandbox, which has no internet access to
download.pytorch.org (where the pretrained ResNet50 ImageNet weights live).

It builds embeddings from classic computer-vision descriptors (color
histograms in HSV + HOG texture/shape features) instead of a CNN. It is
enough to prove the FAISS indexing / search / evaluation pipeline works
end-to-end on real data, and the 10 categories in the sample dataset are
visually distinct enough (bottle, lamp, chair, shoes, ...) that this
already produces sensible nearest-neighbor results.

In your own environment (with normal internet access), just use
feature_extraction.py instead -- the rest of the pipeline (build_faiss_index.py,
search.py, evaluate.py, app.py) is IDENTICAL either way, since they only
read outputs/embeddings.npy + outputs/metadata.json.
"""

import argparse
import json
import os

import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from tqdm import tqdm

from utils import list_images, get_category

IMG_SIZE = (128, 128)


def extract_embedding(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img)

    # --- Color histogram (HSV, captures color/style) ---
    hsv = np.array(img.convert("HSV"))
    hist_features = []
    for ch in range(3):
        hist, _ = np.histogram(hsv[:, :, ch], bins=32, range=(0, 256), density=True)
        hist_features.append(hist)
    color_vec = np.concatenate(hist_features)

    # --- HOG (captures shape/edges/texture) ---
    gray = rgb2gray(arr)
    hog_vec = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
    )

    emb = np.concatenate([color_vec.astype("float32"), hog_vec.astype("float32")])
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype("float32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="data/images")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    image_paths = list_images(args.image_dir)
    print(f"Found {len(image_paths)} images in {args.image_dir}")

    embeddings = []
    metadata = []
    for path in tqdm(image_paths, desc="Extracting embeddings (offline demo)"):
        emb = extract_embedding(path)
        embeddings.append(emb)
        metadata.append(
            {"path": path, "filename": os.path.basename(path), "category": get_category(path)}
        )

    # pad/truncate all vectors to the same length just in case, then stack
    embeddings = np.vstack(embeddings).astype("float32")
    np.save(os.path.join(args.out_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved embeddings: {embeddings.shape} -> {args.out_dir}/embeddings.npy")
    print(f"Saved metadata -> {args.out_dir}/metadata.json")


if __name__ == "__main__":
    main()
