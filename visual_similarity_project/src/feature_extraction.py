"""
feature_extraction.py
Step 1 of the pipeline: Image Feature Extraction

Uses a pretrained ResNet50 (ImageNet weights) with the final classification
layer removed, so each image is converted into a 2048-dim dense embedding
vector that captures its visual content (shape, color, texture, style).

Run:
    python src/feature_extraction.py --image_dir data/images --out_dir outputs

Note: the first run downloads the pretrained ImageNet weights (~100MB),
so it needs internet access. This is a one-time download; torch caches it
under ~/.cache/torch afterwards.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

from utils import list_images, get_category


def build_model(device: str) -> nn.Module:
    """Load pretrained ResNet50 and strip the final FC (classification) layer."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    resnet = models.resnet50(weights=weights)
    # Remove the last fully-connected layer -> output is the 2048-d pooled
    # feature vector instead of 1000-class logits.
    modules = list(resnet.children())[:-1]
    model = nn.Sequential(*modules)
    model.eval()
    model.to(device)
    return model


def get_transform() -> transforms.Compose:
    """Standard ImageNet preprocessing pipeline expected by ResNet50."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


@torch.no_grad()
def extract_embedding(model: nn.Module, transform, image_path: str, device: str) -> np.ndarray:
    """Extract a single 2048-d L2-normalized embedding for one image."""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    features = model(tensor)  # shape: (1, 2048, 1, 1)
    features = features.squeeze().cpu().numpy().astype("float32")
    # L2-normalize so that cosine similarity == dot product later
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="data/images")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = build_model(device)
    transform = get_transform()

    image_paths = list_images(args.image_dir)
    print(f"Found {len(image_paths)} images in {args.image_dir}")

    embeddings = []
    metadata = []
    for path in tqdm(image_paths, desc="Extracting embeddings"):
        emb = extract_embedding(model, transform, path, device)
        embeddings.append(emb)
        metadata.append(
            {"path": path, "filename": os.path.basename(path), "category": get_category(path)}
        )

    embeddings = np.vstack(embeddings).astype("float32")
    np.save(os.path.join(args.out_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved embeddings: {embeddings.shape} -> {args.out_dir}/embeddings.npy")
    print(f"Saved metadata -> {args.out_dir}/metadata.json")


if __name__ == "__main__":
    main()
