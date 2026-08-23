# Visual Product Similarity & Image-Based Recommendation System (Amazon-Style)

An end-to-end computer-vision pipeline that recommends visually similar
products from a product photo — the way Amazon's "Similar Items" works —
using deep-learning image embeddings, FAISS approximate nearest-neighbor
search, and a Streamlit UI.

## What's inside

```
visual_similarity_project/
├── data/images/                    500 sample product photos (10 categories)
├── src/
│   ├── utils.py                    helpers (list images, parse category)
│   ├── feature_extraction.py       Step 1: ResNet50 CNN embeddings (production)
│   ├── feature_extraction_offline_demo.py   offline fallback extractor (see note below)
│   ├── build_faiss_index.py        Step 2: build FAISS ANN index
│   ├── search.py                   Step 3+4: query + rank/filter top-K
│   └── evaluate.py                 Step 5: Precision@K / Recall@K
├── app.py                          Streamlit web app
├── outputs/                        generated embeddings, index, metrics
└── requirements.txt
```

## ⚠️ Important note on the CNN model

`feature_extraction.py` uses a **pretrained ResNet50** (ImageNet weights,
final classification layer removed) exactly as specified in the project
brief. The first time you run it, PyTorch downloads the pretrained weights
from `download.pytorch.org` (~100MB) — **you need normal internet access
for this step.**

This sandbox environment I built the project in has a locked-down network
that can't reach `download.pytorch.org`, so I couldn't download the real
ImageNet weights here. To still prove the full pipeline works end-to-end
on your actual data, I built `feature_extraction_offline_demo.py` — a
drop-in substitute that uses classical CV descriptors (HSV color
histograms + HOG shape/texture features) instead of a CNN. Everything
downstream (FAISS indexing, search, ranking, evaluation, the Streamlit
app) is **identical** either way — they only read `embeddings.npy` +
`metadata.json`, so swapping extractors is a one-line change.

**When you run this on your own machine, just use `feature_extraction.py`**
— you'll get much stronger results, since a CNN captures far richer visual
semantics than color/edge histograms.

Results below (`Precision@5 ≈ 0.21` overall) were produced with the
offline demo extractor as a sanity check. Expect **noticeably higher
precision (commonly 0.7–0.95 on clean, well-separated categories like
this sample set) once you switch to the ResNet50 embeddings.**

## Setup

```bash
pip install -r requirements.txt
```

## Step-by-step usage

### Step 1 — Extract image embeddings
```bash
cd src
python feature_extraction.py --image_dir ../data/images --out_dir ../outputs
```
This loads pretrained ResNet50, strips the final FC layer, and converts
every image into a 2048-dim L2-normalized embedding vector. Saves
`outputs/embeddings.npy` and `outputs/metadata.json`.

*(No internet? Use `python feature_extraction_offline_demo.py --image_dir ../data/images --out_dir ../outputs` instead — same output format.)*

### Step 2 — Build the FAISS index
```bash
python build_faiss_index.py --out_dir ../outputs
```
Builds an `IndexFlatIP` (inner product) index. Since embeddings are
L2-normalized, inner product = cosine similarity. Saves `outputs/faiss.index`.

### Step 3 & 4 — Search, rank, and filter
```bash
python search.py --query ../data/images/bottle_0_1771054550088.jpg --k 5 --extractor cnn
```
Extracts the query embedding, retrieves the Top-K by cosine similarity,
and (optionally) filters by category:
```bash
python search.py --query <path> --k 5 --category_filter shoes --extractor cnn
```

### Step 5 — Evaluate
```bash
python evaluate.py --k 5 --out_dir ../outputs
```
Since there's no hand-labeled "similar pairs" ground truth, each product's
**category** is used as a relevance proxy (a retrieved item is "correct"
if it's the same category as the query) — the standard approach for
datasets like Stanford Online Products. Reports overall and per-category
Precision@K / Recall@K, saved to `outputs/evaluation_k{K}.json`.

### Run the web app
```bash
streamlit run app.py
```
Upload a photo (or pick a sample from the catalog), tune Top-K and an
optional category filter in the sidebar, and see visually similar
products ranked by similarity score.

To switch the app from the offline demo extractor to the real CNN, open
`app.py` and change:
```python
EXTRACTOR = "offline_demo"   # -> "cnn"
```

## Actual results from this sample dataset (offline demo extractor)

500 images across 10 categories (backpack, bottle, chair, headphones,
lamp, laptop, shoes, sunglasses, tshirt, watch), 50 each.

| Metric | Value |
|---|---|
| Overall Precision@5 | 0.214 |
| Overall Recall@5 | 0.022 |

Per-category Precision@5 ranged from 0.14 (laptop) to 0.37 (shoes) —
expected for hand-crafted color/shape features. Swapping in ResNet50
embeddings should push most categories well above 0.7.

## Extending this project

- **Bigger dataset:** point `--image_dir` at Stanford Online Products or
  DeepFashion for a stronger portfolio result (the brief recommends 1000+ images).
- **Better ANN index:** swap `IndexFlatIP` for `IndexIVFFlat` or `IndexHNSWFlat`
  once you have >50k embeddings, for faster approximate search.
- **Metadata filters:** extend `metadata.json` with price/availability and
  pass extra filters into `search()`.
- **Fine-tuning:** fine-tune ResNet50 on triplet/contrastive loss over
  known similar-product pairs for even tighter embeddings.
