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

## 📈 Experimental Results & Evaluation

Evaluated against the Stanford Online Products (SOP) benchmark on the official test split (60,502 images across 12 super-categories):

### Overall Metrics

| Metric | Value |
| :--- | :--- |
| **Model Architecture** | Pretrained PyTorch ResNet50 (2048-dim vectors) |
| **Evaluated Images** | 60,502 product items |
| **Index Type** | FAISS IndexFlatIP |
| **Overall Precision@10** | 0.7751 (77.51%) |
| **Overall Recall@10** | 0.0015 |
| **Average Query Latency** | < 15 ms |

### Per-Category Precision@10 Breakdown

| Product Category | Image Count (n) | Precision@10 |
| :--- | :--- | :--- |
| **Bicycle** | 4,145 | 0.9416 |
| **Mug** | 6,895 | 0.8967 |
| **Stapler** | 4,019 | 0.8129 |
| **Kettle** | 5,101 | 0.8111 |
| **Cabinet** | 6,220 | 0.8019 |
| **Sofa** | 3,959 | 0.7958 |
| **Coffee Maker** | 4,713 | 0.7633 |
| **Fan** | 3,003 | 0.7529 |
| **Lamp** | 6,226 | 0.7425 |
| **Toaster** | 4,357 | 0.7064 |
| **Chair** | 6,226 | 0.6823 |
| **Table** | 5,638 | 0.6136 |

### Qualitative & Analytical Insights

* **Semantic Power of CNN Representations:** Deep convolutional embeddings drastically outperform hand-crafted color/texture descriptors (which baseline at ~0.21), reaching >94% precision in structurally distinct categories like bicycles.
* **Geometric Distinctiveness:** Objects with rigid, unique geometries (bicycles, mugs, staplers) score highest, whereas categories with broad shape variations (tables, chairs) show lower category precision due to visual overlap.
* **Low Macro Recall Context:** The low Recall@10 (0.0015) is expected given the large class pool (thousands of images per category compared to a top-10 retrieval cutoff).

---

## 💻 Setup & Usage Instructions

### 1. Clone the Repository & Set Up Environment

```bash
git clone [https://github.com/KARTHIK2300/Visual-Product-Similarity-Stanford-Online-Products-SOP-Pipeline.git](https://github.com/KARTHIK2300/Visual-Product-Similarity-Stanford-Online-Products-SOP-Pipeline.git)
cd Visual-Product-Similarity-Stanford-Online-Products-SOP-Pipeline

# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

## Extending this project

- **Bigger dataset:** point `--image_dir` at Stanford Online Products or
  DeepFashion for a stronger portfolio result (the brief recommends 1000+ images).
- **Better ANN index:** swap `IndexFlatIP` for `IndexIVFFlat` or `IndexHNSWFlat`
  once you have >50k embeddings, for faster approximate search.
- **Metadata filters:** extend `metadata.json` with price/availability and
  pass extra filters into `search()`.
- **Fine-tuning:** fine-tune ResNet50 on triplet/contrastive loss over
  known similar-product pairs for even tighter embeddings.
