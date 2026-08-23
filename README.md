# Visual Product Similarity & Image-Based Recommendation System (Colab Edition)

An end-to-end computer vision retrieval system that recommends visually similar products from query images, mirroring Amazon's "Similar Items" feature[cite: 2]. This version is optimized for execution within Google Colab using a free GPU instance.

---

## 📈 Experimental Results & Evaluation

Evaluated against the Stanford Online Products (SOP) benchmark on the official test split (60,502 images across 12 super-categories):

### Overall Metrics

| Metric | Value |
| :--- | :--- |
| **Model Architecture** | Pretrained PyTorch ResNet50 (2048-dim vectors) |
| **Evaluated Images** | 60,502 product items |
| **Index Type** | FAISS IndexFlatIP[cite: 3] |
| **Overall Precision@10** | 0.7751 (77.51%)[cite: 3] |
| **Overall Recall@10** | 0.0015[cite: 3] |
| **Average Query Latency** | < 15 ms[cite: 3] |

### Per-Category Precision@10 Breakdown

| Product Category | Image Count (n) | Precision@10 |
| :--- | :--- | :--- |
| **Bicycle** | 4,145 | 0.9416[cite: 3] |
| **Mug** | 6,895 | 0.8967[cite: 3] |
| **Stapler** | 4,019 | 0.8129[cite: 3] |
| **Kettle** | 5,101 | 0.8111[cite: 3] |
| **Cabinet** | 6,220 | 0.8019[cite: 3] |
| **Sofa** | 3,959 | 0.7958[cite: 3] |
| **Coffee Maker** | 4,713 | 0.7633[cite: 3] |
| **Fan** | 3,003 | 0.7529[cite: 3] |
| **Lamp** | 6,226 | 0.7425[cite: 3] |
| **Toaster** | 4,357 | 0.7064[cite: 3] |
| **Chair** | 6,226 | 0.6823[cite: 3] |
| **Table** | 5,638 | 0.6136[cite: 3] |

### Qualitative & Analytical Insights

* **Semantic Power of CNN Representations:** Deep convolutional embeddings drastically outperform hand-crafted color/texture descriptors (which baseline at ~0.21), reaching >94% precision in structurally distinct categories like bicycles[cite: 3].
* **Geometric Distinctiveness:** Objects with rigid, unique geometries (bicycles, mugs, staplers) score highest, whereas categories with broad shape variations (tables, chairs) show lower category precision due to visual overlap[cite: 3].
* **Low Macro Recall Context:** The low Recall@10 (0.0015) is expected given the large class pool (thousands of images per category compared to a top-10 retrieval cutoff)[cite: 3].

---

## 💻 Google Colab Setup & Usage Instructions

Since this pipeline requires heavy image processing, running it in Google Colab with a GPU is highly recommended to demonstrate a scalable, production-like recommendation pipeline[cite: 2].

### 1. Environment Preparation
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the `SOP_pipeline_colab.ipynb` notebook file into your Colab workspace.
3. **Enable GPU:** Navigate to `Runtime` > `Change runtime type` > Select `GPU` (e.g., T4 GPU) > Click `Save`.

### 2. Run the Full Pipeline from the Notebook
Execute the cells in the notebook sequentially to perform the following actions:

* **Step 1: Install Dependencies & Download Data:** The initial cells will install `faiss-cpu`, `streamlit`, and download the Stanford Online Products Dataset directly to the Colab environment.
* **Step 2: Parse Dataset Metadata:** Converts the native `.txt` dataset files into the required `catalog_metadata.json` format.
* **Step 3: Extract ResNet50 Embeddings:** Uses the GPU to process the dataset in batches, saving the extracted 2048-dimensional vectors to `outputs/embeddings.npy`[cite: 2, 3]. 
* **Step 4: Build the FAISS Vector Index:** Constructs the vector database and saves it to `outputs/faiss.index`[cite: 2, 3].
* **Step 5: Run Quantitative Evaluation:** Calculates Precision@K and Recall@K against the dataset[cite: 2, 3].

### 3. Launch the Streamlit Web Application via LocalTunnel
Because Colab virtual machines do not expose local ports to the web, you must use LocalTunnel to view the UI.

1. Ensure the `app.py` Streamlit code is written to the Colab filesystem (using the `%%writefile app.py` magic command).
2. Run the following command in a notebook cell to get your Colab instance's public IP address (you will need this for the LocalTunnel password):
   ```python
   import urllib
   print("Endpoint IP:", urllib.request.urlopen('[https://ipv4.icanhazip.com](https://ipv4.icanhazip.com)').read().decode('utf8').strip("\n"))
