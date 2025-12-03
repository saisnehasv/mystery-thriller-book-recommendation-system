# Thrills & Mysteries — A Personalized Book Recommender System

### STATS 507 — Final Project
**Author:** Sai Sneha Siddapura Venkataramappa  
**Uniqname:** saisneha  

---

## Project Overview

**Thrills & Mysteries** is a content-driven **personalized book recommendation system** tailored for the **Mystery & Thriller** genre.  

The project demonstrates a **comprehensive data science workflow**, encompassing:

- Exploratory Data Analysis (EDA) and visualization  
- Natural Language Processing and textual embeddings  
- Dimensionality reduction and clustering  
- Hybrid content-based recommendation scoring  
- Semantic recommendation explanation and interpretability  
- An interactive **Gradio interface** for local exploration  

This project emphasizes reproducibility, interpretability, and a step-by-step demonstration of **data-driven recommendation techniques**.

---

## Workflow Highlights

### 1. Exploratory Data Analysis (EDA)
- Comprehensive **data cleaning and preprocessing** to ensure high-quality inputs  
- Visualization of **distributions, top genres, and word frequencies**  
- **Word clouds, n-gram analysis, and TF-IDF** for textual insights  
- Quantitative and qualitative exploration of dataset characteristics  

### 2. Embedding Models Comparison
- Explored **multiple sentence-transformer embeddings** to assess which best captures semantic meaning:
  - `all-MiniLM-L6-v2`  
  - `all-mpnet-base-v2`  
- Evaluated embeddings based on **clustering behavior, similarity retrieval, and recommendation quality**  
- Provides insights into **how embedding choice impacts downstream recommendation results**  

### 3. Dimensionality Reduction & Clustering Experiments
- Applied **PCA** to understand variance and information distribution in embeddings  
- Compared **t-SNE vs UMAP** for 2D projection and visual exploration of book clusters  
- Explored **KMeans vs DBSCAN** to identify latent groupings or sub-genres  
- These experiments highlight **the impact of dimensionality reduction and clustering choices** on interpreting the dataset and informing recommendation strategies

### 4. Recommendation System
- Implemented a **content-based recommendation approach** using **K-Nearest Neighbors (KNN)** on sentence embeddings  
- Incorporated a **hybrid scoring mechanism** combining:
  - **Cosine similarity** of embeddings (semantic relevance)  
  - **Jaccard similarity** of genre vectors (genre alignment)  
- Weighted by a hyperparameter **α**, allowing tuning of the balance between semantic and genre-based similarity  
- Recommendations include **ranked tables** displaying:
  - Book title & author  
  - Rating  
  - Description snippet  
  - Genre information  
- Designed to be **interpretable and explainable**, with optional **TF-IDF keywords** highlighting why a book was recommended  
- Emphasizes **flexibility**, allowing different similarity metrics and weighting schemes to be experimented with  

### 5. Evaluation
- Recommendations are evaluated using **semantic relevance metrics**. A book is considered relevant if its **cosine similarity** to the query exceeds a threshold  
- Metrics computed include:
  - **Precision@K** – fraction of recommended books that are relevant  
  - **Recall@K** – fraction of relevant books retrieved  
  - **MAP@K** – mean average precision across recommendations  
  - **NDCG** – normalized discounted cumulative gain, accounting for rank of relevant books  
  - **HitRate** – whether at least one relevant book is retrieved  
  - **MRR** – mean reciprocal rank of the first relevant book  
- Evaluations are performed across multiple thresholds to gauge robustness  

### 6. Interpretation of Recommendations
- Each recommendation can be **explained using hybrid scores**:
  - **Cosine similarity** captures semantic closeness between book descriptions  
  - **Genre similarity** uses Jaccard index to measure overlap in genre tags  
- Optional **TF-IDF keyword analysis** highlights the most relevant terms driving similarity for each recommended book  
- Supports **visual interpretability** via **UMAP projections**, showing:
  - The query book  
  - Its top-N recommendations  
  - Clustering and relative distances among books  

### 7. Gradio Interface 
- Built an interactive **Gradio interface** to explore recommendations locally  
- Users can:
  - Input a book title  
  - View **top-N recommendations**
  - See **cover images, descriptions, ratings, genres, and hybrid scores**  
- **Note:** The interface is **not deployed online**. To access it, one must run the notebook locally and execute the Gradio cell  

---

## Project Structure

```
Thrills-and-Mysteries/
│
├── notebooks/
│   └── Thrills_and_Mysteries.ipynb    # Main Jupyter notebook with full workflow
│
├── styles/
│   └── style.css                       # Custom CSS for Gradio interface
│
├── graphs/
│   └── ...                             # All visual outputs: distributions, word clouds,
│                                       # PCA, t-SNE, UMAP, clustering plots
│
├── embeddings/
│   ├── all-MiniLM-L6-v2/
│   │   └── train_embeddings.npy       # Embeddings for training subset
│   └── all-mpnet-base-v2/
│       ├── train_embeddings.npy
│       ├── test_embeddings.npy
│       └── full_embeddings.npy
│
├── Documentation/
│   ├── Project Proposal
│   └── Final Project Report
│
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---

## Running the Project

To run the project locally:

### 1. Clone the repository
```bash
git clone <repository-url>
cd Thrills-and-Mysteries
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Jupyter Notebook
```bash
jupyter notebook notebooks/Thrills_and_Mysteries.ipynb
```

### 4. Explore the notebook
- Run the cells sequentially to reproduce all analysis, experiments, and the Gradio interface
- The Gradio interface is **local only**; it launches in a browser tab after executing the respective notebook cell

---

## Key Dependencies

- `pandas` — Data manipulation and analysis
- `numpy` — Numerical computing
- `scikit-learn` — Machine learning algorithms (KNN, clustering, metrics)
- `sentence-transformers` — Pre-trained embedding models
- `nltk` — Natural language processing
- `matplotlib`, `seaborn` — Data visualization
- `plotly` — Interactive visualizations
- `umap-learn` — Dimensionality reduction
- `gradio` — Interactive interface
- `Pillow` — Image processing

See `requirements.txt` for the complete list with version specifications.

---

## Key Features

- **Comprehensive EDA** — Deep dive into the Mystery & Thriller genre dataset  
- **Multiple Embedding Models** — Comparison of semantic representations  
- **Dimensionality Reduction** — PCA, t-SNE, and UMAP for visualization  
- **Clustering Analysis** — KMeans and DBSCAN for pattern discovery  
- **Hybrid Recommendations** — Semantic + genre-based similarity scoring  
- **Robust Evaluation** — Multiple metrics (Precision, Recall, MAP, NDCG, MRR)  
- **Interpretable Results** — TF-IDF keywords and visual explanations  
- **Interactive Interface** — Gradio UI for exploring recommendations  

---

## Results & Insights

The project reveals:
- How **embedding choice** significantly impacts recommendation quality
- The **trade-offs** between different dimensionality reduction techniques
- The **effectiveness of hybrid scoring** in balancing semantic and categorical similarity
- **Interpretable recommendations** that can explain why books are suggested

For detailed results, refer to the **Final Project Report** in the `Documentation/` folder.

---

## License

This project is for educational purposes as part of DATASCI 507 : Data Analytics using Python coursework.

---

## Contact

**Sai Sneha Siddapura Venkataramappa**  
Uniqname: saisneha

For questions or feedback, please reach out via the course communication channels.

---

## Acknowledgments

- Dataset: https://huggingface.co/datasets/euclaise/goodreads_100k
- Pre-trained models: [Sentence Transformers library](https://huggingface.co/sentence-transformers)
- Course: DATASCI 507 : Data Analytics using Python, University of Michigan