# NLP Text Classification & Hierarchical Topic Clustering

## Overview

This project builds an end-to-end NLP pipeline on the **20 Newsgroups dataset** (10,000 documents) for multi-class text classification.

The project consists of three parts:

* Part 1: Classic TF-IDF Classification (Completed)
* Part 2: SentenceTransformer Embeddings
* Part 3: Topic Clustering + 2-Level Topic Tree

This README documents progress through **Part 1**.

---

# Dataset

We use the scikit-learn **20 Newsgroups** dataset.

* 20 topic categories
* 10,000 sampled documents
* 80/20 train-test split (stratified)
* Headers, footers, and quotes removed

Dataset loader:

```python
fetch_20newsgroups
```

---

# Part 1 — TF-IDF Classification

## Approach

We build a supervised multi-class classifier using:

* **TF-IDF vectorization**
* scikit-learn **Pipeline** (prevents data leakage)
* 4 required classifiers:

  1. Multinomial Naive Bayes
  2. Logistic Regression
  3. Linear SVM
  4. Random Forest

Evaluation metrics:

* Accuracy
* Macro-F1
* Top confusion pairs (instead of full confusion matrix)

---

# Results (TF-IDF)

| Model                   | Accuracy   | Macro-F1   |
| ----------------------- | ---------- | ---------- |
| Multinomial Naive Bayes | 0.7065     | 0.6762     |
| Logistic Regression     | 0.7165     | 0.7004     |
| **Linear SVM (Best)**   | **0.7245** | **0.7138** |
| Random Forest           | 0.6395     | 0.6222     |

## Best Model

**TF-IDF + Linear SVM**

* Macro-F1: **0.7138**
* Accuracy: **0.7245**

Linear SVM achieved the best overall performance.

---

# Confusion Analysis

Most frequent misclassifications occurred between semantically similar categories:

* Religion-related topics

  * `talk.religion.misc` ↔ `soc.religion.christian`
  * `alt.atheism` → `soc.religion.christian`

* Politics

  * `talk.politics.misc` ↔ `talk.politics.guns`

* Computer-related categories

  * `comp.sys.ibm.pc.hardware` ↔ `comp.os.ms-windows.misc`
  * `comp.windows.x` ↔ `comp.os.ms-windows.misc`

These confusions are expected due to overlapping vocabulary.

---
#  Part 2 — SentenceTransformer Embeddings

**Model:** `all-MiniLM-L6-v2` - each document encoded as a 384-dimensional dense vector  
**Classifiers:** Same 4 as Part 1  
**Note on MNB:** `MultinomialNB` requires non-negative inputs (counts/frequencies).  
SentenceTransformer embeddings contain negative floats, so `GaussianNB` is used instead.  
This is documented in results and does not affect the comparison.

Embeddings are cached to `outputs/part2/*.npy` after the first run - re-runs are instant.

### Part 1 vs Part 2 Comparison

| Approach | Best Model | Macro-F1 |
|---|---|---|
| TF-IDF (Part 1) | LinearSVM | ~0.71 |
| SentenceTransformer (Part 2) | LinearSVM | ~0.70-0.75 |

**Why TF-IDF competes well on 20 Newsgroups:**  
This dataset has strong keyword signals — words like `nasa`, `gun`, `god`, `windows` are highly discriminative per category. TF-IDF captures these directly.

**Why embeddings are generally more powerful:**  
Embeddings capture *meaning*, not just word identity. They handle synonyms, paraphrasing, and context. On real-world noisy or short-text tasks, embeddings consistently outperform TF-IDF.

---


# Project Structure

```
NLP-Text-Classification/
│
├── src/
│   ├── data.py
│   ├── eval.py
│   ├── part1_classification.py
│   ├── part2_embeddings.py
│   └── part3_clustering.py
│
├── outputs/
│   ├── part1/results.json        
│   ├── part2/results.json       
│   ├── part2/train_embeddings.npy
│   ├── part2/test_embeddings.npy
│   └── part3/
│
├── data/
│   └── sklearn_cache/20news-bydate_py3.pkz            # 20 Newsgroups cached here after first download
├── requirements.txt
└── README.md
```

---

# Installation Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd NLP-Text-Classification
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If requirements.txt is missing, install manually:

```bash
pip install numpy pandas scikit-learn matplotlib tqdm sentence-transformers
```

---

# How to Run 

From the **project root directory**:

```bash
python src/part1_classification.py
```

What this script does:

1. Loads 10,000 documents from 20 Newsgroups
2. Splits into 80/20 train-test sets
3. Builds a TF-IDF pipeline
4. Trains all 4 classifiers
5. Prints Accuracy and Macro-F1
6. Displays top confusion pairs
7. Saves results to:

```
outputs/part1/results.json
```

---

```bash
python src/part2_embeddings.py

What this script does:
1. Load data same as Part 1
2. Encode documents into numbers via SentenceTransformer (cached to disk)
3. Train 4 classifiers on those 384-number vectors 
4. Evaluate compute_metrics() and top_confusions() from eval.py 
5. Save results to: 
```

```
outputs/part2/results.json
```

---

# Reproducibility

* Random seed fixed at 42
* Stratified train-test split
* TF-IDF limited to 50,000 max features
* English stopwords removed

---

