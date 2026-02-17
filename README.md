
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
* scikit-learn **Pipeline** (to prevent data leakage)
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

Linear SVM performed best overall, achieving the highest macro-F1 score and accuracy.

---

# Confusion Analysis

The most common classification confusions occurred between semantically similar categories:

* Religion-related topics

  * `talk.religion.misc` ↔ `soc.religion.christian`
  * `alt.atheism` → `soc.religion.christian`

* Politics

  * `talk.politics.misc` ↔ `talk.politics.guns`

* Computer-related categories

  * `comp.sys.ibm.pc.hardware` ↔ `comp.os.ms-windows.misc`
  * `comp.windows.x` ↔ `comp.os.ms-windows.misc`

These confusions are expected due to overlapping vocabulary within related domains.

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
│   └── part1/results.json
│
├── requirements.txt
└── README.md
```

---

# How to Run Part 1

From the project root:

```bash
python src/part1_classification.py
```

Results will be saved to:

```
outputs/part1/results.json
```

---

# Technical Notes

* TF-IDF features limited to 50,000 max features
* English stopwords removed
* Pipelines used to prevent data leakage
* Fixed random seed for reproducibility




