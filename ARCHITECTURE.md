# Project Architecture

## 1. System Overview
This project implements a three-tier NLP pipeline designed for both supervised text classification and unsupervised hierarchical clustering.

## 2. Module Responsibilities
* **`data.py`**: Handles data ingestion, 80/20 stratified splitting, and text cleaning (removal of headers, footers, and quotes).
* **`eval.py`**: Provides centralized evaluation logic, calculating Accuracy, Macro-F1, and identifying top confusion pairs.
* **`part1_classification.py`**: Responsible for the baseline TF-IDF vectorization and training four supervised classifiers (NB, Logistic Regression, SVM, Random Forest).
* **`part2_embeddings.py`**: Manages neural feature extraction using `all-MiniLM-L6-v2` and caches 384-dimensional dense vectors to disk for efficiency.
* **`part3_clustering.py`**: Executes the unsupervised discovery phase, including Elbow Method analysis, K-Means clustering, and hierarchical tree generation.

## 3. Data Flow
1. **Raw Data**: Scikit-learn 20 Newsgroups dataset is loaded and cleaned.
2. **Feature Transformation**:
    * **Path A**: Text is converted to sparse matrices via TF-IDF.
    * **Path B**: Text is encoded into dense semantic embeddings via S-BERT.
3. **Supervised Learning**: Features are fed into classifiers; results are stored in `outputs/` as JSON.
4. **Unsupervised Discovery**: S-BERT embeddings are analyzed by the Clustering Engine to produce a 2-level hierarchical topic tree.

## 4. Clustering Strategy (Part 3)
* **Optimal K Discovery**: Uses the **Elbow Method** to balance cluster granularity and interpretability.
* **Hierarchy Generation**: Recursive K-Means on the two largest clusters to find sub-topics.
* **Labeling Engine**: A hybrid system utilizing **Gemini Pro (LLM)** for semantic summarization with a **Local Expert Mapping** fallback for API efficiency.