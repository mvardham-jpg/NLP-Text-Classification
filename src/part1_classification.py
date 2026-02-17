# src/part1_classification.py
# Part 1 (TF-IDF): 4 classifiers + accuracy/macro-F1 + top confusion pairs
# Saves results to outputs/part1/results.json

import json
import os

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from data import load_20newsgroups_sample
from eval import compute_metrics, top_confusions


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_part1():
    # 1) Load data
    X_train, X_test, y_train, y_test, target_names = load_20newsgroups_sample(
        n_samples=10000, test_size=0.2, random_state=42
    )

    # 2) TF-IDF vectorizer (classic sparse features)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)

    # 3) Required models
    models = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "LinearSVM": LinearSVC(),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
    }

    results = []

    # 4) Train + evaluate each model using a leakage-safe pipeline
    for model_name, model in models.items():
        print(f"\n=== TF-IDF + {model_name} ===")

        pipe = Pipeline([
            ("vectorizer", vectorizer),
            ("model", model),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)

        # Top confusion pairs (instead of full confusion matrix)
        conf_pairs = top_confusions(y_test, y_pred, k=10)
        conf_readable = []
        for count, i, j in conf_pairs:
            conf_readable.append({
                "count": int(count),
                "true_label": target_names[i],
                "pred_label": target_names[j],
            })

        print("Accuracy:", round(metrics["accuracy"], 4))
        print("Macro-F1:", round(metrics["macro_f1"], 4))
        print("Top confusion pairs (true -> pred):")
        for item in conf_readable[:5]:
            print(f"  {item['count']} : {item['true_label']} -> {item['pred_label']}")

        results.append({
            "vectorizer": "tfidf",
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "top_confusions": conf_readable,
        })

    # 5) Save results
    out_dir = "outputs/part1"
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "results.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # 6) Print best model (by Macro-F1)
    best = max(results, key=lambda r: r["macro_f1"])
    print("\n✅ BEST (by Macro-F1):")
    print(
        f"TF-IDF + {best['model']} | "
        f"Macro-F1={best['macro_f1']:.4f} | Acc={best['accuracy']:.4f}"
    )
    print("Saved:", out_path)


if __name__ == "__main__":
    run_part1()

