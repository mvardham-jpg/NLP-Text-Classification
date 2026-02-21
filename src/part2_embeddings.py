# src/part2_embeddings.py
# Part 2: SentenceTransformer embeddings + 4 classifiers
# Mirrors part1_classification.py exactly -- same structure, same eval helpers
# Saves results to outputs/part2/results.json
# Saves embeddings to outputs/part2/*.npy for Part 3 to reuse

import json
import os
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.naive_bayes import GaussianNB        # MNB fails on negative floats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from data import load_20newsgroups_sample
from eval import compute_metrics, top_confusions


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_part2():
    # 1) Load data -- identical call to Part 1
    X_train, X_test, y_train, y_test, target_names = load_20newsgroups_sample(
        n_samples=10000, test_size=0.2, random_state=42
    )

    # 2) Embed documents with SentenceTransformer
    #    Cache to disk so re-runs are instant and Part 3 can load them
    out_dir = "outputs/part2"
    ensure_dir(out_dir)

    train_cache = os.path.join(out_dir, "train_embeddings.npy")
    test_cache  = os.path.join(out_dir, "test_embeddings.npy")

    model_st = SentenceTransformer("all-MiniLM-L6-v2")

    if os.path.exists(train_cache):
        print("Loading cached train embeddings ...")
        X_train_emb = np.load(train_cache)
    else:
        print("Encoding train set (this takes a few minutes) ...")
        X_train_emb = model_st.encode(list(X_train), batch_size=64,
                                      show_progress_bar=True, convert_to_numpy=True)
        np.save(train_cache, X_train_emb)

    if os.path.exists(test_cache):
        print("Loading cached test embeddings ...")
        X_test_emb = np.load(test_cache)
    else:
        print("Encoding test set ...")
        X_test_emb = model_st.encode(list(X_test), batch_size=64,
                                     show_progress_bar=True, convert_to_numpy=True)
        np.save(test_cache, X_test_emb)

    print(f"Embedding shape: {X_train_emb.shape}")

    # 3) Same 4 models as Part 1
    #    GaussianNB replaces MultinomialNB -- MNB requires non-negative inputs,
    #    but embeddings contain negative floats so MNB throws a ValueError.
    models = {
        "GaussianNB":          GaussianNB(),
        "LogisticRegression":  LogisticRegression(max_iter=2000),
        "LinearSVM":           LinearSVC(),
        "RandomForest":        RandomForestClassifier(n_estimators=300,
                                                      random_state=42, n_jobs=-1),
    }

    results = []

    # 4) Train + evaluate -- no Pipeline needed, embeddings are already computed
    for model_name, model in models.items():
        print(f"\n=== SentenceTransformer + {model_name} ===")

        model.fit(X_train_emb, y_train)
        y_pred = model.predict(X_test_emb)

        metrics = compute_metrics(y_test, y_pred)

        conf_pairs = top_confusions(y_test, y_pred, k=10)
        conf_readable = []
        for count, i, j in conf_pairs:
            conf_readable.append({
                "count":      int(count),
                "true_label": target_names[i],
                "pred_label": target_names[j],
            })

        print("Accuracy:", round(metrics["accuracy"], 4))
        print("Macro-F1:", round(metrics["macro_f1"], 4))
        print("Top confusion pairs (true -> pred):")
        for item in conf_readable[:5]:
            print(f"  {item['count']} : {item['true_label']} -> {item['pred_label']}")

        results.append({
            "vectorizer":     "sentence-transformer (all-MiniLM-L6-v2)",
            "model":          model_name,
            "accuracy":       metrics["accuracy"],
            "macro_f1":       metrics["macro_f1"],
            "top_confusions": conf_readable,
        })

    # 5) Save results
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # 6) Best model
    best = max(results, key=lambda r: r["macro_f1"])
    print("\nBEST (by Macro-F1):")
    print(
        f"SentenceTransformer + {best['model']} | "
        f"Macro-F1={best['macro_f1']:.4f} | Acc={best['accuracy']:.4f}"
    )
    print("Saved:", out_path)


if __name__ == "__main__":
    run_part2()
