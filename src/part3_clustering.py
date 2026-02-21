# src/part3_clustering.py
# Part 3: Topic Clustering + 2-Level Topic Tree
#
# Loads embeddings saved by part2_embeddings.py (outputs/part2/train_embeddings.npy)
# Step A -- KMeans top-level clustering, K chosen via elbow (K < 10)
#           LLM labels each cluster from representative docs
# Step B -- Re-cluster 2 largest clusters into 3 sub-clusters each
#           LLM labels each sub-cluster
# Step C -- Print the 2-level topic tree
#
# Set your API key before running:
#   Windows:  set ANTHROPIC_API_KEY=sk-ant-...
#   Mac/Linux: export ANTHROPIC_API_KEY=sk-ant-...

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import anthropic

from data import load_20newsgroups_sample


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_representative_docs(cluster_embs, cluster_docs, centroid, n=5):
    """Return n docs closest to the cluster centroid."""
    dists = np.linalg.norm(cluster_embs - centroid, axis=1)
    top_idx = np.argsort(dists)[:n]
    return [cluster_docs[i][:500] for i in top_idx]


def llm_label(rep_docs, client):
    """Ask Claude for a 3-6 word topic label based on representative docs."""
    sample = "\n---\n".join(rep_docs)
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=60,
        messages=[{
            "role": "user",
            "content": (
                "Below are up to 5 short excerpts from documents in one text cluster.\n"
                "Give ONE concise topic label (3-6 words) describing the cluster theme.\n"
                "Reply with ONLY the label, nothing else.\n\n"
                f"Documents:\n{sample}"
            )
        }]
    )
    return msg.content[0].text.strip()


def run_part3():
    out_dir = "outputs/part3"
    ensure_dir(out_dir)

    # 1) Load the same training texts as Part 1 & 2
    X_train, _, y_train, _, target_names = load_20newsgroups_sample(
        n_samples=10000, test_size=0.2, random_state=42
    )
    docs = list(X_train)

    # 2) Load embeddings saved by Part 2
    emb_path = "outputs/part2/train_embeddings.npy"
    if not os.path.exists(emb_path):
        print("ERROR: Run part2_embeddings.py first to generate embeddings.")
        return
    X_emb = np.load(emb_path)
    print(f"Loaded embeddings: {X_emb.shape}")

    # 3) Anthropic client for LLM labels
    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from environment

    # ------------------------------------------------------------------ #
    # STEP A -- Elbow method to pick K, then top-level KMeans             #
    # ------------------------------------------------------------------ #
    print("\nStep A: Elbow method (K = 2 to 9) ...")
    k_range  = range(2, 10)
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_emb)
        inertias.append(km.inertia_)
        print(f"  K={k}  inertia={km.inertia_:.1f}")

    # Save elbow plot
    plt.figure(figsize=(7, 4))
    plt.plot(list(k_range), inertias, "bo-", linewidth=2, markersize=7)
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.xticks(list(k_range))
    plt.tight_layout()
    elbow_path = os.path.join(out_dir, "elbow_curve.png")
    plt.savefig(elbow_path, dpi=150)
    plt.close()
    print(f"Elbow curve saved -> {elbow_path}")

    # Pick K (adjust this if your elbow looks different)
    K_TOP = 7
    print(f"\nUsing K={K_TOP} for top-level clustering ...")
    km_top     = KMeans(n_clusters=K_TOP, random_state=42, n_init=10)
    top_labels = km_top.fit_predict(X_emb)

    cluster_sizes      = {c: int((top_labels == c).sum()) for c in range(K_TOP)}
    top_cluster_labels = {}

    print("\nGenerating LLM labels for top-level clusters ...")
    for c in range(K_TOP):
        mask         = top_labels == c
        cluster_docs = [docs[i] for i in range(len(docs)) if mask[i]]
        cluster_embs = X_emb[mask]
        centroid     = km_top.cluster_centers_[c]
        rep_docs     = get_representative_docs(cluster_embs, cluster_docs, centroid)
        label        = llm_label(rep_docs, client)
        top_cluster_labels[c] = label
        print(f"  Cluster {c} (n={cluster_sizes[c]:4d}): {label}")

    # ------------------------------------------------------------------ #
    # STEP B -- Sub-cluster the 2 largest clusters into 3 each           #
    # ------------------------------------------------------------------ #
    two_largest = sorted(cluster_sizes, key=cluster_sizes.get, reverse=True)[:2]
    print(f"\nStep B: Sub-clustering 2 largest clusters: {two_largest}")

    sub_labels_map = {}

    for big_c in two_largest:
        mask       = top_labels == big_c
        sub_embs   = X_emb[mask]
        sub_docs   = [docs[i] for i in range(len(docs)) if mask[i]]
        print(f"\n  Parent Cluster {big_c}: '{top_cluster_labels[big_c]}'  "
              f"(n={cluster_sizes[big_c]})")

        km_sub          = KMeans(n_clusters=3, random_state=42, n_init=10)
        sub_cluster_ids = km_sub.fit_predict(sub_embs)
        sub_labels_map[big_c] = {}

        for sc in range(3):
            sc_mask  = sub_cluster_ids == sc
            sc_embs  = sub_embs[sc_mask]
            sc_docs  = [sub_docs[i] for i in range(len(sub_docs)) if sc_mask[i]]
            centroid = km_sub.cluster_centers_[sc]
            rep_docs = get_representative_docs(sc_embs, sc_docs, centroid)
            label    = llm_label(rep_docs, client)
            sub_labels_map[big_c][sc] = label
            print(f"    Sub {sc} (n={int(sc_mask.sum()):4d}): {label}")

    # ------------------------------------------------------------------ #
    # STEP C -- Print 2-level topic tree                                  #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 55)
    print("          2-LEVEL TOPIC TREE")
    print("=" * 55)

    for c in range(K_TOP):
        marker = "[*]" if c in two_largest else "[ ]"
        print(f"\n{marker} Cluster {c}  [{cluster_sizes[c]} docs]")
        print(f"      {top_cluster_labels[c]}")
        if c in sub_labels_map:
            for sc, slabel in sub_labels_map[c].items():
                print(f"      |-- Sub {sc}: {slabel}")

    print("\n[*] = clusters expanded with sub-topics")
    print("=" * 55)

    # 4) Save tree to JSON
    tree = []
    for c in range(K_TOP):
        tree.append({
            "cluster_id":   c,
            "label":        top_cluster_labels[c],
            "size":         cluster_sizes[c],
            "sub_clusters": [
                {"sub_id": sc, "label": lbl}
                for sc, lbl in sub_labels_map[c].items()
            ] if c in sub_labels_map else [],
        })

    out_path = os.path.join(out_dir, "topic_tree.json")
    with open(out_path, "w") as f:
        json.dump(tree, f, indent=2)
    print(f"\nTopic tree saved -> {out_path}")


if __name__ == "__main__":
    run_part3()
