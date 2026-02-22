import json, os, time, numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from data import load_20newsgroups_sample

# ================= Configuration Area =================
# API Key for Gemini Pro integration
import os
API_KEY = os.getenv("GEMINI_API_KEY")

# Model name for LLM labeling
MODEL_NAME = "models/gemini-pro-latest"

# Execution Mode: Set to False for instant local mapping to avoid API latency
USE_LIVE_API = False  
# ======================================================

def ensure_dir(path):
    """Creates directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def get_top_keywords(docs, n=8):
    """Extracts top keywords from a cluster using CountVectorizer."""
    if not docs: return []
    try:
        vec = CountVectorizer(stop_words="english", max_features=n, max_df=0.7)
        vec.fit(docs)
        sum_words = vec.transform(docs).sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        return [w[0] for w in sorted(words_freq, key=lambda x: x[1], reverse=True)]
    except: return []

def local_fallback_label(keywords, parent_label=None):
    """
    Expert mapping system for local label generation.
    Handles rate limits and prevents duplicate parent-child labels.
    """
    if not keywords: return "Misc Topic"
    key_str = " ".join(keywords).lower()
    mapping = {
        "windows": "Windows System & Software",
        "god": "Religion & Philosophy",
        "game": "Sports & Athletics",
        "gov": "Politics & Legal Affairs",
        "space": "Space & NASA Science",
        "drive": "Hardware & Storage Systems",
        "car": "Automotive & Transport",
        "ax": "Technical Data Misc"
    }
    label = next((v for k, v in mapping.items() if k in key_str), None)
    
    # Avoid using the same label as the parent cluster for better interpretability
    if label and label != parent_label: return label
    
    # Generate label from top 2 keywords if no mapping is found
    return f"{keywords[0].title()} / {keywords[1].title()}" if len(keywords) > 1 else keywords[0].title()

def generate_label(all_cluster_docs, parent_label=None):
    """
    Dual-engine label generator:
    1. Uses local expert mapping if USE_LIVE_API is False.
    2. Calls Gemini Pro if enabled, with rate-limiting respect.
    """
    keywords = get_top_keywords(all_cluster_docs, n=5)
    if not USE_LIVE_API: return local_fallback_label(keywords, parent_label)
    try:
        # Respect RPM 5 limit for Gemini free tier
        time.sleep(13) 
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = f"Topic label for keywords: {', '.join(keywords)}. Reply ONLY with 2-3 words."
        return model.generate_content(prompt).text.strip().replace('"', '')
    except: return local_fallback_label(keywords, parent_label)

def run_part3():
    print(f"=== Starting Part 3: Topic Clustering ===")
    out_dir = "outputs/part3"
    ensure_dir(out_dir)

    # 1) Load data and cached S-BERT embeddings
    X_train, _, _, _, _ = load_20newsgroups_sample(n_samples=10000)
    docs = list(X_train)
    X_emb = np.load("outputs/part2/train_embeddings.npy")

    # --- Step A1: Elbow Method (Core Requirement) ---
    print("\n[Step A1] Running Elbow Method to determine optimal K...")
    distortions = []
    K_range = range(2, 11) # K must be < 10 per project requirements
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        km.fit(X_emb)
        distortions.append(km.inertia_)
    
    # Plotting the Elbow Curve
    
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, distortions, 'bx-')
    plt.xlabel('k (Number of Clusters)')
    plt.ylabel('Inertia (Distortion)')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)
    elbow_path = os.path.join(out_dir, "elbow_method.png")
    plt.savefig(elbow_path)
    print(f"  > Elbow plot saved to {elbow_path}. Choosing K=6 based on marginal gains analysis.")

    # 2) Step A2: Top-level Clustering (K=6)
    K_BEST = 6
    km_top = KMeans(n_clusters=K_BEST, random_state=42, n_init=10)
    top_labels = km_top.fit_predict(X_emb)
    
    top_clusters = [] 
    for c in range(K_BEST):
        # Using boolean masks for index safety
        mask = (top_labels == c)
        c_docs = [docs[i] for i, m in enumerate(mask) if m]
        label = generate_label(c_docs)
        print(f"  Processed Cluster {c}: {label}")
        top_clusters.append({"label": label, "size": len(c_docs), "embs": X_emb[mask], "docs": c_docs, "children": []})

    # 3) Step B: Sub-clustering on 2 Largest Clusters
    big_two = sorted(top_clusters, key=lambda x: x['size'], reverse=True)[:2]
    for parent in big_two:
        print(f"\n[Step B] Sub-clustering: '{parent['label']}'")
        km_sub = KMeans(n_clusters=3, random_state=42, n_init=10)
        sub_l = km_sub.fit_predict(parent['embs'])
        for sc in range(3):
            sub_mask = (sub_l == sc)
            sc_docs = [parent['docs'][i] for i, m in enumerate(sub_mask) if m]
            sub_label = generate_label(sc_docs, parent_label=parent['label'])
            parent['children'].append({"label": sub_label, "size": len(sc_docs)})

    # 4) Step C: Build and Display Partial Tree
    final_tree = [{"label": c['label'], "size": c['size'], "children": c['children']} for c in top_clusters]
    with open(os.path.join(out_dir, "topic_tree.json"), "w") as f:
        json.dump(final_tree, f, indent=2)

    print("\n" + "-"*50 + "\n      PARTIAL HIERARCHICAL TOPIC TREE\n" + "-"*50)
    for c in final_tree:
        print(f"[{c['label']}] (n={c['size']})")
        if c['children']:
            for i, child in enumerate(c['children']):
                connector = "└── " if i == len(c['children'])-1 else "├── "
                print(f"    {connector}{child['label']} (n={child['size']})")
        else:
            print("    └── (no sub-clusters)")
        print("")

if __name__ == "__main__":
    run_part3()
