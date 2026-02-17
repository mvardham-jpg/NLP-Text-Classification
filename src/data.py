# src/data.py

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


def load_20newsgroups_sample(
    n_samples=10000,
    test_size=0.2,
    random_state=42
):
    """
    Loads 20 Newsgroups, samples n_samples documents,
    and returns a stratified train/test split.
    """

    # Load dataset
    dataset = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes"),
    data_home="data/sklearn_cache"
)

    X = np.array(dataset.data)
    y = np.array(dataset.target)
    target_names = dataset.target_names

    # Randomly sample n_samples
    rng = np.random.RandomState(random_state)
    indices = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)

    X_small = X[indices]
    y_small = y[indices]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_small,
        y_small,
        test_size=test_size,
        stratify=y_small,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test, target_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, target_names = load_20newsgroups_sample()

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    print("Number of classes:", len(set(y_train)))
    print("\nExample document snippet:\n")

    for doc in X_train:
        if len(doc.strip()) > 100:
            print(doc[:500])
            break

