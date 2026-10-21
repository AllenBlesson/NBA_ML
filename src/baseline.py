
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
)

# File locations and settings
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "cleaned_nba.csv"
RESULTS = ROOT / "results"

FEATURES = [
    "usg_percent",
    "ts_percent",
    "x3p_ar",
    "ast_percent",
    "trb_percent",
]
SEED = 0
K_VALUES = range(2, 9)

# 1. Load the data and use the existing splits
df = pd.read_csv(DATA_PATH)

train = df[df["split"] == "train"]
val = df[df["split"] == "val"]
test = df[df["split"] == "test"]

# 2. Scale features using training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(train[FEATURES])
X_val = scaler.transform(val[FEATURES])
X_test = scaler.transform(test[FEATURES])

# 3. Choose k using validation silhouette score
k_scores = []
best_k = None
best_score = -1

for k in K_VALUES:
    model = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    model.fit(X_train)

    labels = model.predict(X_val)
    score = silhouette_score(X_val, labels)

    k_scores.append({
        "k": k,
        "silhouette_val": round(score, 4),
    })

    if score > best_score:
        best_k = k
        best_score = score

# Calculate the evaluation metrics for a model
def evaluate(name, k, val_labels, test_labels):
    return {
        "model": name,
        "k": k,
        "silhouette_val": round(
            silhouette_score(X_val, val_labels), 4
        ),
        "silhouette_test": round(
            silhouette_score(X_test, test_labels), 4
        ),
        "davies_bouldin_test": round(
            davies_bouldin_score(X_test, test_labels), 4
        ),
    }

# 4. Naive baseline: assign random cluster labels
rng = np.random.RandomState(SEED)
random_val = rng.randint(0, best_k, size=len(val))
random_test = rng.randint(0, best_k, size=len(test))

naive = evaluate(
    "Naive: random labels", best_k, random_val, random_test
)

# 5. Reference baseline: use traditional player positions
encoder = LabelEncoder()
encoder.fit(df["pos"])

position_val = encoder.transform(val["pos"])
position_test = encoder.transform(test["pos"])

positions = evaluate(
    "Reference: position labels",
    len(encoder.classes_),
    position_val,
    position_test,
)

# 6. Course-level baseline: K-Means
kmeans = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
kmeans.fit(X_train)

kmeans_val = kmeans.predict(X_val)
kmeans_test = kmeans.predict(X_test)

kmeans_result = evaluate(
    "Course-level baseline: k-means",
    best_k,
    kmeans_val,
    kmeans_test,
)

# 7. Candidate model: Gaussian Mixture Model
gmm = GaussianMixture(n_components=best_k, random_state=SEED)
gmm.fit(X_train)

gmm_val = gmm.predict(X_val)
gmm_test = gmm.predict(X_test)

gmm_result = evaluate(
    "Candidate: Gaussian mixture",
    best_k,
    gmm_val,
    gmm_test,
)

# Compare the two models' test-set cluster assignments
ari = round(adjusted_rand_score(kmeans_test, gmm_test), 4)

# Stability 
kmeans_runs = [
    KMeans(n_clusters=best_k, random_state=seed, n_init=10).fit(X_train).predict(X_test)
    for seed in range(10)
]
gmm_runs = [
    GaussianMixture(n_components=best_k, random_state=seed).fit(X_train).predict(X_test)
    for seed in range(10)
]

# Mean ARI over every pair of runs
def mean_ari(runs):
    scores = [adjusted_rand_score(first, second) for first, second in combinations(runs, 2)]
    return round(float(np.mean(scores)), 4)

stability = {"kmeans": mean_ari(kmeans_runs), "gmm": mean_ari(gmm_runs)}
print("Mean ARI across 10 seeds:", stability)

# 8. Save the results
rows = [naive, positions, kmeans_result, gmm_result]
RESULTS.mkdir(parents=True, exist_ok=True)

with open(RESULTS / "baseline_table.md", "w") as file:
    file.write("| model | k | silhouette_val | silhouette_test | davies_bouldin_test |\n")
    file.write("|---|---|---|---|---|\n")

    for row in rows:
        file.write(
            f"| {row['model']} | {row['k']} | "
            f"{row['silhouette_val']} | {row['silhouette_test']} | "
            f"{row['davies_bouldin_test']} |\n"
        )

pd.DataFrame(k_scores).to_csv(
    RESULTS / "k_selection.csv", index=False
)

summary = {
    "best_k_from_validation": best_k,
    "k_selection": k_scores,
    "baseline_results": rows,
    "ari_kmeans_vs_gmm_test": ari,
    "mean_ari_across_seeds": stability,
}

with open(RESULTS / "summary.json", "w") as file:
    json.dump(summary, file, indent=2)

print(json.dumps(summary, indent=2))
print("Results saved in the results folder.")
