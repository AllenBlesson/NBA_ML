import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

DATA_PATH = "NBA_ML/data/processed/cleaned_nba.csv"
FEATURES = ["usg_percent", "ts_percent", "x3p_ar", "ast_percent", "trb_percent"]
RANDOM_STATE = 0
K_RANGE = range(2, 9)

df = pd.read_csv(DATA_PATH)
train = df[df["split"] == "train"]
val = df[df["split"] == "val"]
test = df[df["split"] == "test"]

# scale features, fit scaler on train only
scaler = StandardScaler()
X_train = scaler.fit_transform(train[FEATURES])
X_val = scaler.transform(val[FEATURES])
X_test = scaler.transform(test[FEATURES])

# try different k values on k-means, pick the best one using validation silhouette score
k_selection = []
best_k, best_score = None, -1
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_train)
    score = silhouette_score(X_val, km.predict(X_val))
    k_selection.append({"k": k, "silhouette_val": round(score, 4)})
    if score > best_score:
        best_k, best_score = k, score

def score_row(name, k, val_labels, test_labels):
    return {
        "model": name,
        "k": k,
        "silhouette_val": round(silhouette_score(X_val, val_labels), 4),
        "silhouette_test": round(silhouette_score(X_test, test_labels), 4),
        "davies_bouldin_test": round(davies_bouldin_score(X_test, test_labels), 4),
    }

# naive baseline: random cluster labels (same k as k-means)
rng = np.random.RandomState(RANDOM_STATE)
naive_val_labels = rng.randint(0, best_k, size=len(val))
naive_test_labels = rng.randint(0, best_k, size=len(test))
naive_row = score_row("Naive: random labels", best_k, naive_val_labels, naive_test_labels)

# reference baseline: use player position as the "cluster"
pos_encoder = LabelEncoder()
pos_encoder.fit(df["pos"])
pos_val_labels = pos_encoder.transform(val["pos"])
pos_test_labels = pos_encoder.transform(test["pos"])
reference_row = score_row("Reference: position labels", len(pos_encoder.classes_), pos_val_labels, pos_test_labels)

# course-level baseline: k-means with the chosen k
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
kmeans.fit(X_train)
kmeans_val_labels = kmeans.predict(X_val)
kmeans_test_labels = kmeans.predict(X_test)
kmeans_row = score_row("Course-level baseline: k-means", best_k, kmeans_val_labels, kmeans_test_labels)

# candidate: Gaussian mixture with the same k, for a fair comparison
gmm = GaussianMixture(n_components=best_k, random_state=RANDOM_STATE)
gmm.fit(X_train)
gmm_val_labels = gmm.predict(X_val)
gmm_test_labels = gmm.predict(X_test)
gmm_row = score_row("Candidate: Gaussian mixture", best_k, gmm_val_labels, gmm_test_labels)

# how much do kmeans and gmm agree on the test set?
ari = round(adjusted_rand_score(kmeans_test_labels, gmm_test_labels), 4)

rows = [naive_row, reference_row, kmeans_row, gmm_row]

os.makedirs("results", exist_ok=True)

# save results table
with open("results/baseline_table.md", "w") as f:
    f.write("| model | k | silhouette_val | silhouette_test | davies_bouldin_test |\n")
    f.write("|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['model']} | {r['k']} | {r['silhouette_val']} | {r['silhouette_test']} | {r['davies_bouldin_test']} |\n")

pd.DataFrame(k_selection).to_csv("results/k_selection.csv", index=False)

summary = {
    "best_k_from_validation": best_k,
    "k_selection": k_selection,
    "baseline_results": rows,
    "ari_kmeans_vs_gmm_test": ari,
}
with open("results/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print("\nSaved results/baseline_table.md, results/k_selection.csv, results/summary.json")