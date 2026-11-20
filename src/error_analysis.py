"""List NBA player-seasons where K-means and GMM disagree at k = 2."""

import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import FEATURES, OUT_PATH, OUTPUT, fit_model, split_and_scale


def main():
    data = pd.read_csv(OUT_PATH)
    x_train, _, x_test = split_and_scale(data, FEATURES)
    kmeans = fit_model("K-means", 2, 0, x_train)
    gmm = fit_model("GMM", 2, 0, x_train)

    kmeans_train_labels = kmeans.predict(x_train)
    gmm_train_labels = gmm.predict(x_train)
    gmm_test_labels = gmm.predict(x_test)

    # At k = 2, swapping 0 and 1 may make the label names line up better.
    same = (kmeans_train_labels == gmm_train_labels).sum()
    swapped = (kmeans_train_labels == 1 - gmm_train_labels).sum()
    if swapped > same:
        gmm_test_labels = 1 - gmm_test_labels

    columns = ["player", "season", "tm", "mp", *FEATURES]
    players = data.loc[data["split"] == "test", columns].copy()
    players["kmeans_cluster"] = kmeans.predict(x_test)
    players["gmm_aligned"] = gmm_test_labels
    players["gmm_max_probability"] = gmm.predict_proba(x_test).max(axis=1)

    disagreements = players[players["kmeans_cluster"] != players["gmm_aligned"]]
    disagreements = disagreements.sort_values(["gmm_max_probability", "player"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "player_disagreements.csv"
    disagreements.to_csv(path, index=False)
    percent = 100 * len(disagreements) / len(players)
    print(f"{len(disagreements)} of {len(players)} test rows disagreed ({percent:.1f}%).")
    print("Saved", path)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
