"""Compare K-means and GMM on NBA player-seasons."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import FEATURES, OUT_PATH


OUTPUT = Path(__file__).resolve().parents[1] / "results" / "experiments"
MODELS = ["K-means", "GMM"]
SEEDS = range(10)
K_VALUES = range(2, 9)


def split_and_scale(data, features):
    train = data[data["split"] == "train"]
    val = data[data["split"] == "val"]
    test = data[data["split"] == "test"]

    # Fit on training data so later seasons do not affect the scaling.
    scaler = StandardScaler().fit(train[features])
    return (
        scaler.transform(train[features]),
        scaler.transform(val[features]),
        scaler.transform(test[features]),
    )


def fit_model(name, k, seed, x_train, covariance="full"):
    if name == "K-means":
        model = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=seed)
    else:
        model = GaussianMixture(
            n_components=k,
            covariance_type=covariance,
            n_init=10,
            max_iter=300,
            random_state=seed,
        )
    return model.fit(x_train)


def main():
    data = pd.read_csv(OUT_PATH)
    x_train, x_val, x_test = split_and_scale(data, FEATURES)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Step 1: Pick k for each model using the validation seasons.
    fits = {}
    validation_rows = []
    for name in MODELS:
        for k in K_VALUES:
            for seed in SEEDS:
                model = fit_model(name, k, seed, x_train)
                fits[name, k, seed] = model
                validation_rows.append({
                    "model": name,
                    "k": k,
                    "seed": seed,
                    "validation_silhouette": silhouette_score(x_val, model.predict(x_val)),
                })

    validation = pd.DataFrame(validation_rows)
    val_scores = validation.groupby(["model", "k"])["validation_silhouette"].mean().unstack(0)
    selected_k = val_scores.idxmax().to_dict()
    validation.to_csv(OUTPUT / "validation_runs.csv", index=False)
    print("Best k on validation:", selected_k)

    # Step 2: Evaluate both models on the same 2024 rows.
    test_rows = []
    comparison_ks = sorted({2, 5, *selected_k.values()})
    for k in comparison_ks:
        for seed in SEEDS:
            for name in MODELS:
                labels = fits[name, k, seed].predict(x_test)
                test_rows.append(score_test(name, k, seed, x_test, labels))

            # Random labels provide a baseline at the same value of k.
            rng = np.random.default_rng(seed)
            rng.integers(k, size=len(x_val))
            labels = rng.integers(k, size=len(x_test))
            test_rows.append(score_test("Random labels", k, seed, x_test, labels))

    positions = data.loc[data["split"] == "test", "pos"]
    primary_positions = positions.str.split("-").str[0]
    test_rows.append(score_test("Raw positions", positions.nunique(), None, x_test, positions))
    test_rows.append(score_test(
        "Five positions", primary_positions.nunique(), None, x_test, primary_positions,
    ))

    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(OUTPUT / "test_runs.csv", index=False)
    summary = test_results.groupby(["model", "k"]).agg(
        mean_silhouette=("silhouette", "mean"),
        sd_silhouette=("silhouette", "std"),
        mean_davies_bouldin=("davies_bouldin", "mean"),
    )
    summary.to_csv(OUTPUT / "model_summary.csv")
    print(summary.round(4).to_string())

    # Step 3: Change one setting at a time and compare with the k=2 models.
    checks = []
    noise = np.random.default_rng(3052).normal(0, 0.25, x_test.shape)
    for name in MODELS:
        for seed in SEEDS:
            model = fits[name, 2, seed]
            checks.append(check_row(name, seed, "All features", x_test, model.predict(x_test)))
            # Score the changed assignments on the same original test features.
            checks.append(check_row(name, seed, "Added noise", x_test, model.predict(x_test + noise)))

        for feature in FEATURES:
            kept = [column for column in FEATURES if column != feature]
            reduced_train, _, reduced_test = split_and_scale(data, kept)
            for seed in SEEDS:
                model = fit_model(name, 2, seed, reduced_train)
                checks.append(check_row(
                    name, seed, f"Without {feature}", x_test, model.predict(reduced_test),
                ))

    for covariance in ["tied", "diag", "spherical"]:
        for seed in SEEDS:
            model = fit_model("GMM", 2, seed, x_train, covariance)
            checks.append(check_row("GMM", seed, covariance, x_test, model.predict(x_test)))

    pd.DataFrame(checks).to_csv(OUTPUT / "model_checks.csv", index=False, encoding="utf-8")

    ax = val_scores.plot(marker="o")
    ax.set(xlabel="Number of clusters", ylabel="Mean validation silhouette")
    ax.figure.tight_layout()
    ax.figure.savefig(OUTPUT / "validation_scores.png", dpi=160)
    plt.close(ax.figure)
    print("Saved results to", OUTPUT)


def score_test(name, k, seed, x_test, labels):
    return {
        "model": name,
        "k": k,
        "seed": seed,
        "silhouette": silhouette_score(x_test, labels),
        "davies_bouldin": davies_bouldin_score(x_test, labels),
    }


def check_row(name, seed, change, x_test, labels):
    return {
        "model": name,
        "seed": seed,
        "change": change,
        "silhouette": silhouette_score(x_test, labels),
    }


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
