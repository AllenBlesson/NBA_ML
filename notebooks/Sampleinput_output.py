import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Load cleaned data
df = pd.read_csv("NBA_ML/data/processed/cleaned_nba.csv")

features = [
    "usg_percent",
    "ts_percent",
    "x3p_ar",
    "ast_percent",
    "trb_percent"
]

# -------------------------
# TRAINING DATA
# -------------------------

# Use seasons up to 2021 for training
train = df[df["season"] <= 2021].copy()

# Use 2024 as an example test season
test = df[df["season"] == 2024].copy()

# Get the feature values
X_train = train[features]

# Standardize the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# -------------------------
# K-MEANS
# -------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=0,
    n_init=10
)

kmeans.fit(X_train_scaled)

# Assign clusters to training and test players
train["kmeans_cluster"] = kmeans.labels_

test_scaled = scaler.transform(test[features])
test["kmeans_cluster"] = kmeans.predict(test_scaled)

# -------------------------
# GAUSSIAN MIXTURE MODEL
# -------------------------

gmm = GaussianMixture(
    n_components=4,
    random_state=0
)

gmm.fit(X_train_scaled)

train["gmm_cluster"] = gmm.predict(X_train_scaled)
test["gmm_cluster"] = gmm.predict(test_scaled)

# Probability of belonging to the assigned GMM cluster
probabilities = gmm.predict_proba(test_scaled)

test["gmm_probability"] = probabilities.max(axis=1)

# -------------------------
# SHOW EXAMPLE RESULTS
# -------------------------

print("Example 2024 players:")
print(
    test[
        [
            "player",
            "kmeans_cluster",
            "gmm_cluster",
            "gmm_probability"
        ]
    ].sample(5)
)

# -------------------------
# CLUSTER PROFILES
# -------------------------

print("\nK-Means cluster profiles:")

profiles = train.groupby("kmeans_cluster")[features].mean()

print(profiles)

print("\nGMM cluster profiles:")

profiles = train.groupby("gmm_cluster")[features].mean()

print(profiles)

# -------------------------
# CHOOSE A STAR PLAYER
# -------------------------

# For now, use the highest-usage player in 2024
star = test.loc[test["usg_percent"].idxmax()]

print("\nStar player:")
print(star["player"])

print("\nStar statistics:")
print(star[features])

# -------------------------
# SIMPLE COMPLEMENTARY PLAYER EXAMPLE
# -------------------------

# A simple rule:
#
# Look for players who:
# 1. Are in a different cluster than the star
# 2. Have lower usage than the star
#
# This is only a starting rule.
# The final project rule should be decided before
# looking at the final results.

candidates = test[
    (test["gmm_cluster"] != star["gmm_cluster"]) &
    (test["usg_percent"] < star["usg_percent"])
].copy()

# Sort by three-point attempt rate
candidates = candidates.sort_values(
    "x3p_ar",
    ascending=False
)

print("\nExample complementary players:")

print(
    candidates[
        [
            "player",
            "gmm_cluster",
            "usg_percent",
            "ts_percent",
            "x3p_ar",
            "ast_percent",
            "trb_percent"
        ]
    ].head(5)
)