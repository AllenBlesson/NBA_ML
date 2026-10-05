import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the NBA data
df = pd.read_csv("NBA_ML/data/raw/nba_stats/Advanced.csv")

print("Original data:")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# Columns we need for the project
columns = [
    "player",
    "season",
    "tm",
    "mp",
    "usg_percent",
    "ts_percent",
    "x3p_ar",
    "ast_percent",
    "trb_percent"
]

df = df[columns]

# Convert statistics to numbers
numeric_columns = [
    "season",
    "mp",
    "usg_percent",
    "ts_percent",
    "x3p_ar",
    "ast_percent",
    "trb_percent"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Remove rows with missing values
df = df.dropna()

# Keep seasons 2016 to 2024
df = df[(df["season"] >= 2016) & (df["season"] <= 2024)]

# For players who played for multiple teams,
# keep their combined "TOT" row.
players_with_tot = df[df["tm"] == "TOT"][["player", "season"]].drop_duplicates()

for _, row in players_with_tot.iterrows():

    player = row["player"]
    season = row["season"]

    player_season = df[
        (df["player"] == player) &
        (df["season"] == season)
    ]

    if len(player_season) > 1:

        df = df[
            ~(
                (df["player"] == player) &
                (df["season"] == season) &
                (df["tm"] != "TOT")
            )
        ]

# Check for invalid values
print("\nInvalid values:")

print("Negative minutes:", (df["mp"] < 0).sum())
print("Usage below 0:", (df["usg_percent"] < 0).sum())
print("Usage above 100:", (df["usg_percent"] > 100).sum())
print("TS below 0:", (df["ts_percent"] < 0).sum())
print("TS above 1:", (df["ts_percent"] > 1).sum())
print("3P attempt rate below 0:", (df["x3p_ar"] < 0).sum())
print("3P attempt rate above 1:", (df["x3p_ar"] > 1).sum())
print("AST below 0:", (df["ast_percent"] < 0).sum())
print("TRB below 0:", (df["trb_percent"] < 0).sum())

# Only keep players with at least 500 minutes
rows_before_minutes = len(df)

df = df[df["mp"] >= 500]

rows_after_minutes = len(df)

print("\n500-minute filter:")
print("Rows before:", rows_before_minutes)
print("Rows after:", rows_after_minutes)
print("Rows removed:", rows_before_minutes - rows_after_minutes)

# Reset row numbers
df = df.reset_index(drop=True)

print("\nCleaned data:")
print("Rows:", len(df))
print("Seasons:", df["season"].min(), "to", df["season"].max())

print("\nRows per season:")
print(df["season"].value_counts().sort_index())

# Basic statistics
print("\nSummary statistics:")
print(df[
    [
        "mp",
        "usg_percent",
        "ts_percent",
        "x3p_ar",
        "ast_percent",
        "trb_percent"
    ]
].describe())

# -------------------------
# PLOTS
# -------------------------

# Usage distribution
plt.figure()
plt.hist(df["usg_percent"], bins=20)
plt.xlabel("Usage %")
plt.ylabel("Number of Players")
plt.title("Distribution of Usage")
plt.savefig("NBA_ML/figures/usage_distribution.png")
plt.show()

# True shooting distribution
plt.figure()
plt.hist(df["ts_percent"], bins=20)
plt.xlabel("True Shooting %")
plt.ylabel("Number of Players")
plt.title("Distribution of True Shooting")
plt.savefig("NBA_ML/figures/true_shooting_distribution.png")
plt.show()

# Three-point attempt rate
plt.figure()
plt.hist(df["x3p_ar"], bins=20)
plt.xlabel("3-Point Attempt Rate")
plt.ylabel("Number of Players")
plt.title("Distribution of 3-Point Attempt Rate")
plt.savefig("NBA_ML/figures/three_point_attempt_rate.png")
plt.show()

# Correlation between features
features = [
    "usg_percent",
    "ts_percent",
    "x3p_ar",
    "ast_percent",
    "trb_percent"
]

plt.figure()
sns.heatmap(df[features].corr(), annot=True)
plt.title("Feature Correlations")
plt.savefig("NBA_ML/figures/feature_correlations.png")
plt.show()

# See how some features changed over time
yearly = df.groupby("season")[features].mean()

plt.figure()
plt.plot(yearly.index, yearly["usg_percent"])
plt.xlabel("Season")
plt.ylabel("Average Usage %")
plt.title("Average Usage Over Time")
plt.savefig("NBA_ML/figures/average_usage_over_time.png")
plt.show()

plt.figure()
plt.plot(yearly.index, yearly["ts_percent"])
plt.xlabel("Season")
plt.ylabel("Average True Shooting %")
plt.title("Average True Shooting Over Time")
plt.savefig("NBA_ML/figures/average_true_shooting_over_time.png")
plt.show()

# Show a few examples
print("\nExample players:")
print(df.sample(5))

# Save cleaned data
df.to_csv("NBA_ML/data/processed/cleaned_nba.csv", index=False)

print("\nCleaned data saved to NBA_ML/data/processed/cleaned_nba.csv")