import pandas as pd

# Load dataset
adv = pd.read_csv("NBA_ML/data/raw/nba_stats/Advanced.csv")

# Select columns needed for the project
cols = [
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

df = adv[cols].copy()

# Convert numerical columns to numeric
numeric_cols = [
    "season",
    "mp",
    "usg_percent",
    "ts_percent",
    "x3p_ar",
    "ast_percent",
    "trb_percent"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove duplicate rows and missing values
df = df.drop_duplicates()
df = df.dropna()

# Keep seasons from 2016 to 2024
df = df[(df["season"] >= 2016) & (df["season"] <= 2024)]

# Keep players with at least 500 minutes
df = df[df["mp"] >= 500]

# Reset index
df = df.reset_index(drop=True)

# Check cleaned dataset
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSample:")
print(df.sample(3, random_state=0))