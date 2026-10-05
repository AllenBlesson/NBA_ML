import pandas as pd

adv = pd.read_csv("data/raw/nba_stats/Advanced.csv")
print(adv.columns.tolist())

cols = ["player", "season", "tm", "mp", "usg_percent", "ts_percent",
        "x3p_ar", "ast_percent", "trb_percent"]
sample = adv[(adv["season"] >= 2016) & (adv["mp"] >= 500)][cols]
print(sample.sample(3, random_state=0))