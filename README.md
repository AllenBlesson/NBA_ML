# NBA ML: Finding Complementary Fits for a Star Player

CSCI 3052U Machine Learning I group project.

Team: Nooh Alavi, Alan Damy, Allen Blesson, Musa Zubair, Pasoon Akhunzada

## What this project does
We group NBA player-seasons (2016-2024) into play-style types using k-means and Gaussian Mixture Models. Then, for a given star player, we look for player types that complement their style.

## Data
- NBA Stats (1980-2024) by Rodney Carroll on Kaggle: https://www.kaggle.com/datasets/rodneycarroll78/nba-stats-1980-2024
- License: CC0 (stats come from Basketball-Reference). More details in `data/README.md`.
- Raw files are in `data/raw/nba_stats/`. The cleaned data is `data/processed/cleaned_nba.csv`.

## Folders
- `data/`: raw and cleaned data
- `notebooks/`: EDA and clustering code
- `figures/`: plots used in the reports
- `reports/`: milestone reports (PDF)


## How to run
1. Install the packages: `pip install -r requirements.txt`
2. Open a terminal in the folder that contains `NBA_ML`
3. Run the EDA and cleaning code first (`notebooks/SampleEDA`). It creates `data/processed/cleaned_nba.csv` and the figures.
4. Then run `notebooks/Sampleinput_output.py` for the sample k-means and GMM output.

## Other files
- `AI_USE.md`: How we used AI tools
- `CONTRIBUTIONS.md`: Who contributed to what
