# NBA ML: Finding Complementary Fits for a Star Player

CSCI 3052U Machine Learning I group project.

Team 27: Nooh Alavi, Alan Damy, Allen Blesson, Musa Zubair, Pasoon Akhunzada

## What this project does
We group NBA player-seasons (2016-2024) into play-style types using k-means and Gaussian Mixture Models. Then, for a given star player, we look for player types that complement their style.

## Data
- NBA Stats (1980-2024) by Rodney Carroll on Kaggle: https://www.kaggle.com/datasets/rodneycarroll78/nba-stats-1980-2024
- License: CC0 (stats come from Basketball-Reference). More details in `data/README.md`.
- Raw files are in `data/raw/nba_stats/`. The cleaned data is `data/processed/cleaned_nba.csv`.

## Folders
- `data/`: raw and cleaned data
- `src/`: pipeline and baseline code
- `results/`: baseline table and k selection results
- `notebooks/`: EDA code from Milestone 2
- `figures/`: plots used in the reports
- `reports/`: milestone reports (PDF)



## How to run
1. Install the packages: `pip install -r requirements.txt`
2. Download the dataset into `data/raw/nba_stats/`
3. Run the data pipeline: `python src/pipeline.py` (writes `data/processed/cleaned_nba.csv` and `log.json`)
4. Run the baselines: `python src/baseline.py` (writes the files in `results/`)

## Releases
- `v1.0-baseline`: Milestone 3 baseline reproduction

## Other files
- `AI_USE.md`: How we used AI tools
- `CONTRIBUTIONS.md`: Who contributed to what
