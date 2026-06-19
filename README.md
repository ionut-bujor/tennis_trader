# tennis_trader

A machine learning system that predicts ATP tennis match outcomes and simulates a betting strategy against historical Betfair market odds.

Built in Python using LightGBM, trained on 6 years of ATP match data (2019–2024) with real Betfair odds.

---

## How it works

**1. Feature engineering**
Each match is represented twice — once from each player's perspective — with 30+ features capturing:
- Recent form (rolling win rate, serve %, break point save rate, ace/double-fault rate)
- Surface-specific win rate
- Head-to-head rank differential and ratio
- Days rest, age, deciding-set win rate

**2. Model training (`model.py`)**
A LightGBM classifier is trained with TimeSeriesSplit cross-validation to avoid data leakage. The final model trains on 2020–2022 data and is evaluated on a held-out 2023+ test set.

**3. Backtesting (`backtest.py`)**
The model's predicted win probability is compared against an implied market probability derived from player rankings (a proxy for Betfair odds). A flat £10 bet is placed whenever the model's edge exceeds 5%. P&L is calculated after a 5% Betfair commission, with results broken down monthly.

---

## Results

> Run `model.py` then `backtest.py` to reproduce.

Key metrics printed to console:
- Cross-validated accuracy across 5 time-ordered folds
- Hold-out accuracy (2023+ matches)
- Total bets, win rate, ROI, and monthly P&L breakdown

Results CSV saved to `backtest_results.csv`.

---

## Project structure

```
tennis_trader/
│
├── model.py                  # Feature construction, model training, evaluation
├── backtest.py               # Betting simulation against market odds
├── backtest_real_odds.py     # Backtest variant using real Betfair odds CSVs
│
├── matches_features.parquet  # Engineered feature dataset (generated from ATP data)
├── model.pkl                 # Saved trained model
├── backtest_results.csv      # Output from backtest.py
│
└── data/
    ├── atp_matches/          # Raw ATP match results by year (2019–2024)
    └── odds/                 # Historical Betfair odds by tournament
```

---

## Setup

```bash
pip install lightgbm pandas numpy scikit-learn pyarrow
```

Then run in order:

```bash
python model.py       # trains the model, saves model.pkl
python backtest.py    # simulates the betting strategy, saves backtest_results.csv
```

---

## Data sources

- ATP match results: [tennis-data.co.uk](http://www.tennis-data.co.uk/) / Jeff Sackmann's [tennis_atp](https://github.com/JeffSackmann/tennis_atp)
- Betfair historical odds: [tennis-data.co.uk](http://www.tennis-data.co.uk/)

---

## Limitations

- Market odds in `backtest.py` are approximated from player rankings rather than actual pre-match Betfair prices. `backtest_real_odds.py` uses the real odds CSVs where available but coverage is partial.
- No live trading integration — this is a research/backtesting tool only.
- Past performance does not imply future edge; tennis markets are efficient and odds move quickly.
