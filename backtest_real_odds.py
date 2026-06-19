import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

with open('model.pkl','rb') as f:
    saved = pickle.load(f)
model, feat_cols = saved['model'], saved['features']

print("Model expects these features:")
print(feat_cols)

df   = pd.read_parquet('matches_features.parquet')
odds = pd.read_csv('odds_all_years.csv')

df = df[df['tourney_date'] >= '2023-01-01'].copy()
print(f"\nHold-out matches: {len(df)}")

odds['Date'] = pd.to_datetime(odds['Date'], dayfirst=True)

def our_last(n):  return str(n).strip().split()[-1].lower()
def odds_last(n): return str(n).strip().split()[0].lower()

df['w_last']   = df['winner_name'].apply(our_last)
df['l_last']   = df['loser_name'].apply(our_last)
odds['w_last'] = odds['Winner'].apply(odds_last)
odds['l_last'] = odds['Loser'].apply(odds_last)
df['surf']     = df['surface'].str.lower().str.strip()
odds['surf']   = odds['Surface'].str.lower().str.strip()
df['month']    = pd.to_datetime(df['tourney_date']).dt.to_period('M')
odds['month']  = odds['Date'].dt.to_period('M')

o_rows = []
for _, o in odds.iterrows():
    o_rows.append({'w_last': o['w_last'], 'l_last': o['l_last'],
                   'surf': o['surf'], 'month': o['month'],
                   'winner_odds': o['AvgW'], 'loser_odds': o['AvgL']})
odds2 = pd.DataFrame(o_rows)

merged = df.merge(odds2, on=['w_last','l_last','surf','month'], how='inner')
merged = merged.drop_duplicates(subset=['w_last','l_last','surf','month'])
print(f"Matched: {len(merged)} matches with real odds")

np.random.seed(42)
rows = []
for _, m in merged.iterrows():
    flip   = np.random.rand() > 0.5
    r, o   = ('l','w') if flip else ('w','l')
    p1_won = 0 if flip else 1
    avg_p1 = m['loser_odds']  if flip else m['winner_odds']
    p1_rank= m['loser_rank']  if flip else m['winner_rank']
    p2_rank= m['winner_rank'] if flip else m['loser_rank']
    p1_age = m['loser_age']   if flip else m['winner_age']
    p2_age = m['winner_age']  if flip else m['loser_age']

    rows.append({
        'p1_won':           p1_won,
        'surface':          m['surface'],
        'tourney_level':    m['tourney_level'],
        'best_of':          m['best_of'],
        'round':            m['round'],
        'rank_diff':        p1_rank - p2_rank,
        'rank_ratio':       p1_rank / max(p2_rank, 1),
        # form stats — exact column names from parquet
        'p1_form_won':      m[f'{r}_form_roll_won'],
        'p1_form_1st':      m[f'{r}_form_roll_1st_pct'],
        'p1_form_1st_w':    m[f'{r}_form_roll_1st_won'],
        'p1_form_2nd_w':    m[f'{r}_form_roll_2nd_won'],
        'p1_form_bp':       m[f'{r}_form_roll_bp_save'],
        'p1_form_ace':      m[f'{r}_form_roll_ace_rate'],
        'p1_form_df':       m[f'{r}_form_roll_df_rate'],
        'p2_form_won':      m[f'{o}_form_roll_won'],
        'p2_form_1st':      m[f'{o}_form_roll_1st_pct'],
        'p2_form_1st_w':    m[f'{o}_form_roll_1st_won'],
        'p2_form_2nd_w':    m[f'{o}_form_roll_2nd_won'],
        'p2_form_bp':       m[f'{o}_form_roll_bp_save'],
        'p2_form_ace':      m[f'{o}_form_roll_ace_rate'],
        'p2_form_df':       m[f'{o}_form_roll_df_rate'],
        'form_won_delta':   m[f'{r}_form_roll_won']      - m[f'{o}_form_roll_won'],
        'serve_delta':      m[f'{r}_form_roll_1st_won']  - m[f'{o}_form_roll_1st_won'],
        'bp_delta':         m[f'{r}_form_roll_bp_save']  - m[f'{o}_form_roll_bp_save'],
        # new features
        'p1_age':           p1_age,
        'p2_age':           p2_age,
        'age_diff':         p1_age - p2_age,
        'p1_days_rest':     m[f'{r}_form_days_rest'],
        'p2_days_rest':     m[f'{o}_form_days_rest'],
        'p1_surface_wr':    m[f'{r}_form_roll_surface_won'],
        'p2_surface_wr':    m[f'{o}_form_roll_surface_won'],
        'p1_deciding_wr':   m[f'{r}_form_roll_deciding_won'],
        'p2_deciding_wr':   m[f'{o}_form_roll_deciding_won'],
        'rest_delta':       m[f'{r}_form_days_rest'] - m[f'{o}_form_days_rest'],
        'surface_wr_delta': m[f'{r}_form_roll_surface_won'] - m[f'{o}_form_roll_surface_won'],
        'market_odds':      avg_p1,
        'market_prob':      1 / avg_p1,
    })

model_df = pd.DataFrame(rows)
for col in ['surface','tourney_level','round']:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))

# Verify all expected features are present before running
missing = [f for f in feat_cols if f not in model_df.columns]
if missing:
    print(f"\nWARNING — missing features: {missing}")
else:
    print("\nAll features present — no missing columns.")

X = model_df[feat_cols].fillna(-1)
model_df['prob'] = model.predict_proba(X)[:,1]
model_df['edge'] = model_df['prob'] - model_df['market_prob']
model_df['bet'] = (model_df['edge'] > 0.05) & (model_df['market_prob'] > 0.50) & (model_df['market_prob'] < 0.65)
STAKE      = 10
COMMISSION = 0.05

def pnl(row):
    if not row['bet']: return 0
    if row['p1_won'] == 1:
        return STAKE * (row['market_odds'] - 1) * (1 - COMMISSION)
    return -STAKE

model_df['pnl'] = model_df.apply(pnl, axis=1)

bets         = model_df[model_df['bet']]
n_bets       = len(bets)
n_wins       = int(bets['p1_won'].sum())
total_staked = n_bets * STAKE
total_pnl    = model_df['pnl'].sum()
roi          = total_pnl / total_staked * 100 if total_staked > 0 else 0

print()
print("=" * 50)
print("BACKTEST — HONEST VERSION")
print("=" * 50)
print(f"Bets placed:   {n_bets} of {len(model_df)} matches")
print(f"Win rate:      {n_wins/n_bets*100:.1f}%")
print(f"Total staked:  £{total_staked:.0f}")
print(f"Total P&L:     £{total_pnl:.2f}")
print(f"ROI:           {roi:.1f}%")
print(f"Avg per bet:   £{total_pnl/max(n_bets,1):.2f}")

# Check NaN rates for key features
print("\nNaN rates in matched data (before fillna):")
for f in ['p1_days_rest','p1_surface_wr','p1_deciding_wr','p1_form_won']:
    nan_pct = model_df[f].isna().mean() * 100
    print(f"  {f}: {nan_pct:.1f}% NaN")