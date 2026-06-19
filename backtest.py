import pandas as pd
import numpy as np
import pickle

# Load model and data
with open('model.pkl','rb') as f:
    saved = pickle.load(f)
model    = saved['model']
feat_cols= saved['features']

df = pd.read_parquet('matches_features.parquet')

# Rebuild model_df (same as model.py) — just winner rows this time
from sklearn.preprocessing import LabelEncoder
rows = []
for _, m in df.iterrows():
    for p1, p2, label in [('winner','loser',1),('loser','winner',0)]:
        r = 'w' if p1=='winner' else 'l'
        o = 'l' if p1=='winner' else 'w'
        rows.append({
            'date': m['tourney_date'], 'label': label,
            'surface': m['surface'], 'tourney_level': m['tourney_level'],
            'best_of': m['best_of'], 'round': m['round'],
            'rank_diff':     m[f'{p1}_rank'] - m[f'{p2}_rank'],
            'rank_ratio':    m[f'{p1}_rank'] / max(m[f'{p2}_rank'],1),
            'p1_form_won':   m[f'{r}_form_won'],
            'p1_form_1st':   m[f'{r}_form_1st_pct'],
            'p1_form_1st_w': m[f'{r}_form_1st_won'],
            'p1_form_2nd_w': m[f'{r}_form_2nd_won'],
            'p1_form_bp':    m[f'{r}_form_bp_save'],
            'p1_form_ace':   m[f'{r}_form_ace_rate'],
            'p1_form_df':    m[f'{r}_form_df_rate'],
            'p2_form_won':   m[f'{o}_form_won'],
            'p2_form_1st':   m[f'{o}_form_1st_pct'],
            'p2_form_1st_w': m[f'{o}_form_1st_won'],
            'p2_form_2nd_w': m[f'{o}_form_2nd_won'],
            'p2_form_bp':    m[f'{o}_form_bp_save'],
            'p2_form_ace':   m[f'{o}_form_ace_rate'],
            'p2_form_df':    m[f'{o}_form_df_rate'],
            'form_won_delta':m[f'{r}_form_won']    - m[f'{o}_form_won'],
            'serve_delta':   m[f'{r}_form_1st_won']- m[f'{o}_form_1st_won'],
            'bp_delta':      m[f'{r}_form_bp_save']- m[f'{o}_form_bp_save'],
            # store for backtest
            'winner_rank':   m['winner_rank'],
            'loser_rank':    m['loser_rank'],
        })

model_df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
for col in ['surface','tourney_level','round']:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))

X = model_df[feat_cols].fillna(-1)
model_df['prob'] = model.predict_proba(X)[:,1]

# ── Simulate Betfair implied odds from rank ───────────────────────────────────
# In reality you'd use historical Betfair odds — this approximates the market
# using Elo-style rank conversion so the backtest is still meaningful
def rank_to_prob(r1, r2):
    # Simple approximation: log-rank ratio → probability
    log_ratio = np.log(r2 / max(r1, 1))
    return 1 / (1 + np.exp(-0.5 * log_ratio))

model_df['market_prob'] = model_df.apply(
    lambda r: rank_to_prob(r['winner_rank'], r['loser_rank'])
              if r['label']==1
              else 1 - rank_to_prob(r['winner_rank'], r['loser_rank']), axis=1)

# ── Betting strategy: bet when model edge > threshold ────────────────────────
EDGE_THRESHOLD = 0.05   # model thinks prob is 5%+ higher than market
BETFAIR_COMMISSION = 0.05
STAKE = 10              # flat £10 per bet

# Only use hold-out period (last 20% of data by time)
cutoff = int(len(model_df) * 0.8)
test   = model_df.iloc[cutoff:].copy()

test['edge']       = test['prob'] - test['market_prob']
test['bet']        = test['edge'] > EDGE_THRESHOLD
test['market_odds']= 1 / test['market_prob'].clip(0.01, 0.99)

# P&L calculation
def calc_pnl(row):
    if not row['bet']: return 0
    if row['label'] == 1:  # we bet and won
        profit = STAKE * (row['market_odds'] - 1)
        return profit * (1 - BETFAIR_COMMISSION)
    else:
        return -STAKE

test['pnl'] = test.apply(calc_pnl, axis=1)
test['cumulative_pnl'] = test['pnl'].cumsum()

bets     = test[test['bet']]
winners  = bets[bets['label']==1]
n_bets   = len(bets)
n_wins   = len(winners)
total_staked = n_bets * STAKE
total_pnl    = test['pnl'].sum()
roi          = (total_pnl / total_staked * 100) if total_staked > 0 else 0
win_rate     = (n_wins / n_bets * 100) if n_bets > 0 else 0

print("=" * 50)
print("BACKTEST RESULTS (hold-out period)")
print("=" * 50)
print(f"Total matches assessed:  {len(test)//2}")
print(f"Bets placed:             {n_bets}")
print(f"Win rate:                {win_rate:.1f}%")
print(f"Total staked:            £{total_staked:.0f}")
print(f"Total P&L:               £{total_pnl:.2f}")
print(f"ROI:                     {roi:.1f}%")
print(f"Avg P&L per bet:         £{total_pnl/max(n_bets,1):.2f}")
print()
print("Edge distribution in triggered bets:")
print(bets['edge'].describe().round(3).to_string())
print()
# Monthly breakdown
test['month'] = pd.to_datetime(test['date']).dt.to_period('M')
monthly = test.groupby('month')['pnl'].agg(['sum','count']).rename(
    columns={'sum':'pnl','count':'all_rows'})
monthly['bets'] = test[test['bet']].groupby('month')['pnl'].count()
monthly = monthly[monthly['bets'] > 0].fillna(0)
print("Monthly P&L:")
print(monthly[['bets','pnl']].tail(12).to_string())

test[['date','prob','market_prob','edge','bet','pnl','cumulative_pnl']].to_csv(
    'backtest_results.csv', index=False)
print("\nFull results saved to backtest_results.csv")
