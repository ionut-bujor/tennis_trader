import pandas as pd
import numpy as np
import urllib.request
from pathlib import Path

years = range(2019, 2025)

for y in years:
    fname = f"atp_matches_{y}.csv"
    if not Path(fname).exists():
        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{y}.csv"
        print(f"Downloading {y}...")
        urllib.request.urlretrieve(url, fname)
        print(f"  done.")

dfs = [pd.read_csv(f"atp_matches_{y}.csv", low_memory=False) for y in years]
matches = pd.concat(dfs, ignore_index=True)
matches['tourney_date'] = pd.to_datetime(matches['tourney_date'], format='%Y%m%d')
matches = matches.sort_values('tourney_date').reset_index(drop=True)
print(f"Loaded: {len(matches)} matches")

key_cols = ['w_ace','w_df','w_svpt','w_1stIn','w_1stWon','w_2ndWon','w_bpSaved','w_bpFaced',
            'l_ace','l_df','l_svpt','l_1stIn','l_1stWon','l_2ndWon','l_bpSaved','l_bpFaced',
            'winner_rank','loser_rank','minutes','winner_age','loser_age']
matches = matches.dropna(subset=key_cols).reset_index(drop=True)
print(f"After cleaning: {len(matches)} matches")

for p in ['w','l']:
    svpt  = matches[f'{p}_svpt'].clip(lower=1)
    first = matches[f'{p}_1stIn'].clip(lower=1)
    matches[f'{p}_1st_pct'] = matches[f'{p}_1stIn']  / svpt
    matches[f'{p}_1st_won'] = matches[f'{p}_1stWon'] / first
    matches[f'{p}_2nd_won'] = matches[f'{p}_2ndWon'] / (svpt - matches[f'{p}_1stIn']).clip(lower=1)
    matches[f'{p}_ace_rate']= matches[f'{p}_ace']    / svpt
    matches[f'{p}_df_rate'] = matches[f'{p}_df']     / svpt
    matches[f'{p}_bp_save'] = matches[f'{p}_bpSaved']/ matches[f'{p}_bpFaced'].clip(lower=1)

def went_deciding(score, best_of):
    try:
        return 1 if len(str(score).split()) == best_of else 0
    except:
        return 0

matches['went_deciding'] = matches.apply(
    lambda r: went_deciding(r['score'], r['best_of']), axis=1)

print("Building H2H records (this takes a few minutes)...")
h2h_rows = []
for i, row in matches.iterrows():
    w = row['winner_id']
    l = row['loser_id']
    date = row['tourney_date']
    surface = row['surface']

    past = matches[
        (matches['tourney_date'] < date) &
        (
            ((matches['winner_id'] == w) & (matches['loser_id'] == l)) |
            ((matches['winner_id'] == l) & (matches['loser_id'] == w))
        )
    ]
    past_surface = past[past['surface'] == surface]

    p_low  = min(w, l)
    p_high = max(w, l)

    low_wins_total   = len(past[past['winner_id'] == p_low])
    low_wins_surface = len(past_surface[past_surface['winner_id'] == p_low])
    total_played     = len(past)
    surface_played   = len(past_surface)

    h2h_rows.append({
        'idx':               i,
        'p_low':             p_low,
        'low_h2h_wins':      low_wins_total,
        'low_h2h_surf_wins': low_wins_surface,
        'h2h_total':         total_played,
        'h2h_surface_total': surface_played,
        'low_h2h_winrate':   low_wins_total   / max(total_played, 1),
        'low_h2h_surf_wr':   low_wins_surface / max(surface_played, 1),
    })

h2h_df = pd.DataFrame(h2h_rows).set_index('idx')
matches = matches.join(h2h_df)
print("H2H done.")

print("Building player records...")
records = []
for _, row in matches.iterrows():
    for role in ['winner','loser']:
        p = 'w' if role == 'winner' else 'l'
        won = 1 if role == 'winner' else 0
        deciding_win = 1 if (won == 1 and row['went_deciding']) else 0
        records.append({
            'date':            row['tourney_date'],
            'player_id':       row[f'{role}_id'],
            'won':             won,
            'age':             row[f'{role}_age'],
            'surface':         row['surface'],
            'minutes':         row['minutes'],
            '1st_pct':         row[f'{p}_1st_pct'],
            '1st_won':         row[f'{p}_1st_won'],
            '2nd_won':         row[f'{p}_2nd_won'],
            'bp_save':         row[f'{p}_bp_save'],
            'ace_rate':        row[f'{p}_ace_rate'],
            'df_rate':         row[f'{p}_df_rate'],
            'deciding_played': row['went_deciding'],
            'deciding_won':    deciding_win,
        })

player_df = pd.DataFrame(records).sort_values(['player_id','date']).reset_index(drop=True)

roll_cols = ['1st_pct','1st_won','2nd_won','bp_save','ace_rate','df_rate','won',
             'deciding_played','deciding_won']

for col in roll_cols:
    player_df[f'roll_{col}'] = (
        player_df.groupby('player_id')[col]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )

player_df['roll_surface_won'] = (
    player_df.groupby(['player_id','surface'])['won']
    .transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
)

player_df['days_rest'] = (
    player_df.groupby('player_id')['date']
    .transform(lambda x: x.diff().dt.days)
).clip(0, 30)

player_df['roll_minutes'] = (
    player_df.groupby('player_id')['minutes']
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

print("Merging rolling stats back to matches...")
extra_cols = ['roll_surface_won','days_rest','roll_minutes']
player_lookup = player_df.set_index(['player_id','date'])

def get_rolls(pid, date):
    try:
        r = player_lookup.loc[(pid, date)]
        if isinstance(r, pd.DataFrame): r = r.iloc[0]
        result = {f'roll_{c}': r[f'roll_{c}'] for c in roll_cols}
        result['roll_surface_won'] = r['roll_surface_won']
        result['days_rest']        = r['days_rest']
        result['roll_minutes']     = r['roll_minutes']
        return result
    except:
        result = {f'roll_{c}': np.nan for c in roll_cols}
        result['roll_surface_won'] = np.nan
        result['days_rest']        = np.nan
        result['roll_minutes']     = np.nan
        return result

w_rolls = [get_rolls(row['winner_id'], row['tourney_date']) for _, row in matches.iterrows()]
l_rolls = [get_rolls(row['loser_id'],  row['tourney_date']) for _, row in matches.iterrows()]

matches = pd.concat([
    matches,
    pd.DataFrame(w_rolls).add_prefix('w_form_'),
    pd.DataFrame(l_rolls).add_prefix('l_form_'),
], axis=1)

matches.to_parquet('matches_features.parquet', index=False)
print(f"\nDone. Saved {len(matches)} matches with {len(matches.columns)} columns.")