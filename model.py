import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('matches_features.parquet')
print(f"Loaded {len(df)} matches")

rows = []
for _, m in df.iterrows():
    base = {
        'surface':       m['surface'],
        'tourney_level': m['tourney_level'],
        'best_of':       m['best_of'],
        'date':          m['tourney_date'],
        'round':         m['round'],
    }
    for p1, p2 in [('winner','loser'), ('loser','winner')]:
        r = 'w' if p1 == 'winner' else 'l'
        o = 'l' if p1 == 'winner' else 'w'

        rows.append({**base,
            'label':            1 if p1 == 'winner' else 0,
            'rank_diff':        m[f'{p1}_rank'] - m[f'{p2}_rank'],
            'rank_ratio':       m[f'{p1}_rank'] / max(m[f'{p2}_rank'], 1),
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
            'p1_age':           m[f'{p1}_age'],
            'p2_age':           m[f'{p2}_age'],
            'age_diff':         m[f'{p1}_age'] - m[f'{p2}_age'],
            'p1_days_rest':     m[f'{r}_form_days_rest'],
            'p2_days_rest':     m[f'{o}_form_days_rest'],
            'p1_surface_wr':    m[f'{r}_form_roll_surface_won'],
            'p2_surface_wr':    m[f'{o}_form_roll_surface_won'],
            'p1_deciding_wr':   m[f'{r}_form_roll_deciding_won'],
            'p2_deciding_wr':   m[f'{o}_form_roll_deciding_won'],
            'rest_delta':       m[f'{r}_form_days_rest'] - m[f'{o}_form_days_rest'],
            'surface_wr_delta': m[f'{r}_form_roll_surface_won'] - m[f'{o}_form_roll_surface_won'],
        })

model_df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)

for col in ['surface','tourney_level','round']:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))

feat_cols = [c for c in model_df.columns if c not in ['label','date']]
X = model_df[feat_cols].fillna(-1)
y = model_df['label']

print(f"Model rows: {len(X)}, features: {len(feat_cols)}")

tscv = TimeSeriesSplit(n_splits=5)
scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                                num_leaves=31, min_child_samples=20, subsample=0.8,
                                colsample_bytree=0.8, random_state=42, verbose=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    probs = model.predict_proba(X_val)[:,1]
    acc = (probs.round() == y_val).mean()
    scores.append(acc)
    print(f"Fold {fold+1}: accuracy={acc:.3f}")

print(f"\nMean CV accuracy: {np.mean(scores):.3f}")

cutoff = pd.Timestamp('2023-01-01')
train_mask = (model_df['date'] >= pd.Timestamp('2020-01-01')) & (model_df['date'] < cutoff)
test_mask  = model_df['date'] >= cutoff

print(f"\nTraining on {train_mask.sum()} rows before 2023")
print(f"Testing on  {test_mask.sum()} rows from 2023+")

final_model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                                  num_leaves=31, min_child_samples=20, subsample=0.8,
                                  colsample_bytree=0.8, random_state=42, verbose=-1)
final_model.fit(X[train_mask], y[train_mask])

test_probs = final_model.predict_proba(X[test_mask])[:,1]
test_acc   = (test_probs.round() == y[test_mask]).mean()
print(f"Hold-out accuracy (2023+): {test_acc:.3f}")

importance = pd.DataFrame({
    'feature': feat_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 15 most important features:")
print(importance.head(15).to_string(index=False))

with open('model.pkl','wb') as f:
    pickle.dump({'model': final_model, 'features': feat_cols}, f)
print("\nModel saved to model.pkl")