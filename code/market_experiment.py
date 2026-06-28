"""
Wave-3 Full Market-Odds Coverage Experiment
Uses Polymarket data (56/64) + Dixon-Coles model imputation for 8 missing matches.
Runs 4 experiments; saves all OOF probs as artifacts.
"""
import json, os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
warnings.filterwarnings('ignore')

SEED = 0
np.random.seed(SEED)

DATA_DIR = "/home/user/research/wave3-market-odds/fifa_data"
ART_DIR  = "/home/user/research/wave3-market-odds/artifacts"
POLY_JSON = f"{ART_DIR}/polymarket_raw.json"
os.makedirs(ART_DIR, exist_ok=True)

# ── Name normalization: Polymarket → FIFA dataset names ────────────────────
POLY_TO_FIFA = {
    'Czech Republic': 'Czechia',
    'Ivory Coast':    "Côte d'Ivoire",
    'Turkey':         'Türkiye',
    'United States':  'USA',
    'DR Congo':       'Congo DR',
    'Cape Verde':     'Cabo Verde',
    'Iran':           'IR Iran',
}

def normalize(name):
    return POLY_TO_FIFA.get(name, name)

# ── Load canonical match data ──────────────────────────────────────────────
matches   = pd.read_csv(f"{DATA_DIR}/matches_detailed.csv")
teams     = pd.read_csv(f"{DATA_DIR}/teams.csv")
completed = matches[matches['status'] == 'Completed'].copy().reset_index(drop=True)

def get_label(row):
    if row['home_score'] > row['away_score']: return 'H'
    elif row['home_score'] == row['away_score']: return 'D'
    return 'A'

completed['label'] = completed.apply(get_label, axis=1)
le = LabelEncoder()
le.fit(['A', 'D', 'H'])
y = le.transform(completed['label'].values)
print(f"Classes: {le.classes_}  (A=0, D=1, H=2)")
print(f"Label counts: {dict(zip(le.classes_, np.bincount(y)))}")

# ── Elo features ───────────────────────────────────────────────────────────
team_elo = teams.set_index('team_name')['elo_rating'].to_dict()
host_teams = {'Mexico', 'USA', 'Canada'}

elo_diffs = []
host_advs = []
for _, row in completed.iterrows():
    h, a = row['home_team_name'], row['away_team_name']
    diff = team_elo.get(h, 1700) - team_elo.get(a, 1700)
    hadv = int(h in host_teams) - int(a in host_teams)
    elo_diffs.append(diff); host_advs.append(hadv)
elo_diffs = np.array(elo_diffs)
host_advs = np.array(host_advs)

# ── Load Polymarket JSON ───────────────────────────────────────────────────
with open(POLY_JSON) as f:
    poly_data = json.load(f)

# Build lookup: (home_normalized, away_normalized) → entry
poly_dict = {}
for m in poly_data.get('group_matches', []):
    key = (normalize(m['home']), normalize(m['away']))
    poly_dict[key] = {'p_market': m.get('p_market'), 'p_model': m.get('p_model')}

# ── Assemble market_probs matrix [A, D, H] for all 64 matches ─────────────
market_probs = np.zeros((64, 3))
market_source = []
covered = []

for i, row in completed.iterrows():
    key = (row['home_team_name'], row['away_team_name'])
    entry = poly_dict.get(key)
    
    if entry and entry.get('p_market') is not None:
        pm = entry['p_market']
        market_probs[i] = [pm['A'], pm['D'], pm['H']]
        market_source.append('polymarket')
        covered.append(True)
    elif entry and entry.get('p_model') is not None:
        pm = entry['p_model']
        market_probs[i] = [pm['A'], pm['D'], pm['H']]
        market_source.append('p_model')
        covered.append(False)
    else:
        # Elo-logistic fallback (should not happen with proper normalization)
        diff = elo_diffs[i] + 70 * host_advs[i]
        p_h = 1 / (1 + 10 ** (-diff / 400))
        p_a = 1 / (1 + 10 ** (diff / 400))
        p_d = max(0.08, 1 - p_h - p_a)
        s = p_h + p_d + p_a
        market_probs[i] = [p_a/s, p_d/s, p_h/s]
        market_source.append('elo_fallback')
        covered.append(False)
        print(f"  FALLBACK for match {i}: {key}")

covered = np.array(covered)
src_counts = pd.Series(market_source).value_counts().to_dict()
print(f"\nCoverage: {covered.sum()}/64 from Polymarket")
print(f"Sources:  {src_counts}")
print(f"Row sums: min={market_probs.sum(axis=1).min():.4f}, max={market_probs.sum(axis=1).max():.4f}")

# ── CV helpers ─────────────────────────────────────────────────────────────
RSKF = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)

def eval_fixed_probs(probs, y, label=''):
    """Evaluate fixed (non-CV-trained) probability predictions."""
    fold_losses = []
    oof = np.zeros_like(probs)
    oof_cnt = np.zeros(len(y))
    
    for tr_idx, te_idx in RSKF.split(probs, y):
        fold_losses.append(log_loss(y[te_idx], probs[te_idx]))
        oof[te_idx] += probs[te_idx]
        oof_cnt[te_idx] += 1
    
    oof /= oof_cnt[:, None]
    mean_ll = np.mean(fold_losses)
    std_ll  = np.std(fold_losses)
    acc     = np.mean(np.argmax(oof, axis=1) == y)
    print(f"  {label}: log-loss={mean_ll:.4f} ± {std_ll:.4f}, acc={acc:.4f}")
    return mean_ll, std_ll, np.array(fold_losses), oof

def eval_cv_logistic(X, y, C=1.0, label=''):
    """Train logistic inside each fold and evaluate OOF."""
    fold_losses = []
    oof = np.zeros((len(y), 3))
    oof_cnt = np.zeros(len(y))
    
    for tr_idx, te_idx in RSKF.split(X, y):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]
        
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
        
        clf = LogisticRegression(solver='lbfgs', C=C, max_iter=1000, random_state=SEED)
        clf.fit(Xtr_s, ytr)
        probs = clf.predict_proba(Xte_s)
        
        fold_losses.append(log_loss(yte, probs))
        oof[te_idx] += probs
        oof_cnt[te_idx] += 1
    
    oof /= oof_cnt[:, None]
    mean_ll = np.mean(fold_losses)
    std_ll  = np.std(fold_losses)
    acc     = np.mean(np.argmax(oof, axis=1) == y)
    print(f"  {label}: log-loss={mean_ll:.4f} ± {std_ll:.4f}, acc={acc:.4f}")
    return mean_ll, std_ll, np.array(fold_losses), oof

def wilcoxon_paired(folds_a, folds_b, name_a, name_b):
    delta = np.mean(folds_a) - np.mean(folds_b)
    try:
        stat, p = stats.wilcoxon(folds_a, folds_b, alternative='less')
    except:
        p = 1.0
    verdict = "GREEN" if p < 0.05 else "FLAT"
    print(f"    {name_a} < {name_b}: Δ={delta:+.4f}, p={p:.4f} → {verdict}")
    return p, delta

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Elo baseline (canonical)
print("\n══ EXP 1: Elo-Logistic Baseline ══")
X_elo = np.column_stack([elo_diffs, host_advs])
elo_mean, elo_std, elo_folds, elo_oof = eval_cv_logistic(X_elo, y, C=1.0, label='Elo-logistic (C=1)')

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Market fixed predictions (64/64 with p_model imputation)
print("\n══ EXP 2: Market Fixed Predictions ══")
mkt_mean, mkt_std, mkt_folds, mkt_oof = eval_fixed_probs(market_probs, y, label='Market 64/64 (poly+p_model)')

# Only Polymarket 56 covered matches (base-rate for 8 missing)
base_rates = np.array([15/64, 18/64, 31/64])
market_56 = market_probs.copy()
for i in range(64):
    if not covered[i]:
        market_56[i] = base_rates
mkt56_mean, mkt56_std, mkt56_folds, mkt56_oof = eval_fixed_probs(market_56, y, label='Market 56/64 (base-rate for 8)')

print("\n  Significance vs Elo baseline:")
wilcoxon_paired(mkt_folds, elo_folds, 'Market-64', 'Elo')
wilcoxon_paired(mkt56_folds, elo_folds, 'Market-56', 'Elo')

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Market + Elo logistic blend (CV)
print("\n══ EXP 3: Market + Elo Logistic Blend (CV) ══")
eps = 1e-9
mkt_log = np.log(np.clip(market_probs, eps, 1-eps))  # log-space probs

# Feature set A: all 3 market log-probs + elo + host
Xa = np.column_stack([mkt_log, elo_diffs, host_advs])
blendA_mean, blendA_std, blendA_folds, blendA_oof = eval_cv_logistic(Xa, y, C=0.1, label='Market-3logp+Elo (C=0.1)')

# Feature set B: market log-odds ratio H/A + draw term + elo_diff only
logHA = mkt_log[:, 2] - mkt_log[:, 0]   # log(p_H/p_A)
logD  = mkt_log[:, 1]                    # log(p_D)
Xb = np.column_stack([logHA, logD, elo_diffs])
blendB_mean, blendB_std, blendB_folds, blendB_oof = eval_cv_logistic(Xb, y, C=0.1, label='Market-logHA+logD+Elo (C=0.1)')

# Feature set C: market probs direct + elo
Xc = np.column_stack([market_probs, elo_diffs])
blendC_mean, blendC_std, blendC_folds, blendC_oof = eval_cv_logistic(Xc, y, C=0.1, label='Market-probs+Elo (C=0.1)')

print("\n  Significance vs Elo baseline:")
wilcoxon_paired(blendA_folds, elo_folds, 'BlendA', 'Elo')
wilcoxon_paired(blendB_folds, elo_folds, 'BlendB', 'Elo')
wilcoxon_paired(blendC_folds, elo_folds, 'BlendC', 'Elo')

print("\n  Significance vs Market-only:")
wilcoxon_paired(blendA_folds, mkt_folds, 'BlendA', 'Market-64')
wilcoxon_paired(blendB_folds, mkt_folds, 'BlendB', 'Market-64')

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: De-vig method comparison (on 56 covered matches)
print("\n══ EXP 4: De-Vig Method Comparison (56 Polymarket matches) ══")
covered_idx = np.where(covered)[0]
y_cov = y[covered_idx]
poly_cov = market_probs[covered_idx]  # already proportional de-vigged by Polymarket

# Simulate raw decimal odds with 5% margin (overround = sum(implied) = 1.05)
raw_implied = poly_cov * 1.05   # inflate by margin

def proportional_devig(q):
    q = np.array(q)
    return q / q.sum()

def shin_devig(q):
    q = np.array(q, dtype=float)
    def eq(z):
        if abs(1-z) < 1e-10: return 1.0
        p = [(np.sqrt(z**2 + 4*(1-z)*qi**2) - z) / (2*(1-z)) for qi in q]
        return sum(p) - 1.0
    try:
        z = brentq(eq, 1e-10, 1 - 1e-10)
        p = np.array([(np.sqrt(z**2 + 4*(1-z)*qi**2) - z) / (2*(1-z)) for qi in q])
    except:
        p = q / q.sum()
    return p / p.sum()

prop_probs = np.array([proportional_devig(q) for q in raw_implied])
shin_probs = np.array([shin_devig(q) for q in raw_implied])

def fixed_56_eval(probs, y_56, label):
    eps = 1e-10
    p = np.clip(probs, eps, 1-eps)
    ll = np.mean([-np.log(p[i, y_56[i]]) for i in range(len(y_56))])
    acc = np.mean(np.argmax(probs, axis=1) == y_56)
    print(f"  {label}: ll={ll:.4f}, acc={acc:.4f}")
    return ll

ll_poly_orig = fixed_56_eval(poly_cov, y_cov, 'Polymarket (pre-de-vigged)')
ll_prop = fixed_56_eval(prop_probs, y_cov, 'Proportional (5% margin sim)')
ll_shin = fixed_56_eval(shin_probs, y_cov, 'Shin (5% margin sim)')
print(f"  Shin vs Proportional delta: {ll_shin - ll_prop:.6f} (on simulated margin)")
print(f"  Note: Polymarket is already de-vigged so methods should be ~equal")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE ARTIFACTS
print("\n══ SAVING ARTIFACTS ══")

oof_df = pd.DataFrame({
    'match_id': completed['match_id'].values,
    'home': completed['home_team_name'].values,
    'away': completed['away_team_name'].values,
    'label': completed['label'].values,
    'market_source': market_source,
    'mkt_p_A': market_probs[:, 0], 'mkt_p_D': market_probs[:, 1], 'mkt_p_H': market_probs[:, 2],
    'elo_oof_A': elo_oof[:, 0], 'elo_oof_D': elo_oof[:, 1], 'elo_oof_H': elo_oof[:, 2],
    'mkt64_oof_A': mkt_oof[:, 0], 'mkt64_oof_D': mkt_oof[:, 1], 'mkt64_oof_H': mkt_oof[:, 2],
    'blendA_oof_A': blendA_oof[:, 0], 'blendA_oof_D': blendA_oof[:, 1], 'blendA_oof_H': blendA_oof[:, 2],
})
oof_df.to_csv(f"{ART_DIR}/oof_probabilities.csv", index=False)

metrics = {
    'description': 'Wave-3 Full Market-Odds Coverage experiments',
    'n_matches': 64, 'n_folds': 50, 'cv': '5x10 RepeatedStratifiedKFold seed=0',
    'coverage': {'polymarket': int(covered.sum()), 'p_model_imputed': int((~covered).sum())},
    'results': {
        'elo_baseline':         {'mean': round(elo_mean, 4), 'std': round(elo_std, 4)},
        'market_64_fixed':      {'mean': round(mkt_mean, 4), 'std': round(mkt_std, 4)},
        'market_56_base_imp':   {'mean': round(mkt56_mean, 4), 'std': round(mkt56_std, 4)},
        'market_elo_blend_A':   {'mean': round(blendA_mean, 4), 'std': round(blendA_std, 4)},
        'market_elo_blend_B':   {'mean': round(blendB_mean, 4), 'std': round(blendB_std, 4)},
        'market_elo_blend_C':   {'mean': round(blendC_mean, 4), 'std': round(blendC_std, 4)},
    },
    'campaign_baselines': {'elo_logistic': 0.8337, 'wave2_ensemble': 0.7608},
}
with open(f"{ART_DIR}/exp01_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

run_info = {
    'script': 'code/market_experiment.py',
    'command': 'python3 code/market_experiment.py',
    'features': {
        'market_probs': 'Polymarket pre-match de-vigged 1X2 probs (56/64); p_model for 8 missing',
        'elo_diff': 'home_elo - away_elo from teams.csv',
        'host_adv': 'is_host flag (MEX/USA/CAN)',
    },
    'packages': {'sklearn': '1.9.0', 'scipy': 'installed', 'numpy': 'installed'},
    'seed': 0,
    'cv': '5-fold x 10-repeat RepeatedStratifiedKFold',
    'data_source_market': 'amirdaraee/world-cup-predictions GitHub (wc26_predictions.json)',
    'data_source_fifa': 'mominullptr/fifa-world-cup-2026-dataset',
}
with open(f"{ART_DIR}/exp01_run.json", 'w') as f:
    json.dump(run_info, f, indent=2)

print("\n══ FINAL SUMMARY ══")
print(f"Elo baseline:              {elo_mean:.4f} ± {elo_std:.4f}")
print(f"Market 64/64 (fixed):      {mkt_mean:.4f} ± {mkt_std:.4f}  Δ={mkt_mean-elo_mean:+.4f}")
print(f"Market 56/64 (base-imp):   {mkt56_mean:.4f} ± {mkt56_std:.4f}  Δ={mkt56_mean-elo_mean:+.4f}")
print(f"Market+Elo blend A:        {blendA_mean:.4f} ± {blendA_std:.4f}  Δ={blendA_mean-elo_mean:+.4f}")
print(f"Market+Elo blend B:        {blendB_mean:.4f} ± {blendB_std:.4f}  Δ={blendB_mean-elo_mean:+.4f}")
print(f"Market+Elo blend C:        {blendC_mean:.4f} ± {blendC_std:.4f}  Δ={blendC_mean-elo_mean:+.4f}")
print(f"Wave-2 frontier:           0.7608")
