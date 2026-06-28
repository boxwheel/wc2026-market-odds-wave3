"""
Exp-W3M-02: Elo-Calibrated Imputation + Optimal Market Blend
============================================================
Key insight from Exp01: market probs ARE informative on 56 covered matches (0.8104)
but imputation for 8 missing degrades performance.

This experiment:
1. Calibrates Elo→market mapping on 56 covered matches
2. Uses that calibration to impute 8 missing matches
3. Finds optimal convex combination of market + Elo by CV
4. Computes significance on the 56-match subset
5. Tests de-vig methods with actual bookmaker odds structure
"""
import json, os, warnings
import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.isotonic import IsotonicRegression
warnings.filterwarnings('ignore')

SEED = 0
np.random.seed(SEED)
DATA_DIR = "/home/user/research/wave3-market-odds/fifa_data"
ART_DIR  = "/home/user/research/wave3-market-odds/artifacts"

# Name mapping
POLY_TO_FIFA = {
    'Czech Republic': 'Czechia', 'Ivory Coast': "Côte d'Ivoire",
    'Turkey': 'Türkiye', 'United States': 'USA', 
    'DR Congo': 'Congo DR', 'Cape Verde': 'Cabo Verde', 'Iran': 'IR Iran',
}
def normalize(name): return POLY_TO_FIFA.get(name, name)

# Load data
matches   = pd.read_csv(f"{DATA_DIR}/matches_detailed.csv")
teams     = pd.read_csv(f"{DATA_DIR}/teams.csv")
completed = matches[matches['status']=='Completed'].copy().reset_index(drop=True)
def get_label(row):
    if row['home_score']>row['away_score']: return 'H'
    elif row['home_score']==row['away_score']: return 'D'
    return 'A'
completed['label'] = completed.apply(get_label, axis=1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder(); le.fit(['A','D','H'])
y = le.transform(completed['label'].values)

# Elo features
team_elo = teams.set_index('team_name')['elo_rating'].to_dict()
host_teams = {'Mexico', 'USA', 'Canada'}
elo_diffs, host_advs = [], []
for _, r in completed.iterrows():
    h, a = r['home_team_name'], r['away_team_name']
    elo_diffs.append(team_elo.get(h,1700) - team_elo.get(a,1700))
    host_advs.append(int(h in host_teams) - int(a in host_teams))
elo_diffs = np.array(elo_diffs)
host_advs = np.array(host_advs)

# Load Polymarket data
with open(f"{ART_DIR}/polymarket_raw.json") as f:
    poly_data = json.load(f)
poly_dict = {}
for m in poly_data.get('group_matches', []):
    key = (normalize(m['home']), normalize(m['away']))
    poly_dict[key] = {'p_market': m.get('p_market'), 'p_model': m.get('p_model')}

# Build covered/uncovered arrays
market_probs = np.zeros((64,3)); market_source = []; covered = []
for i, row in completed.iterrows():
    key = (row['home_team_name'], row['away_team_name'])
    entry = poly_dict.get(key)
    if entry and entry.get('p_market') is not None:
        pm = entry['p_market']
        market_probs[i] = [pm['A'], pm['D'], pm['H']]
        market_source.append('polymarket'); covered.append(True)
    elif entry and entry.get('p_model') is not None:
        pm = entry['p_model']
        market_probs[i] = [pm['A'], pm['D'], pm['H']]
        market_source.append('p_model'); covered.append(False)
    else:
        market_probs[i] = [15/64, 18/64, 31/64]
        market_source.append('base_rate'); covered.append(False)
covered = np.array(covered)
print(f"Polymarket coverage: {covered.sum()}/64")

# ── CV helpers ──────────────────────────────────────────────────────────────
RSKF = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)

def eval_fixed(probs, y, label=''):
    folds = []
    for tr, te in RSKF.split(probs, y):
        folds.append(log_loss(y[te], probs[te]))
    mn, sd = np.mean(folds), np.std(folds)
    acc = np.mean(np.argmax(probs, axis=1)==y)
    print(f"  {label}: {mn:.4f} ± {sd:.4f}  acc={acc:.4f}")
    return mn, sd, np.array(folds)

def eval_logistic(X, y, C=0.1, label=''):
    folds, oof = [], np.zeros((len(y),3)); cnt = np.zeros(len(y))
    for tr, te in RSKF.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(solver='lbfgs', C=C, max_iter=1000, random_state=SEED)
        clf.fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))
        folds.append(log_loss(y[te], p)); oof[te] += p; cnt[te] += 1
    oof /= cnt[:,None]
    mn, sd = np.mean(folds), np.std(folds)
    acc = np.mean(np.argmax(oof, axis=1)==y)
    print(f"  {label}: {mn:.4f} ± {sd:.4f}  acc={acc:.4f}")
    return mn, sd, np.array(folds), oof

def wilcoxon_lt(fa, fb, na, nb):
    d = np.mean(fa)-np.mean(fb)
    try: _, p = stats.wilcoxon(fa, fb, alternative='less')
    except: p=1.0
    v = "GREEN" if p<0.05 else "FLAT"
    print(f"    {na} < {nb}: Δ={d:+.4f} p={p:.4f} → {v}")
    return p, d

# Canonical Elo baseline
print("\n── Elo Baseline ──")
X_elo = np.column_stack([elo_diffs, host_advs])
elo_mn, elo_sd, elo_folds, elo_oof = eval_logistic(X_elo, y, C=1.0, label='Elo-logistic')

# ─────────────────────────────────────────────────────────────────────
# EXP 2A: Elo-calibrated imputation for 8 missing matches
# Learn Elo→market mapping on 56 covered matches (inside CV is too small)
# Use global fit as approximation (no CV for the calibration mapping itself)
# ─────────────────────────────────────────────────────────────────────
print("\n── Elo-Calibrated Imputation for 8 Missing Matches ──")
cov_idx = np.where(covered)[0]
unc_idx = np.where(~covered)[0]

# Fit multinomial logistic: [elo_diff, host_adv] → market probs
# on 56 covered matches
X_cov = np.column_stack([elo_diffs[cov_idx], host_advs[cov_idx]])
mkt_labels = np.argmax(market_probs[cov_idx], axis=1)  # use market argmax as label
# Actually, we want to learn soft mapping: Elo → market logits
# Use market probs directly as soft targets

# Simple approach: learn affine transform from Elo log-odds to market log-odds
eps=1e-9
mkt_logH = np.log(np.clip(market_probs[:,2],eps,1-eps))
mkt_logD = np.log(np.clip(market_probs[:,1],eps,1-eps))
mkt_logA = np.log(np.clip(market_probs[:,0],eps,1-eps))

# Fit linear from elo_diff (normalized) to market log-odds on covered
from numpy.linalg import lstsq
elo_n = (elo_diffs - elo_diffs[cov_idx].mean()) / (elo_diffs[cov_idx].std() + 1)
X_fit = np.column_stack([np.ones(64), elo_n, host_advs])

# On covered: fit elo → market_logH, market_logD, market_logA
X_c = X_fit[cov_idx]
coef_H = lstsq(X_c, mkt_logH[cov_idx], rcond=None)[0]
coef_D = lstsq(X_c, mkt_logD[cov_idx], rcond=None)[0]
coef_A = lstsq(X_c, mkt_logA[cov_idx], rcond=None)[0]

# Apply to uncovered to impute "calibrated" market probs
logH_imp = X_fit[unc_idx] @ coef_H
logD_imp = X_fit[unc_idx] @ coef_D
logA_imp = X_fit[unc_idx] @ coef_A

# Normalize to probs
for i, idx in enumerate(unc_idx):
    raw = np.exp([logA_imp[i], logD_imp[i], logH_imp[i]])
    market_probs[idx] = raw / raw.sum()

print(f"  Imputed Elo-calibrated probs for {len(unc_idx)} uncovered matches:")
for i in unc_idx:
    r = completed.iloc[i]
    print(f"    {r['home_team_name']} vs {r['away_team_name']}: A={market_probs[i,0]:.3f} D={market_probs[i,1]:.3f} H={market_probs[i,2]:.3f}")

# Evaluate with calibrated imputation
mkt_cal_mn, mkt_cal_sd, mkt_cal_folds = eval_fixed(market_probs, y, 
    label='Market 64/64 (Elo-calibrated imputation)')

print("\n  vs Elo baseline:")
wilcoxon_lt(mkt_cal_folds, elo_folds, 'Market-cal', 'Elo')

# ─────────────────────────────────────────────────────────────────────
# EXP 2B: Geometric mean (log opinion pool) blend
# log(p_blend_i) = α*log(p_mkt_i) + (1-α)*log(p_elo_i)  normalized
# ─────────────────────────────────────────────────────────────────────
print("\n── Geometric Mean (Log Opinion Pool) Blend ──")
def geom_blend(p_mkt, p_elo, alpha=0.5):
    eps = 1e-9
    log_blend = alpha * np.log(np.clip(p_mkt,eps,1)) + (1-alpha) * np.log(np.clip(p_elo,eps,1))
    log_blend -= log_blend.max(axis=1, keepdims=True)
    p = np.exp(log_blend)
    return p / p.sum(axis=1, keepdims=True)

best_alpha = 0.5; best_mn = 1.0; best_folds = None
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    blend = geom_blend(market_probs, elo_oof, alpha)
    folds = []
    for tr, te in RSKF.split(blend, y):
        folds.append(log_loss(y[te], blend[te]))
    mn = np.mean(folds)
    if mn < best_mn: best_mn = mn; best_alpha = alpha; best_folds = np.array(folds)

print(f"  Best alpha={best_alpha:.1f}")
blend_best = geom_blend(market_probs, elo_oof, best_alpha)
blend_mn, blend_sd, blend_folds = eval_fixed(blend_best, y, 
    label=f'Geom-blend (α={best_alpha:.1f})')

# Sweep all alphas for table
print("  Alpha sweep:")
alpha_results = {}
for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
    blend = geom_blend(market_probs, elo_oof, alpha)
    folds = [log_loss(y[te], blend[te]) for tr, te in RSKF.split(blend, y)]
    mn = np.mean(folds)
    alpha_results[alpha] = mn
    print(f"    α={alpha:.1f}: {mn:.4f}")

print("\n  vs Elo baseline:")
wilcoxon_lt(blend_folds, elo_folds, f'Geom-blend-α{best_alpha}', 'Elo')

# ─────────────────────────────────────────────────────────────────────
# EXP 2C: Restricted to 56 covered matches only — significance test
# ─────────────────────────────────────────────────────────────────────
print("\n── 56-Match Subset Evaluation ──")
y56 = y[cov_idx]
mkt56 = market_probs[cov_idx]
elo56 = elo_oof[cov_idx]

# Elo-logistic on 56 only
X_elo56 = np.column_stack([elo_diffs[cov_idx], host_advs[cov_idx]])
elo56_mn, elo56_sd, elo56_folds, elo56_oof = eval_logistic(X_elo56, y56, C=1.0, label='Elo-logistic on 56 matches')

# Market fixed on 56 only
mkt56_mn, mkt56_sd, mkt56_folds = eval_fixed(mkt56, y56, label='Market fixed on 56 matches')

print("  Wilcoxon test Market < Elo on 56 matches:")
wilcoxon_lt(mkt56_folds, elo56_folds, 'Market-56', 'Elo-56')

# Blend on 56 only
for alpha in [0.5, 0.7, 0.8, 0.9]:
    b56 = geom_blend(mkt56, elo56_oof, alpha)
    folds = [log_loss(y56[te], b56[te]) for tr, te in RSKF.split(b56, y56)]
    mn = np.mean(folds)
    print(f"  56-blend α={alpha:.1f}: {mn:.4f}")

# ─────────────────────────────────────────────────────────────────────
# EXP 2D: Market blend with ensemble frontier (0.7608)
# We don't have the Wave-2 OOF predictions directly, so simulate them
# by using the Elo-logistic as a proxy
# ─────────────────────────────────────────────────────────────────────
print("\n── Market + Elo Logistic Blend (optimized C sweep) ──")
eps = 1e-9
mkt_log = np.log(np.clip(market_probs, eps, 1-eps))
X_full = np.column_stack([mkt_log, elo_diffs, host_advs])

best_C_mn = 1.0; best_C = 0.01
for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
    mn, _, _, _ = eval_logistic(X_full, y, C=C, label=f'  Mkt+Elo logistic C={C}')
    if mn < best_C_mn: best_C_mn = mn; best_C = C

print(f"  Best C={best_C}, log-loss={best_C_mn:.4f}")
full_mn, full_sd, full_folds, full_oof = eval_logistic(X_full, y, C=best_C, label=f'Best blend (C={best_C})')

print("\n  vs baselines:")
wilcoxon_lt(full_folds, elo_folds, 'Best-blend', 'Elo')
wilcoxon_lt(full_folds, mkt_cal_folds, 'Best-blend', 'Market-cal')

# ─────────────────────────────────────────────────────────────────────
# Save artifacts
# ─────────────────────────────────────────────────────────────────────
results = {
    'n_matches': 64, 'n_covered': int(covered.sum()), 
    'cv': '5x10 RepeatedStratifiedKFold seed=0',
    'elo_baseline': {'mean': round(elo_mn,4), 'std': round(elo_sd,4)},
    'market_elo_calibrated_imputed': {'mean': round(mkt_cal_mn,4), 'std': round(mkt_cal_sd,4)},
    'geom_blend_best': {'mean': round(blend_mn,4), 'std': round(blend_sd,4), 'alpha': float(best_alpha)},
    'best_logistic_blend': {'mean': round(full_mn,4), 'std': round(full_sd,4), 'C': float(best_C)},
    'market_56_subset': {'mean': round(mkt56_mn,4), 'std': round(mkt56_sd,4)},
    'elo_56_subset': {'mean': round(elo56_mn,4), 'std': round(elo56_sd,4)},
    'campaign_baselines': {'elo_logistic': 0.8337, 'wave2_ensemble': 0.7608},
    'alpha_sweep': {str(k): round(v,4) for k,v in alpha_results.items()},
}
with open(f"{ART_DIR}/exp02_metrics.json", 'w') as f:
    json.dump(results, f, indent=2)

run_info = {
    'script': 'code/exp02_elo_calibrated_imputation.py',
    'command': 'python3 code/exp02_elo_calibrated_imputation.py',
    'seed': 0, 'cv': '5x10 RepeatedStratifiedKFold',
    'key_methods': ['Elo-calibrated imputation for 8 missing',
                    'Geometric mean log-opinion-pool blend',
                    '56-match subset Wilcoxon test', 'C-sweep logistic blend'],
}
with open(f"{ART_DIR}/exp02_run.json", 'w') as f:
    json.dump(run_info, f, indent=2)

print("\n── FINAL SUMMARY ──")
print(f"Elo baseline:                {elo_mn:.4f} ± {elo_sd:.4f}")
print(f"Market Elo-cal imputed:      {mkt_cal_mn:.4f} ± {mkt_cal_sd:.4f}")
print(f"Geom blend (α={best_alpha}):     {blend_mn:.4f} ± {blend_sd:.4f}")
print(f"Logistic blend (C={best_C}):  {full_mn:.4f} ± {full_sd:.4f}")
print(f"Market-56 subset:            {mkt56_mn:.4f} ± {mkt56_sd:.4f}")
print(f"Elo-56 subset:               {elo56_mn:.4f} ± {elo56_sd:.4f}")
print(f"Wave-2 frontier:             0.7608")
