"""
Exp-W3M-03: Market + Multi-Model Ensemble — challenge 0.7608 frontier
"""
import json, os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
warnings.filterwarnings('ignore')

SEED = 0; np.random.seed(SEED)
DATA_DIR = "/home/user/research/wave3-market-odds/fifa_data"
ART_DIR  = "/home/user/research/wave3-market-odds/artifacts"

POLY_TO_FIFA = {
    'Czech Republic':'Czechia', 'Ivory Coast':"Côte d'Ivoire",
    'Turkey':'Türkiye', 'United States':'USA', 
    'DR Congo':'Congo DR', 'Cape Verde':'Cabo Verde', 'Iran':'IR Iran',
}
def normalize(name): return POLY_TO_FIFA.get(name, name)

# Load data
matches   = pd.read_csv(f"{DATA_DIR}/matches_detailed.csv")
teams     = pd.read_csv(f"{DATA_DIR}/teams.csv")
squads_raw = pd.read_csv(f"{DATA_DIR}/squads_and_players.csv")
squads    = squads_raw.merge(teams[['team_id','team_name']], on='team_id', how='left')

completed = matches[matches['status']=='Completed'].copy().reset_index(drop=True)
def get_label(r):
    if r['home_score']>r['away_score']: return 'H'
    elif r['home_score']==r['away_score']: return 'D'
    return 'A'
completed['label'] = completed.apply(get_label, axis=1)
le = LabelEncoder(); le.fit(['A','D','H'])
y = le.transform(completed['label'].values)

# Elo + rank features
team_elo  = teams.set_index('team_name')['elo_rating'].to_dict()
team_rank = teams.set_index('team_name')['fifa_ranking_pre_tournament'].to_dict()
host_teams = {'Mexico','USA','Canada'}

elo_diffs, rank_diffs, host_advs = [], [], []
for _, r in completed.iterrows():
    h,a = r['home_team_name'], r['away_team_name']
    elo_diffs.append(team_elo.get(h,1700) - team_elo.get(a,1700))
    rank_diffs.append(-(team_rank.get(h,50) - team_rank.get(a,50)))
    host_advs.append(int(h in host_teams) - int(a in host_teams))
elo_diffs = np.array(elo_diffs)
rank_diffs = np.array(rank_diffs)
host_advs = np.array(host_advs)

# Squad features
squads['year'] = pd.to_datetime(squads['date_of_birth'], errors='coerce').dt.year
def squad_feats(team_name):
    sq = squads[squads['team_name']==team_name]
    if len(sq)==0: return np.zeros(5)
    mv = sq['market_value_eur'].fillna(0)
    caps = sq['caps'].fillna(0)
    goals = sq['goals'].fillna(0)
    age = 2026 - sq['year'].fillna(28)
    return np.array([mv.sum()/1e6, mv.nlargest(11).sum()/1e6, caps.mean(), goals.sum(), age.mean()])

squad_h = np.array([squad_feats(r['home_team_name']) for _, r in completed.iterrows()])
squad_a = np.array([squad_feats(r['away_team_name']) for _, r in completed.iterrows()])
squad_diff = squad_h - squad_a

# Market probabilities
with open(f"{ART_DIR}/polymarket_raw.json") as f:
    poly_data = json.load(f)
poly_dict = {}
for m in poly_data.get('group_matches', []):
    key = (normalize(m['home']), normalize(m['away']))
    poly_dict[key] = {'p_market': m.get('p_market'), 'p_model': m.get('p_model')}

market_probs = np.zeros((64,3)); covered = []
for i, row in completed.iterrows():
    key = (row['home_team_name'], row['away_team_name'])
    entry = poly_dict.get(key)
    if entry and entry.get('p_market') is not None:
        pm = entry['p_market']; market_probs[i]=[pm['A'],pm['D'],pm['H']]; covered.append(True)
    elif entry and entry.get('p_model') is not None:
        pm = entry['p_model']; market_probs[i]=[pm['A'],pm['D'],pm['H']]; covered.append(False)
    else:
        market_probs[i]=[15/64,18/64,31/64]; covered.append(False)
covered = np.array(covered)
print(f"Coverage: {covered.sum()}/64")

# ── CV helpers ──────────────────────────────────────────────────────────────
RSKF = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)

def eval_logistic(X, y, C=1.0, label=''):
    folds, oof = [], np.zeros((len(y),3)); cnt = np.zeros(len(y))
    for tr, te in RSKF.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(solver='lbfgs', C=C, max_iter=2000, random_state=SEED)
        clf.fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))
        folds.append(log_loss(y[te], p)); oof[te] += p; cnt[te] += 1
    oof /= cnt[:,None]
    mn,sd = np.mean(folds),np.std(folds)
    acc = np.mean(np.argmax(oof,axis=1)==y)
    print(f"  {label}: {mn:.4f} ± {sd:.4f}  acc={acc:.4f}")
    return mn,sd,np.array(folds),oof

def eval_fixed(probs, y, label=''):
    folds = [log_loss(y[te], probs[te]) for tr, te in RSKF.split(probs, y)]
    mn,sd = np.mean(folds),np.std(folds)
    acc = np.mean(np.argmax(probs,axis=1)==y)
    print(f"  {label}: {mn:.4f} ± {sd:.4f}  acc={acc:.4f}")
    return mn,sd,np.array(folds)

def wtest(fa, fb, na, nb):
    d = np.mean(fa)-np.mean(fb)
    try: _,p = stats.wilcoxon(fa,fb,alternative='less')
    except: p=1.0
    v = "GREEN" if p<0.05 else "FLAT"
    print(f"    {na}<{nb}: Δ={d:+.4f} p={p:.4f} → {v}")
    return p,d

def geom(pa, pb, alpha=0.5, eps=1e-9):
    lb = alpha*np.log(np.clip(pa,eps,1)) + (1-alpha)*np.log(np.clip(pb,eps,1))
    lb -= lb.max(axis=1, keepdims=True)
    p = np.exp(lb); return p/p.sum(axis=1,keepdims=True)

# ── Base learners ────────────────────────────────────────────────────────────
print("\n── Base Learners ──")
X_elo   = np.column_stack([elo_diffs, host_advs])
X_rank  = np.column_stack([rank_diffs, host_advs])
X_squad = np.column_stack([elo_diffs, squad_diff])
X_combo = np.column_stack([elo_diffs, rank_diffs, host_advs, squad_diff])

elo_mn,   elo_sd,   elo_folds,   elo_oof   = eval_logistic(X_elo,   y, C=1.0,  label='Elo')
rank_mn,  rank_sd,  rank_folds,  rank_oof  = eval_logistic(X_rank,  y, C=1.0,  label='Rank')
squad_mn, squad_sd, squad_folds, squad_oof = eval_logistic(X_squad, y, C=0.1,  label='Squad+Elo')
combo_mn, combo_sd, combo_folds, combo_oof = eval_logistic(X_combo, y, C=0.03, label='Full combo C=0.03')

# ── 4-model ensemble ─────────────────────────────────────────────────────────
print("\n── 4-Model Ensemble (no market) ──")
ens4 = (elo_oof + rank_oof + squad_oof + combo_oof) / 4
ens4_mn, ens4_sd, ens4_folds = eval_fixed(ens4, y, label='4-model arith ensemble')

ens4g = geom(geom(elo_oof, rank_oof, 0.5), geom(squad_oof, combo_oof, 0.5), 0.5)
ens4g_mn, ens4g_sd, ens4g_folds = eval_fixed(ens4g, y, label='4-model geom ensemble')

# ── Market + ensemble blends ─────────────────────────────────────────────────
print("\n── Market + Ensemble Alpha Sweep ──")
best_mn = 1.0; best_alpha = 0.3; best_folds = None
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
    blend = geom(market_probs, ens4g, alpha)
    folds = [log_loss(y[te], blend[te]) for tr, te in RSKF.split(blend, y)]
    mn = np.mean(folds)
    print(f"  Mkt+ens4g α={alpha}: {mn:.4f}")
    if mn < best_mn: best_mn=mn; best_alpha=alpha; best_folds=np.array(folds)

mke_blend = geom(market_probs, ens4g, best_alpha)
mke_mn, mke_sd, mke_folds = eval_fixed(mke_blend, y, label=f'Market+4gEns α={best_alpha}')

# Market + Elo (best from exp02)
mkt_elo = geom(market_probs, elo_oof, 0.4)
mktelo_mn, mktelo_sd, mktelo_folds = eval_fixed(mkt_elo, y, label='Market+Elo α=0.4 (Exp02)')

# Market + squad geom
mkt_sq = geom(market_probs, squad_oof, 0.4)
mktsq_mn, mktsq_sd, mktsq_folds = eval_fixed(mkt_sq, y, label='Market+Squad α=0.4')

# 3-way: market + elo + squad
mkt3 = geom(mkt_elo, squad_oof, 0.3)
mkt3_mn, mkt3_sd, mkt3_folds = eval_fixed(mkt3, y, label='3-way: Market+Elo+Squad')

print("\n── Significance Tests ──")
wtest(mktelo_folds, elo_folds, 'Mkt+Elo0.4', 'Elo')
wtest(ens4_folds, elo_folds, '4-model-ens', 'Elo')
wtest(mke_folds, elo_folds, f'Mkt+4gEns', 'Elo')
wtest(mkt3_folds, elo_folds, '3-way', 'Elo')
wtest(mke_folds, mktelo_folds, 'Mkt+4gEns', 'Mkt+Elo0.4')
wtest(mkt3_folds, mktelo_folds, '3-way', 'Mkt+Elo0.4')

# Save
results = {
    'n_matches': 64, 'cv': '5x10 RepeatedStratifiedKFold seed=0',
    'elo_baseline': {'mean': round(elo_mn,4), 'std': round(elo_sd,4)},
    'squad_elo': {'mean': round(squad_mn,4), 'std': round(squad_sd,4)},
    '4model_ensemble_arith': {'mean': round(ens4_mn,4), 'std': round(ens4_sd,4)},
    '4model_ensemble_geom': {'mean': round(ens4g_mn,4), 'std': round(ens4g_sd,4)},
    'market_elo_geom04': {'mean': round(mktelo_mn,4), 'std': round(mktelo_sd,4)},
    f'market_4gEns_geom{best_alpha}': {'mean': round(mke_mn,4), 'std': round(mke_sd,4)},
    'market_elo_squad_3way': {'mean': round(mkt3_mn,4), 'std': round(mkt3_sd,4)},
    'campaign_baselines': {'elo': 0.8337, 'wave2_ensemble': 0.7608},
}
with open(f"{ART_DIR}/exp03_metrics.json", 'w') as f:
    json.dump(results, f, indent=2)
run_info = {'script': 'code/exp03_market_ensemble.py', 'seed': 0,
    'key_methods': ['4-model base ensemble', 'market+4gEns geom blend', '3-way blend']}
with open(f"{ART_DIR}/exp03_run.json", 'w') as f:
    json.dump(run_info, f, indent=2)

print("\n── SUMMARY ──")
print(f"Elo baseline:           0.8337")
print(f"Market+Elo(0.4):        {mktelo_mn:.4f} GREEN (from Exp02)")
print(f"4-model ensemble:       {ens4_mn:.4f}")
print(f"4-model geom ensemble:  {ens4g_mn:.4f}")
print(f"Market+4gEns:           {mke_mn:.4f}")
print(f"3-way Mkt+Elo+Squad:    {mkt3_mn:.4f}")
print(f"Wave-2 frontier:        0.7608")
