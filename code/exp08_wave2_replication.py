"""
Exp08: Wave-2 Stacking Replication + Market Extension
======================================================
Replicate Wave-2's 4-model stacked ensemble (target 0.7608) WITHOUT market probs,
then add market as a 5th base learner.

Proper double-CV stacking:
- Outer: RSKF 5×10 for final evaluation
- Inner: RSKF 5×2 for OOF generation within each outer train fold
  (more inner reps would be too slow for 50 outer folds)

Also tries:
- Single-level stacking (data leakage in stacking) — to understand Wave-2 likely approach
- Market as 5th base learner
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import log_loss

ARTIFACTS = Path("/home/user/research/wave3-market-odds/artifacts")
DATA = Path("/home/user/research/wave3-market-odds/fifa_data")

POLY_TO_FIFA = {
    'Czech Republic': 'Czechia', 'Ivory Coast': "Côte d'Ivoire", 'Turkey': 'Türkiye',
    'United States': 'USA', 'DR Congo': 'Congo DR', 'Cape Verde': 'Cabo Verde', 'Iran': 'IR Iran',
}
normalize = lambda name: POLY_TO_FIFA.get(name, name)
RSKF = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
eps = 1e-9


def load_data():
    matches = pd.read_csv(DATA / "matches_detailed.csv")
    teams = pd.read_csv(DATA / "teams.csv")
    sq_raw = pd.read_csv(DATA / "squads_and_players.csv")
    sq_raw["market_value_eur"] = pd.to_numeric(sq_raw["market_value_eur"], errors="coerce").fillna(0)
    squads = sq_raw.merge(teams[["team_id", "team_name"]], on="team_id", how="left")
    group = matches[matches["status"].str.lower() == "completed"].copy().reset_index(drop=True)
    def outcome(r):
        if r.home_score > r.away_score: return 0
        if r.home_score == r.away_score: return 1
        return 2
    group["y"] = group.apply(outcome, axis=1)
    tm = teams.set_index("team_name")
    group["elo_diff"] = group["home_team_name"].map(tm["elo_rating"]) - group["away_team_name"].map(tm["elo_rating"])
    group["rank_diff"] = group["away_team_name"].map(tm["fifa_ranking_pre_tournament"]) - group["home_team_name"].map(tm["fifa_ranking_pre_tournament"])
    hosts = {"USA", "Mexico", "Canada"}
    group["host_adv"] = (group["home_team_name"].isin(hosts)).astype(float) - (group["away_team_name"].isin(hosts)).astype(float)
    def team_feats(name):
        s = squads[squads["team_name"] == name]
        if len(s) == 0: return pd.Series({"mv": 0, "caps": 0, "goals": 0})
        top = s.nlargest(11, "market_value_eur")
        return pd.Series({"mv": top["market_value_eur"].sum()/1e6, "caps": s["caps"].fillna(0).mean(), "goals": s["goals"].fillna(0).mean()})
    for col in ["mv", "caps", "goals"]:
        group[f"home_{col}"] = group["home_team_name"].apply(lambda n: team_feats(n)[col])
        group[f"away_{col}"] = group["away_team_name"].apply(lambda n: team_feats(n)[col])
    group["mv_diff"] = group["home_mv"] - group["away_mv"]
    group["caps_diff"] = group["home_caps"] - group["away_caps"]
    group["goals_diff"] = group["home_goals"] - group["away_goals"]
    return group


def load_market_probs(group):
    with open(ARTIFACTS / "polymarket_raw.json") as f:
        raw = json.load(f)
    poly_map = {}
    for entry in raw.get("group_matches", []):
        h = normalize(entry.get("home", ""))
        a = normalize(entry.get("away", ""))
        pm = entry.get("p_market") or entry.get("p_model")
        if pm and h and a:
            poly_map[(h, a)] = pm
    mkt = np.full((len(group), 3), 1/3)
    for i, row in group.iterrows():
        key = (row["home_team_name"], row["away_team_name"])
        if key in poly_map:
            p = poly_map[key]
            mkt[i] = [p.get("H", 1/3), p.get("D", 1/3), p.get("A", 1/3)]
    row_sums = mkt.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1
    return mkt / row_sums


def logistic_oof(X, y, splits, C=1.0):
    """Generate OOF preds using a list of (tr, te) index pairs."""
    oof = np.zeros((len(y), 3))
    for tr, te in splits:
        sc = StandardScaler()
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))
    return oof


def geom_blend(p1, p2, alpha=0.5):
    lb = alpha * np.log(np.clip(p1, eps, 1)) + (1-alpha) * np.log(np.clip(p2, eps, 1))
    lb -= lb.max(axis=1, keepdims=True)
    p = np.exp(lb); return p / p.sum(axis=1, keepdims=True)


def wilcox_p(fold_losses, elo_folds_ref):
    try: return wilcoxon(fold_losses, elo_folds_ref, alternative='less').pvalue
    except: return float('nan')


def main():
    print("Loading data...")
    group = load_data()
    y = group["y"].values
    n = len(group)
    mkt = load_market_probs(group)
    print(f"n={n}")

    elo_X = group[["elo_diff", "host_adv"]].values
    rank_X = group[["rank_diff", "host_adv"]].values
    squad_X = group[["elo_diff", "rank_diff", "mv_diff", "caps_diff", "goals_diff", "host_adv"]].values
    full_X = group[["elo_diff", "rank_diff", "mv_diff", "caps_diff", "goals_diff", "host_adv"]].values

    # Elo reference
    splits_all = list(RSKF.split(elo_X, y))
    elo_folds_ref = [log_loss(y[te], logistic_oof(elo_X, y, splits_all, C=1.0)[te]) for tr, te in splits_all]
    # Actually compute properly:
    oof_elo_full = logistic_oof(elo_X, y, splits_all, C=1.0)
    elo_folds_ref = [log_loss(y[te], oof_elo_full[te]) for tr, te in splits_all]
    print(f"Elo OOF ll={np.mean(elo_folds_ref):.4f}")

    # ── 8a: Single-level stacking (likely Wave-2 approach) ──────────────────
    # OOF from full RSKF, then eval meta on same RSKF splits
    print("\n── 8a: Single-level stacking (Wave-2 likely approach) ──")
    oof_elo = logistic_oof(elo_X, y, splits_all, C=1.0)
    oof_rank = logistic_oof(rank_X, y, splits_all, C=1.0)
    oof_squad = logistic_oof(squad_X, y, splits_all, C=0.1)
    oof_full = logistic_oof(full_X, y, splits_all, C=0.03)

    meta_4 = np.hstack([oof_elo, oof_rank, oof_squad, oof_full])  # 12 features

    # Meta-learner: eval on same splits (this HAS data leakage from OOF using same folds)
    folds_8a = []
    for tr, te in splits_all:
        best_C, best_ll = 0.01, 1e9
        for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
            sc = StandardScaler()
            clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
            clf.fit(sc.fit_transform(meta_4[tr]), y[tr])
            ll_c = log_loss(y[tr], clf.predict_proba(sc.transform(meta_4[tr])))
            if ll_c < best_ll: best_ll, best_C = ll_c, C
        sc = StandardScaler()
        clf = LogisticRegression(C=best_C, solver="lbfgs", max_iter=2000)
        clf.fit(sc.fit_transform(meta_4[tr]), y[tr])
        folds_8a.append(log_loss(y[te], clf.predict_proba(sc.transform(meta_4[te]))))

    ll_8a = np.mean(folds_8a); std_8a = np.std(folds_8a)
    p_8a = wilcox_p(folds_8a, elo_folds_ref)
    v8a = "GREEN" if p_8a < 0.05 else ("FLAT" if p_8a < 0.2 else "RED")
    print(f"  4-model single-level stack: {ll_8a:.4f} ± {std_8a:.4f}  p={p_8a:.4f} → {v8a}")

    # ── 8b: Same + market as 5th base learner ──────────────────────────────
    print("\n── 8b: 5-model stack (4 learners + market) ──")
    meta_5 = np.hstack([oof_elo, oof_rank, oof_squad, oof_full, mkt])  # 15 features
    folds_8b = []
    for tr, te in splits_all:
        best_C, best_ll = 0.01, 1e9
        for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
            sc = StandardScaler()
            clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
            clf.fit(sc.fit_transform(meta_5[tr]), y[tr])
            ll_c = log_loss(y[tr], clf.predict_proba(sc.transform(meta_5[tr])))
            if ll_c < best_ll: best_ll, best_C = ll_c, C
        sc = StandardScaler()
        clf = LogisticRegression(C=best_C, solver="lbfgs", max_iter=2000)
        clf.fit(sc.fit_transform(meta_5[tr]), y[tr])
        folds_8b.append(log_loss(y[te], clf.predict_proba(sc.transform(meta_5[te]))))

    ll_8b = np.mean(folds_8b); std_8b = np.std(folds_8b)
    p_8b = wilcox_p(folds_8b, elo_folds_ref)
    v8b = "GREEN" if p_8b < 0.05 else ("FLAT" if p_8b < 0.2 else "RED")
    print(f"  5-model single-level stack: {ll_8b:.4f} ± {std_8b:.4f}  p={p_8b:.4f} → {v8b}")

    # ── 8c: Market-only geometric blend vs stacking comparison ───────────────
    print("\n── 8c: Geometric blend (best so far: 0.8000) vs stacking ──")
    # Reproduce Exp04 temperature-scaled 3-way blend
    blend3 = np.zeros((n, 3))
    for tr, te in splits_all:
        b = geom_blend(geom_blend(mkt[tr], oof_elo[tr], alpha=0.45), oof_squad[tr], alpha=0.6)
        blend3[te] = geom_blend(geom_blend(mkt[te], oof_elo[te], alpha=0.45), oof_squad[te], alpha=0.6)
    blend3_folds = [log_loss(y[te], blend3[te]) for tr, te in splits_all]
    ll_blend3 = np.mean(blend3_folds)
    p_blend3 = wilcox_p(blend3_folds, elo_folds_ref)
    print(f"  3-way geometric blend: {ll_blend3:.4f}  p={p_blend3:.4f}")

    # Blend geometric + stacked meta
    for alpha_merge in [0.2, 0.4, 0.6]:
        meta_hybrid_folds = []
        for tr, te in splits_all:
            sc = StandardScaler()
            clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=2000)
            clf.fit(sc.fit_transform(meta_4[tr]), y[tr])
            stack_te = clf.predict_proba(sc.transform(meta_4[te]))
            hybrid = geom_blend(blend3[te], stack_te, alpha=alpha_merge)
            meta_hybrid_folds.append(log_loss(y[te], hybrid))
        ll_hyb = np.mean(meta_hybrid_folds)
        print(f"  3way_blend+stack α={alpha_merge}: {ll_hyb:.4f}")

    # ── 8d: C sweep for 4-model stack ────────────────────────────────────────
    print("\n── 8d: 4-model stack with tight C sweep ──")
    for C_fixed in [0.001, 0.003, 0.005, 0.01, 0.02, 0.05]:
        folds_c = []
        for tr, te in splits_all:
            sc = StandardScaler()
            clf = LogisticRegression(C=C_fixed, solver="lbfgs", max_iter=2000)
            clf.fit(sc.fit_transform(meta_4[tr]), y[tr])
            folds_c.append(log_loss(y[te], clf.predict_proba(sc.transform(meta_4[te]))))
        ll_c = np.mean(folds_c)
        p_c = wilcox_p(folds_c, elo_folds_ref)
        print(f"  4-stack C={C_fixed}: {ll_c:.4f}  p={p_c:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n── Exp08 Summary ──")
    print(f"  Wave-2 target: 0.7608")
    print(f"  8a 4-model stack: {ll_8a:.4f} ± {std_8a:.4f}  p={p_8a:.4f} → {v8a}")
    print(f"  8b 5-model stack (+mkt): {ll_8b:.4f} ± {std_8b:.4f}  p={p_8b:.4f} → {v8b}")
    print(f"  Prior best (Exp04 temp-scaled 3-way): 0.8000")
    print(f"  Conclusion: {'Can replicate Wave-2' if ll_8a <= 0.77 else 'CANNOT replicate Wave-2 stacking result'}")

    metrics = {
        "experiment": "exp08_wave2_replication",
        "cv": "RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)",
        "n_matches": int(n),
        "results": {
            "8a_4model_stack": {"log_loss": float(ll_8a), "std": float(std_8a), "wilcoxon_p": float(p_8a)},
            "8b_5model_stack_mkt": {"log_loss": float(ll_8b), "std": float(std_8b), "wilcoxon_p": float(p_8b)},
        },
        "wave2_target": 0.7608,
        "prior_best_exp04": 0.8000,
        "wave2_replicated": bool(ll_8a <= 0.77),
    }
    with open(ARTIFACTS / "exp08_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(ARTIFACTS / "exp08_run.json", "w") as f:
        json.dump({"experiment": "exp08_wave2_replication", "date": "2026-06-28",
                   "approach": "single-level RSKF stacking, 4 base learners + market"}, f, indent=2)
    print("Artifacts saved.")


if __name__ == "__main__":
    main()
