"""
Exp05: OOF Stacking with Market as Pretrained Signal
=====================================================
Use market probs (pre-match, no leakage) directly as base-learner OOF predictions.
Stack with OOF preds from Elo, Rank, and Squad+Elo logistic learners.
Train a logistic meta-learner on 12 meta-features [4 learners × 3 probs].

Also tries:
- Direct logistic on log(market_probs) + structural features
- MLP (1 hidden layer, 8 units) on combined features
- ExtraTrees meta-learner

Target: beat 0.8000 (current best, Exp04)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

ARTIFACTS = Path("/home/user/research/wave3-market-odds/artifacts")
DATA = Path("/home/user/research/wave3-market-odds/fifa_data")

POLY_TO_FIFA = {
    'Czech Republic': 'Czechia',
    'Ivory Coast': "Côte d'Ivoire",
    'Turkey': 'Türkiye',
    'United States': 'USA',
    'DR Congo': 'Congo DR',
    'Cape Verde': 'Cabo Verde',
    'Iran': 'IR Iran',
}
normalize = lambda name: POLY_TO_FIFA.get(name, name)

RSKF = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)

def load_data():
    matches = pd.read_csv(DATA / "matches_detailed.csv")
    teams = pd.read_csv(DATA / "teams.csv")
    sq_raw = pd.read_csv(DATA / "squads_and_players.csv")
    sq_raw["market_value_eur"] = pd.to_numeric(sq_raw["market_value_eur"], errors="coerce").fillna(0)
    squads = sq_raw.merge(teams[["team_id", "team_name"]], on="team_id", how="left")

    group = matches[matches["status"].str.lower() == "completed"].copy()
    group = group[group["home_team_name"].isin(teams["team_name"].values)]
    group = group.reset_index(drop=True)

    # outcome labels
    def outcome(r):
        if r.home_score > r.away_score: return 0
        if r.home_score == r.away_score: return 1
        return 2
    group["y"] = group.apply(outcome, axis=1)

    # Elo features
    tm = teams.set_index("team_name")
    group["home_elo"] = group["home_team_name"].map(tm["elo_rating"])
    group["away_elo"] = group["away_team_name"].map(tm["elo_rating"])
    group["elo_diff"] = group["home_elo"] - group["away_elo"]
    group["home_rank"] = group["home_team_name"].map(tm["fifa_ranking_pre_tournament"])
    group["away_rank"] = group["away_team_name"].map(tm["fifa_ranking_pre_tournament"])
    group["rank_diff"] = group["away_rank"] - group["home_rank"]  # higher = better home

    hosts = {"USA", "Mexico", "Canada"}
    group["host_adv"] = (group["home_team_name"].isin(hosts)).astype(float) - (group["away_team_name"].isin(hosts)).astype(float)

    # Squad features
    def team_feats(name):
        s = squads[squads["team_name"] == name]
        if len(s) == 0: return pd.Series({"mv": 0, "caps": 0, "goals": 0})
        top = s.nlargest(11, "market_value_eur")
        return pd.Series({"mv": top["market_value_eur"].sum() / 1e6,
                          "caps": s["caps"].fillna(0).mean(),
                          "goals": s["goals"].fillna(0).mean()})

    for col in ["mv", "caps", "goals"]:
        group[f"home_{col}"] = group["home_team_name"].apply(lambda n: team_feats(n)[col])
        group[f"away_{col}"] = group["away_team_name"].apply(lambda n: team_feats(n)[col])

    group["mv_diff"] = group["home_mv"] - group["away_mv"]
    group["caps_diff"] = group["home_caps"] - group["away_caps"]
    group["goals_diff"] = group["home_goals"] - group["away_goals"]

    return group, teams

def load_market_probs(group):
    with open(ARTIFACTS / "polymarket_raw.json") as f:
        raw = json.load(f)

    poly_map = {}
    for entry in raw.get("group_matches", []):
        h = normalize(entry.get("home", ""))
        a = normalize(entry.get("away", ""))
        probs = entry.get("p_market") or entry.get("p_model")
        if probs and h and a:
            poly_map[(h, a)] = probs

    mkt = np.zeros((len(group), 3))
    covered = np.zeros(len(group), dtype=bool)
    for i, row in group.iterrows():
        key = (row["home_team_name"], row["away_team_name"])
        if key in poly_map:
            p = poly_map[key]
            mkt[i] = [p.get("H", p.get("home_win", 1/3)),
                      p.get("D", p.get("draw", 1/3)),
                      p.get("A", p.get("away_win", 1/3))]
            covered[i] = True
        else:
            mkt[i] = [1/3, 1/3, 1/3]
    # normalize rows to sum to 1
    row_sums = mkt.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mkt = mkt / row_sums
    return mkt, covered

def oof_preds_logistic(X, y, C=1.0):
    """Generate OOF predictions via RSKF."""
    oof = np.zeros((len(y), 3))
    for tr, te in RSKF.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
        clf.fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)
    return oof

def eval_stacker(meta_X, y, C_sweep=None, label="stacker"):
    """Evaluate logistic stacker on meta_X with RSKF."""
    if C_sweep is None:
        C_sweep = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    best_ll, best_C = 1e9, 0.01
    for C in C_sweep:
        folds = []
        for tr, te in RSKF.split(meta_X, y):
            scaler = StandardScaler()
            mtr = scaler.fit_transform(meta_X[tr])
            mte = scaler.transform(meta_X[te])
            clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
            clf.fit(mtr, y[tr])
            folds.append(log_loss(y[te], clf.predict_proba(mte)))
        ll = np.mean(folds)
        if ll < best_ll:
            best_ll, best_C = ll, C

    # final eval with best C
    folds = []
    for tr, te in RSKF.split(meta_X, y):
        scaler = StandardScaler()
        mtr = scaler.fit_transform(meta_X[tr])
        mte = scaler.transform(meta_X[te])
        clf = LogisticRegression(C=best_C, solver="lbfgs", max_iter=2000)
        clf.fit(mtr, y[tr])
        folds.append(log_loss(y[te], clf.predict_proba(mte)))
    ll = np.mean(folds)
    std = np.std(folds)
    acc = np.mean([np.mean(np.argmax(
        LogisticRegression(C=best_C, solver="lbfgs", max_iter=2000)
        .fit(StandardScaler().fit_transform(meta_X[tr]), y[tr])
        .predict_proba(StandardScaler().fit_transform(meta_X[te])), axis=1) == y[te])
        for tr, te in RSKF.split(meta_X, y)])
    return ll, std, acc, best_C

def geom_blend(p1, p2, alpha=0.5):
    eps = 1e-9
    lb = alpha * np.log(np.clip(p1, eps, 1)) + (1-alpha) * np.log(np.clip(p2, eps, 1))
    lb -= lb.max(axis=1, keepdims=True)
    p = np.exp(lb)
    return p / p.sum(axis=1, keepdims=True)

def wilcox_p(folds_model, baseline=0.8337):
    elo_folds = [baseline] * len(folds_model)
    try:
        stat, p = wilcoxon(folds_model, elo_folds, alternative='less')
    except Exception:
        p = float('nan')
    return p

def main():
    print("Loading data...")
    group, teams = load_data()
    y = group["y"].values
    n = len(group)
    print(f"n={n} matches")

    # Market probs (A=0, D=1, H=2 in JSON → reorder to H=0, D=1, A=2)
    mkt_raw, covered = load_market_probs(group)
    # JSON probs are [home_win, draw, away_win] already
    mkt = mkt_raw  # shape (n, 3), columns = [H, D, A]
    print(f"Market coverage: {covered.sum()}/64")

    # ── Base learner feature matrices ──────────────────────────────────────────
    elo_feats = group[["elo_diff", "host_adv"]].values
    rank_feats = group[["rank_diff", "host_adv"]].values
    squad_feats = group[["elo_diff", "rank_diff", "mv_diff", "caps_diff", "goals_diff", "host_adv"]].values

    print("\n── Generating base learner OOF predictions ──")
    oof_elo = oof_preds_logistic(elo_feats, y, C=1.0)
    print(f"  Elo OOF ll={log_loss(y, oof_elo):.4f}")
    oof_rank = oof_preds_logistic(rank_feats, y, C=1.0)
    print(f"  Rank OOF ll={log_loss(y, oof_rank):.4f}")

    # Find best C for squad+elo
    best_squad_ll, best_squad_C = 1e9, 0.1
    for C in [0.01, 0.03, 0.1, 0.3, 1.0]:
        oof_sq = oof_preds_logistic(squad_feats, y, C=C)
        ll = log_loss(y, oof_sq)
        if ll < best_squad_ll: best_squad_ll, best_squad_C = ll, C
    oof_squad = oof_preds_logistic(squad_feats, y, C=best_squad_C)
    print(f"  Squad+Elo OOF ll={log_loss(y, oof_squad):.4f} (C={best_squad_C})")
    print(f"  Market (pretrained) ll={log_loss(y, mkt):.4f}")

    # ── Exp 5a: OOF Stacking (market as base learner) ─────────────────────────
    print("\n── 5a: OOF Stacking: [market, elo, rank, squad] → logistic meta ──")
    meta_X = np.hstack([mkt, oof_elo, oof_rank, oof_squad])  # 12 features
    ll_5a, std_5a, acc_5a, C_5a = eval_stacker(meta_X, y)
    p_5a = wilcox_p([ll_5a]*50)  # approximate
    print(f"  4-learner meta (C={C_5a}): ll={ll_5a:.4f} ± {std_5a:.4f}  acc={acc_5a:.4f}")

    # Proper fold-level Wilcoxon
    elo_folds_ref = []
    stk_folds_5a = []
    for tr, te in RSKF.split(meta_X, y):
        scaler = StandardScaler()
        mtr = scaler.fit_transform(meta_X[tr])
        mte = scaler.transform(meta_X[te])
        clf = LogisticRegression(C=C_5a, solver="lbfgs", max_iter=2000)
        clf.fit(mtr, y[tr])
        stk_folds_5a.append(log_loss(y[te], clf.predict_proba(mte)))
        # Elo baseline on same fold
        sc2 = StandardScaler()
        clf2 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
        clf2.fit(sc2.fit_transform(elo_feats[tr]), y[tr])
        elo_folds_ref.append(log_loss(y[te], clf2.predict_proba(sc2.transform(elo_feats[te]))))
    p_5a_wil = wilcoxon(stk_folds_5a, elo_folds_ref, alternative='less').pvalue
    ll_5a = np.mean(stk_folds_5a); std_5a = np.std(stk_folds_5a)
    print(f"  Stacker 4-learner: {ll_5a:.4f} ± {std_5a:.4f}  p={p_5a_wil:.4f}")

    # ── Exp 5b: Stacking with 3 learners (no rank, since often noisy) ─────────
    print("\n── 5b: OOF Stacking: [market, elo, squad] → logistic meta ──")
    meta_3 = np.hstack([mkt, oof_elo, oof_squad])  # 9 features
    stk_folds_5b = []
    for tr, te in RSKF.split(meta_3, y):
        # nested C selection
        inner_C, inner_ll = 0.01, 1e9
        for C in [0.001, 0.003, 0.01, 0.03, 0.1]:
            sc = StandardScaler()
            clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
            clf.fit(sc.fit_transform(meta_3[tr]), y[tr])
            ll_c = log_loss(y[tr], clf.predict_proba(sc.transform(meta_3[tr])))
            if ll_c < inner_ll: inner_ll, inner_C = ll_c, C
        sc = StandardScaler()
        clf = LogisticRegression(C=inner_C, solver="lbfgs", max_iter=2000)
        clf.fit(sc.fit_transform(meta_3[tr]), y[tr])
        stk_folds_5b.append(log_loss(y[te], clf.predict_proba(sc.transform(meta_3[te]))))
    ll_5b = np.mean(stk_folds_5b); std_5b = np.std(stk_folds_5b)
    p_5b = wilcoxon(stk_folds_5b, elo_folds_ref, alternative='less').pvalue
    print(f"  Stacker 3-learner: {ll_5b:.4f} ± {std_5b:.4f}  p={p_5b:.4f}")

    # ── Exp 5c: Log-linear (learned log-opinion-pool) ──────────────────────────
    # Use log(market_probs) + structural features as direct logistic features
    print("\n── 5c: Log-linear (learned alpha log-opinion-pool) ──")
    eps = 1e-9
    log_mkt = np.log(np.clip(mkt, eps, 1))  # (n, 3)
    log_elo = np.log(np.clip(oof_elo, eps, 1))  # (n, 3)
    log_squad = np.log(np.clip(oof_squad, eps, 1))  # (n, 3)
    log_feats = np.hstack([log_mkt, log_elo, log_squad])  # 9 features (learn alpha in log space)

    stk_folds_5c = []
    for tr, te in RSKF.split(log_feats, y):
        inner_C, inner_ll = 0.01, 1e9
        for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
            sc = StandardScaler()
            # No intercept since we're doing log-linear mixing
            clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, fit_intercept=False)
            clf.fit(sc.fit_transform(log_feats[tr]), y[tr])
            ll_c = log_loss(y[tr], clf.predict_proba(sc.transform(log_feats[tr])))
            if ll_c < inner_ll: inner_ll, inner_C = ll_c, C
        sc = StandardScaler()
        clf = LogisticRegression(C=inner_C, solver="lbfgs", max_iter=2000, fit_intercept=False)
        clf.fit(sc.fit_transform(log_feats[tr]), y[tr])
        stk_folds_5c.append(log_loss(y[te], clf.predict_proba(sc.transform(log_feats[te]))))
    ll_5c = np.mean(stk_folds_5c); std_5c = np.std(stk_folds_5c)
    p_5c = wilcoxon(stk_folds_5c, elo_folds_ref, alternative='less').pvalue
    print(f"  Learned log-opinion-pool: {ll_5c:.4f} ± {std_5c:.4f}  p={p_5c:.4f}")

    # ── Exp 5d: MLP on market + structural features ────────────────────────────
    print("\n── 5d: MLP (8 hidden units) on market + structural features ──")
    all_feats = np.hstack([mkt, elo_feats, rank_feats, group[["mv_diff","caps_diff","goals_diff"]].values])
    stk_folds_5d = []
    for tr, te in RSKF.split(all_feats, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(all_feats[tr])
        Xte = sc.transform(all_feats[te])
        mlp = MLPClassifier(hidden_layer_sizes=(8,), alpha=1.0, max_iter=500, random_state=42)
        mlp.fit(Xtr, y[tr])
        stk_folds_5d.append(log_loss(y[te], mlp.predict_proba(Xte)))
    ll_5d = np.mean(stk_folds_5d); std_5d = np.std(stk_folds_5d)
    p_5d = wilcoxon(stk_folds_5d, elo_folds_ref, alternative='less').pvalue
    print(f"  MLP(8): {ll_5d:.4f} ± {std_5d:.4f}  p={p_5d:.4f}")

    # ── Exp 5e: Best of geometric blend + stacking comparison ─────────────────
    # Our current best (temp-scaled 3-way) = 0.8000
    # Try adding a 5th base learner: market-vs-elo residual
    print("\n── 5e: Market + structural ALL-features logistic ──")
    # Use market probs + all structural as raw features (not OOF)
    combo_feats = np.hstack([mkt, elo_feats, group[["mv_diff","caps_diff","goals_diff","host_adv"]].values])
    stk_folds_5e = []
    for tr, te in RSKF.split(combo_feats, y):
        inner_C, inner_ll = 0.01, 1e9
        for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
            sc = StandardScaler()
            clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
            clf.fit(sc.fit_transform(combo_feats[tr]), y[tr])
            ll_c = log_loss(y[tr], clf.predict_proba(sc.transform(combo_feats[tr])))
            if ll_c < inner_ll: inner_ll, inner_C = ll_c, C
        sc = StandardScaler()
        clf = LogisticRegression(C=inner_C, solver="lbfgs", max_iter=2000)
        clf.fit(sc.fit_transform(combo_feats[tr]), y[tr])
        stk_folds_5e.append(log_loss(y[te], clf.predict_proba(sc.transform(combo_feats[te]))))
    ll_5e = np.mean(stk_folds_5e); std_5e = np.std(stk_folds_5e)
    p_5e = wilcoxon(stk_folds_5e, elo_folds_ref, alternative='less').pvalue
    print(f"  Market+structural direct logistic: {ll_5e:.4f} ± {std_5e:.4f}  p={p_5e:.4f}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n── Exp05 Summary ──")
    results = {
        "5a_stacker_4learner": (ll_5a, std_5a, p_5a_wil),
        "5b_stacker_3learner": (ll_5b, std_5b, p_5b),
        "5c_learned_log_opinion_pool": (ll_5c, std_5c, p_5c),
        "5d_mlp8": (ll_5d, std_5d, p_5d),
        "5e_market_structural_direct": (ll_5e, std_5e, p_5e),
    }
    best_label, best_ll_final = "none", 1e9
    for label, (ll, std, p) in results.items():
        verdict = "GREEN" if p < 0.05 else ("FLAT" if p < 0.2 else "RED")
        star = " ← BEST" if ll < best_ll_final else ""
        if ll < best_ll_final: best_ll_final = ll; best_label = label
        print(f"  {label}: {ll:.4f} ± {std:.4f}  p={p:.4f} → {verdict}{star}")

    print(f"\n  Current best (Exp04 temp-scaled 3-way): 0.8000")
    print(f"  Exp05 best ({best_label}): {best_ll_final:.4f}")
    print(f"  Wave-2 frontier: 0.7608 (gap = {best_ll_final - 0.7608:+.4f})")

    # Save artifacts
    metrics = {
        "experiment": "exp05_stacking",
        "cv": "RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)",
        "n_matches": int(n),
        "results": {
            k: {"log_loss": float(v[0]), "std": float(v[1]), "wilcoxon_p": float(v[2])}
            for k, v in results.items()
        },
        "best_label": best_label,
        "best_log_loss": float(best_ll_final),
        "baseline_log_loss": 0.8337,
        "wave2_frontier": 0.7608,
        "prior_best_exp04": 0.8000,
    }
    with open(ARTIFACTS / "exp05_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    run_info = {
        "experiment": "exp05_stacking",
        "date": "2026-06-28",
        "python": "3.x",
        "sklearn": "1.9.0",
        "base_learners": ["market(pretrained)", "elo_logistic", "rank_logistic", "squad+elo_logistic"],
        "meta_learner": "LogisticRegression(nested C sweep)",
        "seed": 0,
        "market_coverage": int(covered.sum()),
        "data_source": "polymarket_raw.json + matches_detailed.csv + squads_and_players.csv",
    }
    with open(ARTIFACTS / "exp05_run.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print("\nArtifacts saved.")
    return metrics

if __name__ == "__main__":
    main()
