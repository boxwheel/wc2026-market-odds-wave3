"""
Exp07: Precision Calibration — exact alpha optimization + per-class scaling
===========================================================================
With n=64, complex models overfit (Exps 05-06). This experiment squeezes the
remaining gain from the geometric blend via:
1. Continuous α optimization via scipy.minimize (vs grid search in Exp04)
2. Per-class temperature scaling (3 Ts instead of 1)
3. Market-Elo disagreement as a correction signal
4. Separate alpha for Polymarket-covered (56) vs imputed (8) matches
5. Platt scaling on OOF probabilities (alternative calibration)

Target: beat 0.8000 (Exp04)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
from scipy.optimize import minimize, minimize_scalar
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV

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
        return pd.Series({"mv": top["market_value_eur"].sum() / 1e6,
                          "caps": s["caps"].fillna(0).mean(),
                          "goals": s["goals"].fillna(0).mean()})

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
    covered_pairs = set()
    for entry in raw.get("group_matches", []):
        h = normalize(entry.get("home", ""))
        a = normalize(entry.get("away", ""))
        pm = entry.get("p_market")
        p = pm if pm else entry.get("p_model")
        if p and h and a:
            poly_map[(h, a)] = p
            if pm:
                covered_pairs.add((h, a))

    mkt = np.zeros((len(group), 3))
    covered = np.zeros(len(group), dtype=bool)
    for i, row in group.iterrows():
        key = (row["home_team_name"], row["away_team_name"])
        if key in poly_map:
            p = poly_map[key]
            mkt[i] = [p.get("H", 1/3), p.get("D", 1/3), p.get("A", 1/3)]
            covered[i] = key in covered_pairs
        else:
            mkt[i] = [1/3, 1/3, 1/3]
    row_sums = mkt.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return mkt / row_sums, covered


def geom_blend(p1, p2, alpha=0.5):
    lb = alpha * np.log(np.clip(p1, eps, 1)) + (1-alpha) * np.log(np.clip(p2, eps, 1))
    lb -= lb.max(axis=1, keepdims=True)
    p = np.exp(lb)
    return p / p.sum(axis=1, keepdims=True)


def geom_blend3(p1, p2, p3, a1=0.45, a2=0.4):
    b12 = geom_blend(p1, p2, alpha=a1)
    return geom_blend(b12, p3, alpha=(1-a2))


def oof_logistic(X, y, C=1.0):
    oof = np.zeros((len(y), 3))
    for tr, te in RSKF.split(X, y):
        sc = StandardScaler()
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))
    return oof


def temp_scale_per_class(p, T):
    """Per-class temperature scaling. T is length-3 vector."""
    log_p = np.log(np.clip(p, eps, 1))
    scaled = log_p / np.array(T).reshape(1, 3)
    scaled -= scaled.max(axis=1, keepdims=True)
    q = np.exp(scaled)
    return q / q.sum(axis=1, keepdims=True)


def wilcox_p(fold_losses, elo_folds_ref):
    try:
        return wilcoxon(fold_losses, elo_folds_ref, alternative='less').pvalue
    except Exception:
        return float('nan')


def main():
    print("Loading data...")
    group = load_data()
    y = group["y"].values
    n = len(group)

    mkt, covered = load_market_probs(group)
    print(f"n={n}, Market coverage: {covered.sum()}/64 (proper), total probs for 64: yes")

    elo_feats = group[["elo_diff", "host_adv"]].values
    squad_feats = group[["elo_diff", "rank_diff", "mv_diff", "caps_diff", "goals_diff", "host_adv"]].values

    # Elo OOF reference for Wilcoxon
    elo_folds_ref = []
    for tr, te in RSKF.split(elo_feats, y):
        sc = StandardScaler()
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
        clf.fit(sc.fit_transform(elo_feats[tr]), y[tr])
        elo_folds_ref.append(log_loss(y[te], clf.predict_proba(sc.transform(elo_feats[te]))))
    print(f"Elo OOF ll={np.mean(elo_folds_ref):.4f}")

    oof_elo = oof_logistic(elo_feats, y, C=1.0)
    oof_squad = oof_logistic(squad_feats, y, C=0.1)

    # ── 7a: Exact α optimization with scipy.minimize (per fold) ──────────────
    print("\n── 7a: Exact α optimization (scipy.minimize, nested) ──")
    folds_7a = []
    best_a1_vals, best_a2_vals = [], []
    for tr, te in RSKF.split(mkt, y):
        # Optimize a1, a2 on training fold
        def neg_ll(params):
            a1, a2 = params
            if a1 <= 0 or a1 >= 1 or a2 <= 0 or a2 >= 1:
                return 1e9
            b = geom_blend3(mkt[tr], oof_elo[tr], oof_squad[tr], a1=a1, a2=a2)
            return log_loss(y[tr], b)

        best_val, best_params = 1e9, (0.45, 0.4)
        for a1_init in [0.3, 0.45, 0.6]:
            for a2_init in [0.3, 0.45, 0.6]:
                res = minimize(neg_ll, [a1_init, a2_init],
                               method='Nelder-Mead',
                               options={'xatol': 0.001, 'fatol': 0.001, 'maxiter': 200})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x

        a1_opt, a2_opt = np.clip(best_params[0], 0.1, 0.9), np.clip(best_params[1], 0.1, 0.9)
        best_a1_vals.append(a1_opt); best_a2_vals.append(a2_opt)
        b_te = geom_blend3(mkt[te], oof_elo[te], oof_squad[te], a1=a1_opt, a2=a2_opt)
        folds_7a.append(log_loss(y[te], b_te))

    ll_7a = np.mean(folds_7a); std_7a = np.std(folds_7a)
    p_7a = wilcox_p(folds_7a, elo_folds_ref)
    v7a = "GREEN" if p_7a < 0.05 else ("FLAT" if p_7a < 0.2 else "RED")
    print(f"  Exact α opt: {ll_7a:.4f} ± {std_7a:.4f}  p={p_7a:.4f} → {v7a}")
    print(f"  Mean α1={np.mean(best_a1_vals):.3f} ± {np.std(best_a1_vals):.3f}")
    print(f"  Mean α2={np.mean(best_a2_vals):.3f} ± {np.std(best_a2_vals):.3f}")

    # ── 7b: Per-class temperature scaling on 3-way blend ─────────────────────
    print("\n── 7b: Per-class temperature scaling (3 Ts) ──")
    blend3 = geom_blend3(mkt, oof_elo, oof_squad, a1=0.45, a2=0.4)
    folds_7b = []
    for tr, te in RSKF.split(blend3, y):
        # Optimize T_H, T_D, T_A on training fold
        def neg_ll_perclass(T):
            T = np.clip(T, 0.2, 5.0)
            q = temp_scale_per_class(blend3[tr], T)
            return log_loss(y[tr], q)

        best_val, best_T = 1e9, [1.0, 1.0, 1.0]
        for T_D_init in [0.7, 1.0, 1.3, 1.5]:  # draws are often miscalibrated
            res = minimize(neg_ll_perclass, [1.0, T_D_init, 1.0],
                           method='Nelder-Mead',
                           options={'xatol': 0.01, 'fatol': 0.001, 'maxiter': 300})
            if res.fun < best_val:
                best_val = res.fun
                best_T = res.x

        T_opt = np.clip(best_T, 0.2, 5.0)
        q_te = temp_scale_per_class(blend3[te], T_opt)
        folds_7b.append(log_loss(y[te], q_te))

    ll_7b = np.mean(folds_7b); std_7b = np.std(folds_7b)
    p_7b = wilcox_p(folds_7b, elo_folds_ref)
    v7b = "GREEN" if p_7b < 0.05 else ("FLAT" if p_7b < 0.2 else "RED")
    print(f"  Per-class T scaling: {ll_7b:.4f} ± {std_7b:.4f}  p={p_7b:.4f} → {v7b}")

    # ── 7c: Separate alpha for covered vs imputed matches ─────────────────────
    print("\n── 7c: Separate α for covered (56) vs imputed (8) matches ──")
    # For imputed matches, market probs are Dixon-Coles model, not actual market
    # We can weight them differently
    folds_7c = []
    for tr, te in RSKF.split(mkt, y):
        def neg_ll_2alpha(params):
            a1, a2, a1_imp, a2_imp = params
            if not (0 < a1 < 1 and 0 < a2 < 1 and 0 < a1_imp < 1 and 0 < a2_imp < 1):
                return 1e9
            b = np.zeros((len(tr), 3))
            for j, idx in enumerate(tr):
                if covered[idx]:
                    b[j] = geom_blend3(mkt[[idx]], oof_elo[[idx]], oof_squad[[idx]], a1=a1, a2=a2)[0]
                else:
                    b[j] = geom_blend3(mkt[[idx]], oof_elo[[idx]], oof_squad[[idx]], a1=a1_imp, a2=a2_imp)[0]
            return log_loss(y[tr], b)

        best_val = 1e9; best_p = (0.45, 0.4, 0.1, 0.5)
        for a1_imp_init in [0.1, 0.2, 0.3]:
            res = minimize(neg_ll_2alpha, [0.45, 0.4, a1_imp_init, 0.5],
                           method='Nelder-Mead',
                           options={'xatol': 0.01, 'fatol': 0.001, 'maxiter': 400})
            if res.fun < best_val:
                best_val = res.fun; best_p = res.x

        a1, a2, a1_imp, a2_imp = best_p
        a1, a2 = np.clip(a1, 0.1, 0.9), np.clip(a2, 0.1, 0.9)
        a1_imp, a2_imp = np.clip(a1_imp, 0.0, 0.9), np.clip(a2_imp, 0.1, 0.9)

        b_te = np.zeros((len(te), 3))
        for j, idx in enumerate(te):
            a1_use, a2_use = (a1, a2) if covered[idx] else (a1_imp, a2_imp)
            b_te[j] = geom_blend3(mkt[[idx]], oof_elo[[idx]], oof_squad[[idx]], a1=a1_use, a2=a2_use)[0]
        folds_7c.append(log_loss(y[te], b_te))

    ll_7c = np.mean(folds_7c); std_7c = np.std(folds_7c)
    p_7c = wilcox_p(folds_7c, elo_folds_ref)
    v7c = "GREEN" if p_7c < 0.05 else ("FLAT" if p_7c < 0.2 else "RED")
    print(f"  Dual-alpha: {ll_7c:.4f} ± {std_7c:.4f}  p={p_7c:.4f} → {v7c}")

    # ── 7d: Combined: exact α + per-class T + temperature scaling ─────────────
    print("\n── 7d: Combined exact α + single-T scaling ──")
    folds_7d = []
    for tr, te in RSKF.split(mkt, y):
        # Step 1: optimize α
        def neg_ll_a(params):
            a1, a2 = params
            if not (0 < a1 < 1 and 0 < a2 < 1): return 1e9
            b = geom_blend3(mkt[tr], oof_elo[tr], oof_squad[tr], a1=a1, a2=a2)
            return log_loss(y[tr], b)

        best_val, best_a = 1e9, [0.45, 0.4]
        for a1i, a2i in [(0.35, 0.35), (0.45, 0.4), (0.5, 0.45)]:
            res = minimize(neg_ll_a, [a1i, a2i], method='Nelder-Mead',
                           options={'xatol': 0.002, 'fatol': 0.001, 'maxiter': 200})
            if res.fun < best_val: best_val = res.fun; best_a = res.x

        a1_opt, a2_opt = np.clip(best_a[0], 0.1, 0.9), np.clip(best_a[1], 0.1, 0.9)
        blend_tr = geom_blend3(mkt[tr], oof_elo[tr], oof_squad[tr], a1=a1_opt, a2=a2_opt)
        blend_te = geom_blend3(mkt[te], oof_elo[te], oof_squad[te], a1=a1_opt, a2=a2_opt)

        # Step 2: optimize single T
        def neg_ll_T(T):
            T = float(np.clip(T, 0.3, 5.0))
            lb = np.log(np.clip(blend_tr, eps, 1)) / T
            lb -= lb.max(axis=1, keepdims=True)
            q = np.exp(lb); q /= q.sum(axis=1, keepdims=True)
            return log_loss(y[tr], q)

        res_T = minimize_scalar(neg_ll_T, bounds=(0.3, 3.0), method='bounded')
        T_opt = float(np.clip(res_T.x, 0.3, 3.0))
        lb = np.log(np.clip(blend_te, eps, 1)) / T_opt
        lb -= lb.max(axis=1, keepdims=True)
        q_te = np.exp(lb); q_te /= q_te.sum(axis=1, keepdims=True)
        folds_7d.append(log_loss(y[te], q_te))

    ll_7d = np.mean(folds_7d); std_7d = np.std(folds_7d)
    p_7d = wilcox_p(folds_7d, elo_folds_ref)
    v7d = "GREEN" if p_7d < 0.05 else ("FLAT" if p_7d < 0.2 else "RED")
    print(f"  Exact α + T scaling: {ll_7d:.4f} ± {std_7d:.4f}  p={p_7d:.4f} → {v7d}")

    # ── 7e: Market-Elo disagreement correction ─────────────────────────────────
    print("\n── 7e: Market-Elo disagreement as logistic correction ──")
    # disagreement features
    disagree = np.log(np.clip(mkt, eps, 1)) - np.log(np.clip(oof_elo, eps, 1))  # (n, 3)
    blend3_fixed = geom_blend3(mkt, oof_elo, oof_squad, a1=0.45, a2=0.4)

    folds_7e = []
    for tr, te in RSKF.split(blend3_fixed, y):
        # use disagreement to correct blend probs
        eps2 = 1e-9
        log_b = np.log(np.clip(blend3_fixed[tr], eps2, 1))  # (n_tr, 3)
        X_corr = np.hstack([log_b, disagree[tr]])  # 6 features
        X_te_corr = np.hstack([np.log(np.clip(blend3_fixed[te], eps2, 1)), disagree[te]])

        inner_C, inner_ll = 0.01, 1e9
        for C in [0.003, 0.01, 0.03, 0.1, 0.3]:
            clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, fit_intercept=True)
            clf.fit(X_corr, y[tr])
            ll_c = log_loss(y[tr], clf.predict_proba(X_corr))
            if ll_c < inner_ll: inner_ll, inner_C = ll_c, C
        clf = LogisticRegression(C=inner_C, solver="lbfgs", max_iter=2000, fit_intercept=True)
        clf.fit(X_corr, y[tr])
        folds_7e.append(log_loss(y[te], clf.predict_proba(X_te_corr)))

    ll_7e = np.mean(folds_7e); std_7e = np.std(folds_7e)
    p_7e = wilcox_p(folds_7e, elo_folds_ref)
    v7e = "GREEN" if p_7e < 0.05 else ("FLAT" if p_7e < 0.2 else "RED")
    print(f"  Market-Elo disagree correction: {ll_7e:.4f} ± {std_7e:.4f}  p={p_7e:.4f} → {v7e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Exp07 Summary ──")
    results = {
        "7a_exact_alpha": (ll_7a, std_7a, p_7a),
        "7b_perclass_T": (ll_7b, std_7b, p_7b),
        "7c_dual_alpha": (ll_7c, std_7c, p_7c),
        "7d_exact_alpha_T": (ll_7d, std_7d, p_7d),
        "7e_disagree_corr": (ll_7e, std_7e, p_7e),
    }
    best_label, best_ll_final = "none", 1e9
    for label, (ll, std, p) in results.items():
        verdict = "GREEN" if p < 0.05 else ("FLAT" if p < 0.2 else "RED")
        star = " ← BEST" if ll < best_ll_final else ""
        if ll < best_ll_final: best_ll_final = ll; best_label = label
        print(f"  {label}: {ll:.4f} ± {std:.4f}  p={p:.4f} → {verdict}{star}")

    print(f"\n  Prior best (Exp04 temp-scaled): 0.8000")
    print(f"  Exp07 best ({best_label}): {best_ll_final:.4f}")
    delta = best_ll_final - 0.8000
    print(f"  Delta vs Exp04: {delta:+.4f}")
    print(f"  Wave-2 frontier: 0.7608 (gap = {best_ll_final - 0.7608:+.4f})")

    metrics = {
        "experiment": "exp07_precision_calib",
        "cv": "RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)",
        "n_matches": int(n),
        "results": {k: {"log_loss": float(v[0]), "std": float(v[1]), "wilcoxon_p": float(v[2])}
                    for k, v in results.items()},
        "best_label": best_label,
        "best_log_loss": float(best_ll_final),
        "baseline_log_loss": 0.8337,
        "wave2_frontier": 0.7608,
        "prior_best_exp04": 0.8000,
    }
    with open(ARTIFACTS / "exp07_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    run_info = {
        "experiment": "exp07_precision_calib",
        "date": "2026-06-28",
        "key_approaches": ["exact_alpha_nelder_mead", "per_class_temperature", "dual_alpha_covered_vs_imputed",
                           "market_elo_disagreement_correction"],
        "seed": 0,
        "data_source": "polymarket_raw.json + matches_detailed.csv + squads_and_players.csv",
    }
    with open(ARTIFACTS / "exp07_run.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print("Artifacts saved.")
    return metrics


if __name__ == "__main__":
    main()
