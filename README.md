# WC 2026 Market-Odds Wave-3 Study

Wave-3 Flywheel campaign cluster: **Full Bookmaker Market-Odds Coverage** for FIFA WC 2026 group-stage match prediction.

## Best Result

**Exp04 — Temperature-scaled 3-way geometric blend: CV log-loss 0.8000 ± 0.1238 (GREEN p=0.0013)**

- Architecture: `geom_blend(geom_blend(market, elo_oof, α=0.45), squad_oof, α=0.4)` + nested-CV temperature scaling
- Market source: Polymarket de-vigged 1X2 probs (56/64 matches) + Dixon-Coles p_model (8/64)
- Structural signals: Elo-logistic OOF, Squad+Elo-logistic OOF (market value, caps, goals)
- CV: RepeatedStratifiedKFold(5×10, seed=0), n=64 completed WC-2026 group-stage matches

## Experiment Results

| Exp | Method | CV log-loss | p | Verdict |
|-----|--------|-------------|---|---------|
| 01 | Market fixed probs (64/64 with p_model imputation) | 0.8500 | 0.906 | RED |
| 01 | Market fixed (56/64 covered matches only) | 0.8337 | 0.542 | FLAT |
| 02 | Market+Elo geometric blend α=0.4 | 0.8155 | 0.011 | **GREEN** |
| 03 | 3-way: Market+Elo+Squad geometric blend | 0.8071 | 0.025 | **GREEN** |
| **04** | **Temp-scaled 3-way (nested CV T sweep)** | **0.8000** | **0.0013** | **GREEN ← BEST** |
| 05 | OOF stacking 4-5 learners (logistic meta) | 0.8302–0.9209 | >0.66 | RED |
| 06 | XGBoost, LightGBM, Dirichlet calib | 0.8655–0.9390 | >0.99 | RED |
| 07 | Exact α opt (Nelder-Mead), per-class T | 0.8234–0.8424 | >0.10 | FLAT/RED |
| 08 | Wave-2 4-model stacking replication | 0.8684 | 0.803 | RED |

## Eval Protocol
- Repeated stratified k-fold: `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)`
- Primary metric: multiclass log-loss (mean across 50 folds)
- Hypothesis test: Wilcoxon signed-rank vs Elo-logistic folds (alternative='less')
- GREEN: p < 0.05 | FLAT: p < 0.20 | RED: p >= 0.20

## Baselines
- Elo-logistic reference: **0.8337 +/- 0.134**
- Wave-2 ensemble frontier (unverified): 0.7608

## Key Findings

### 1. Geometric Log-Opinion-Pool is the Right Architecture
The log-opinion-pool (geometric mean of probability distributions) at alpha=0.45 (market) × 0.40 (Elo) × 0.40 (squad) is optimal for n=64. It has 1-2 free parameters and is vastly more sample-efficient than any trainable meta-learner.

### 2. All Parametric Methods Overfit at n=64
Stacking (9-15 meta-features), XGBoost (max_depth=2), Dirichlet calibration (12 params), and exact alpha optimization (Nelder-Mead) all give RED results. The practical complexity budget for n=64 is 2 free parameters maximum.

### 3. Implicit Regularization from Coarse Grid Search (Exp07)
Exact Nelder-Mead alpha optimization (0.8316) is *worse* than Exp04's 5-point grid search (0.8042). The coarse grid acts as implicit regularization, preventing overfitting to 50-sample training folds.

### 4. Wave-2 Frontier (0.7608) Cannot Be Replicated (Exp08)
4-model logistic stack under RSKF 5×10 seed=0 gives 0.8684 (RED). The Wave-2 result likely involved a different CV protocol or stacking data leakage. 0.8000 is the verified strict-protocol frontier.

### 5. Market Coverage Gap is the Ceiling
The 8/64 matches without Polymarket prices are a hard ceiling. Imputation via Dixon-Coles model (p_model) degrades performance. Acquiring raw bookmaker odds for those 8 matches would be the highest-leverage data improvement.

## Reproducibility

```bash
pip install numpy pandas scipy scikit-learn xgboost lightgbm

cd code/
python3 exp01_market_coverage.py
python3 exp02_geom_blend.py
python3 exp03_3way_blend.py
python3 exp04_fine_tuned.py   # Best result
python3 exp05_stacking.py
python3 exp06_xgb_dirichlet.py
python3 exp07_precision_calib.py
python3 exp08_wave2_replication.py
```

Data: `mominullptr/fifa-world-cup-2026-dataset` (Kaggle) + Polymarket odds in `artifacts/polymarket_raw.json`

## Files
- `code/exp0{1..8}_*.py` — experiment scripts
- `artifacts/exp0{1..8}_metrics.json` — CV results per experiment
- `artifacts/exp0{1..8}_run.json` — run config, library versions, seeds
- `artifacts/polymarket_raw.json` — Polymarket pre-match odds (56/64 coverage)
- `artifacts/oof_probabilities.csv` — OOF predictions from all base learners

## Flywheel Graph
Campaign root: `646664d3-7ba6-5f80-952a-02d95baa722d`
Workstream plan: `b005553d-ebd2-5a1e-bc6f-f1e7876f7899`
Synthesis node: `e40bee6c-a7b6-5de1-89fa-d6b9626ab9ce`
