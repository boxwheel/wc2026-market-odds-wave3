# WC 2026 Market-Odds Wave-3 Study

Wave-3 Flywheel campaign cluster: **Full Bookmaker Market-Odds Coverage** for FIFA WC 2026 group-stage match prediction.

## Key Finding
Polymarket pre-match de-vigged 1X2 odds cover 56/64 group matches. On those 56, market probabilities achieve log-loss 0.8104 vs 0.8337 Elo baseline. Full 64-match coverage with p_model imputation yields 0.8500. Market + Elo blend best at 0.8237 (FLAT vs baseline).

## Eval Protocol
Repeated stratified 5-fold × 10-repeat CV, seed=0. 64 completed group-stage matches.

## Baselines
- Elo-logistic: 0.8337 ± 0.134
- Wave-2 ensemble frontier: 0.7608

## Structure
- `code/` — experiment scripts
- `artifacts/` — metrics JSON, OOF probabilities, run configs
