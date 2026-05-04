# IHDP-Based DML Overlap Study — Implementation Plan

## Overview

The project uses IHDP covariates as the empirical base for semi-synthetic simulations, evaluating how Double Machine Learning behaves as overlap is deliberately weakened through controlled treatment assignment mechanisms.

The core estimand and diagnostic:

- DML treatment effect estimation under the PLR setup
- overlap diagnostic `R_D = Var(D_tilde) / Var(D)`

## Steps

### 1. Reorganize the project around IHDP ✓

- `hill_data/ihdp_sim.csv` is the active Python-readable source dataset
- `hill_data/ihdp_example.csv` and `hill_data/sim.data` are supporting historical references
- `hill_code/` scripts are reference implementations

### 2. Maintain a reproducible cleaning step ✓

Implemented in `src/clean_ihdp.py`.

- loads `hill_data/ihdp_sim.csv`
- keeps the original sim covariate schema
- writes `processed/ihdp_sim_processed.csv` (985 × 29)

### 3. Define the IHDP covariate set for DML ✓

Documented in `ihdp_dml_spec.md`. The active set is the original `ihdp_sim` schema: 28 covariates covering birth/neonatal measures, maternal demographics, prenatal behavior, state indicators, and family background.

### 4. Build an IHDP-based semi-synthetic generator ✓

Implemented in `src/generate_ihdp_synthetic.py`.

Four propensity models available via `--propensity-model`:

**Structural (primary):** Inspired by Hill (2011). Treatment depends on a race × birthweight interaction. Non-white mothers have propensity that varies steeply with birthweight; white mothers have low flat propensity. Creates interpretable structural overlap violations.

**Logistic (legacy):** Linear combination of 4 clinical covariates (bw, nnhealth, momage, b.marr) through a logistic function. Creates diffuse, gradual overlap violations.

**Highdim (robustness):** 6 continuous covariates + 3 pairwise interactions. Tests higher-dimensional confounding.

**Threshold (robustness):** Step function in birthweight × marriage status. Tests discontinuous confounding.

All models use Hill (2011) response surfaces A (linear, ATE=4) and B (nonlinear Y(0), ATE=4 via omega calibration) for outcomes.

### 5. Implement overlap diagnostics ✓

R_D = Var(D_tilde) / Var(D) is computed within `src/dml_simulation.py` using cross-fitted propensity residuals from the doubleml model.

R_Y = cross-fitted outcome R² is also now computed, using out-of-fold outcome predictions from doubleml. Together (R_D, R_Y) form a joint propensity-outcome diagnostic.

Both are stored alongside each estimation result.

### 6. Implement DML estimation ✓

Implemented in `src/dml_simulation.py`.

- uses `doubleml` PLR with 5-fold cross-fitting
- supports 5 learner configurations: Lasso, XGBoost, Ridge, RF, Lasso+Logistic
- conditions on all 28 covariates
- returns point estimate, SE, 95% CI, coverage indicator, R_D, R_Y, outcome_rmse, outcome_resid_std, propensity_rmse

### 7. Run Monte Carlo experiments ✓

Implemented in `src/monte_carlo.py`.

Three full runs completed:

- **v1** (logistic propensity, clip 0.02/0.98, overlap 0–5): 3,200 estimations. Coverage floor ~73%.
- **v2** (logistic propensity, clip 0.001/0.999, overlap 0–10): 4,800 estimations. Coverage floor ~65.5%.
- **v3** (structural propensity, overlap 0–5): 5,600 estimations. Lasso coverage drops to 27%. Primary results.

All archived in `output/archive/`.

### 8. Produce analysis outputs ✓

Implemented in `src/analyze_results.py`.

Generates 7 figures in `output/figures/`:
- coverage vs overlap strength
- bias vs overlap strength
- RMSE vs overlap strength
- R_D vs overlap strength
- R_D vs coverage (central plot)
- R_D vs |bias|
- SE inflation vs overlap strength

Summary table in `output/tables/mc_summary.csv`.

### 9. Pre-run validation ✓

Implemented in `src/validate_before_mc.py`. Checks ATE calibration, propensity distributions, seed independence, and baseline coverage before committing to a full MC run.

### 10. Clean up repo organization ✓

- generated simulation outputs excluded from version control
- `requirements.txt` for Python dependencies
- `.venv/` for local environment

### 11. R_D threshold analysis (SUPERSEDED)

Earlier analysis identified R_D < 0.70 as a candidate threshold, but R_D is blind to outcome model quality. The same R_D produces 94% and 73% coverage depending on the outcome surface.

### 12. Outcome-side diagnostic (COMPLETED — NEGATIVE RESULT)

Attempted to pair R_D with R_Y (cross-fitted outcome R²) for a joint diagnostic. **R_Y fails** — it adds no predictive value beyond R_D (AUC: 0.649 → 0.645 on nonlinear surface).

**Why:** R_Y is contaminated by treatment predictability. In PLR, ĝ(X) estimates E[Y|X] = g₀(X) + τ·m₀(X). As overlap strengthens, m₀(X) gets more variable, so R_Y mechanically rises even as coverage falls. Within-group residual variance also fails for related reasons.

**This is a structural limitation of PLR**, not fixable by normalization. Outcome model quality assessment in DML remains an open problem.

**v7 MC (structural DGP, 8,400 estimations):** completed. Results and figures in `output/figures/joint/` and `output/tables/joint/`.

**NOTE on existing plots:** Figures in `output/figures/` predate the propensity isolation experiment. They show Lasso R_D=0.66 inflation without the functional form confound caveat. Do not use for current narrative.

## Verification

- `clean_ihdp.py` creates a consistent analysis-ready file from `hill_data/ihdp_sim.csv` ✓
- the synthetic generator preserves IHDP covariate structure while producing tunable overlap ✓
- diagnostic scripts show R_D behavior varies by learner specification ✓
- Monte Carlo results show Lasso coverage collapsing under structural overlap violations ✓
- R_D inflation is primarily from functional form error (regression on binary), not structural misspecification ✓
- Propensity isolation experiment (Lasso+Logistic) confirms +0.04 residual structural inflation ✓
- R_Y separates outcome model quality (smoke test: 0.68 vs 0.34) ✓
- Joint (R_D, R_Y) AUC comparison: pending v7 MC completion

## Key Design Decisions

- **IHDP over LaLonde**: IHDP is already in the repo and matches the bundled R code
- **Structural propensity over logistic**: Inspired by Hill (2011). Creates interpretable overlap violations via race × birthweight interaction. Directly addresses the interpretability concern.
- **Four propensity models for robustness**: structural (primary), logistic (legacy), highdim (interaction-heavy), threshold (discontinuous). Testing diagnostic threshold across multiple DGP types.
- **Semi-synthetic rather than purely observational**: preserves known ground truth for bias and coverage evaluation
- **DML with cross-fitting**: aligned with the project's methodological goal

## Open Questions and Concerns

1. **R_D is blind to the outcome side.** The same R_D value produces 94% coverage (linear surface) and 73% coverage (nonlinear surface). A complete diagnostic may need to incorporate outcome model fit (e.g., R² of the outcome model).
2. **No sharp cliff.** Coverage degrades gradually as R_D drops — there's no single value where DML "breaks." The 0.70 threshold is the best available cut but it's a zone, not a cliff.
3. **Threshold generalizability.** If the optimal R_D cutoff moves across DGPs (v4/v5 results pending), we may need to reframe from a hard threshold to a graded reliability score.
