# Implementation Status

## Purpose

This file summarizes the implementation work completed for the IHDP-based DML overlap project.

## What Has Been Decided

The project uses IHDP as the empirical base for a semi-synthetic simulation study. The recommended design is:

- keep the empirical IHDP covariate distribution (985 rows, 28 covariates)
- generate synthetic treatment assignments using a structural propensity model
- generate synthetic outcomes using Hill (2011) response surfaces
- hold sample size fixed at 985
- vary overlap strength and model complexity across design settings
- run 200 Monte Carlo repetitions per setting
- use two nuisance learners: Lasso and XGBoost

## Data Pipeline

### Source data

- `hill_data/ihdp_sim.csv`: compact simulation-oriented IHDP data (985 × 29)

### Cleaning

- `src/clean_ihdp.py` reads `hill_data/ihdp_sim.csv` and writes `processed/ihdp_sim_processed.csv`

### Synthetic data generation

`src/generate_ihdp_synthetic.py` supports two propensity models:

**Structural model** (`--propensity-model structural`, primary):

Inspired by Hill (2011). Treatment depends on a race × birthweight interaction:

- Non-white mothers: `logit(p) = baseline + strength * (0.5 - 1.5 * zscore(bw))`
- White mothers: `logit(p) = baseline - strength * 1.0`

At strength=0, treatment is random (p = 0.38 for all). As strength increases:
- Non-white low-birthweight infants become nearly certain to be treated
- White mothers become nearly certain to be controls
- The interaction between race and birthweight is the key challenge for propensity estimation

**Logistic model** (`--propensity-model logistic`, legacy):

Linear combination of 4 clinical covariates (bw, nnhealth, momage, b.marr) through a logistic function scaled by `overlap_strength`. Creates diffuse overlap violations. Propensity clipped at [0.001, 0.999].

**Outcome surfaces** (both models):

- Surface A (linear): constant ATE = 4, beta from {0,1,2,3,4}
- Surface B (nonlinear): exponential Y(0), linear Y(1), heterogeneous effects, ATE calibrated to 4 via omega
- Both use 26-column design matrix (intercept + 25 covariates, ethnicity excluded)

## DML Estimation

Implemented in `src/dml_simulation.py`.

- `doubleml` (v0.10.1) PLR with 5-fold cross-fitting
- Lasso (LassoCV) and XGBoost nuisance learners
- XGBoost hyperparameters: n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8
- Conditions on all 28 covariates
- Returns: point estimate, SE, 95% CI, coverage indicator, and R_D
- R_D = Var(D_tilde) / Var(D) computed from cross-fitted propensity residuals

## Monte Carlo Runner

Implemented in `src/monte_carlo.py`. Supports `--propensity-model` flag.

## Completed Monte Carlo Runs

### v1: Logistic propensity, clip [0.02, 0.98], overlap 0–5

3,200 estimations (4 overlap × 2 surfaces × 2 learners × 200 reps).

Coverage floor: ~73% (XGBoost/linear/overlap=3). DML degrades gracefully — SE inflation partially compensates for bias.

Archived: `output/archive/mc_results_v1_clip02_overlap0-5.csv`

### v2: Logistic propensity, clip [0.001, 0.999], overlap 0–10

4,800 estimations (6 overlap × 2 surfaces × 2 learners × 200 reps).

Coverage floor: ~65.5% (XGBoost/linear/overlap=5). Relaxed clipping and higher overlap levels produced more dramatic failures but coverage was still non-monotonic — SE inflation partially recovers coverage at extreme overlap.

Archived: `output/archive/mc_results_v2_clip001_overlap0-10.csv`

### v3: Structural propensity, overlap 0–5 (PRIMARY)

5,600 estimations (7 overlap × 2 surfaces × 2 learners × 200 reps).

**Linear surface:**

| strength | learner | bias | RMSE | coverage | R_D |
|----------|---------|------|------|----------|-----|
| 0.0 | Lasso | +0.00 | 0.15 | 94.0% | 1.00 |
| 0.0 | XGBoost | -0.03 | 0.10 | 93.5% | 1.04 |
| 0.5 | Lasso | -0.11 | 0.23 | 86.0% | 0.94 |
| 0.5 | XGBoost | -0.03 | 0.10 | 96.5% | 0.93 |
| 1.0 | Lasso | -0.23 | 0.38 | 59.5% | 0.84 |
| 1.0 | XGBoost | -0.03 | 0.12 | 97.0% | 0.72 |
| 1.5 | Lasso | -0.34 | 0.52 | 48.5% | 0.77 |
| 1.5 | XGBoost | -0.03 | 0.13 | 97.0% | 0.55 |
| 2.0 | Lasso | -0.43 | 0.64 | 39.0% | 0.73 |
| 2.0 | XGBoost | -0.03 | 0.16 | 94.5% | 0.42 |
| 3.0 | Lasso | -0.54 | 0.77 | 35.0% | 0.68 |
| 3.0 | XGBoost | -0.06 | 0.20 | 94.5% | 0.27 |
| 5.0 | Lasso | -0.64 | 0.88 | 27.0% | 0.66 |
| 5.0 | XGBoost | -0.11 | 0.29 | 94.0% | 0.15 |

**Nonlinear surface:**

| strength | learner | bias | RMSE | coverage | R_D |
|----------|---------|------|------|----------|-----|
| 0.0 | Lasso | -0.01 | 0.17 | 98.5% | 1.00 |
| 0.0 | XGBoost | -0.04 | 0.17 | 96.5% | 1.04 |
| 0.5 | Lasso | +0.02 | 0.37 | 94.0% | 0.94 |
| 0.5 | XGBoost | -0.03 | 0.25 | 97.0% | 0.93 |
| 1.0 | Lasso | -0.07 | 0.57 | 75.0% | 0.84 |
| 1.0 | XGBoost | -0.12 | 0.46 | 91.0% | 0.72 |
| 1.5 | Lasso | -0.15 | 0.86 | 56.0% | 0.77 |
| 1.5 | XGBoost | -0.18 | 0.71 | 78.5% | 0.55 |
| 2.0 | Lasso | -0.21 | 1.13 | 46.0% | 0.73 |
| 2.0 | XGBoost | -0.27 | 0.90 | 74.0% | 0.42 |
| 3.0 | Lasso | -0.29 | 1.37 | 41.0% | 0.68 |
| 3.0 | XGBoost | -0.25 | 1.03 | 72.5% | 0.27 |
| 5.0 | Lasso | -0.38 | 1.59 | 33.5% | 0.66 |
| 5.0 | XGBoost | -0.31 | 1.29 | 73.5% | 0.15 |

### Key findings

1. **Lasso coverage collapses** from 94% to 27% (linear) and 98.5% to 33.5% (nonlinear)
2. **XGBoost on linear surface is robust** — 94% coverage even at strength=5
3. **XGBoost on nonlinear surface degrades moderately** — 73.5% at strength=5
4. **R_D diverges by learner**: Lasso R_D = 0.66, XGBoost R_D = 0.15. However, most of this gap (+0.47 of +0.51) is from Lasso using linear regression on a binary outcome — a functional form error, not structural misspecification. See propensity isolation experiment below.
5. **Lasso bias is systematically negative** — underestimates treatment effect by up to 0.64

### v6: Propensity isolation experiment (COMPLETED)

Holds outcome model constant (LassoCV) and varies only the propensity model:
- Lasso+LassoCV (regression): R_D = 0.66, coverage = 28%
- Lasso+LogisticRegression: R_D = 0.19, coverage = 49%
- Structural misspecification alone (with correct functional form) produces only +0.04 inflation

### v4: Highdim propensity (COMPLETED)

Propensity model: 6 continuous covariates (bw, b.head, preterm, birth.o, nnhealth, momage) with main effects + 3 pairwise interactions (bw×nnhealth, preterm×momage, head×birth.o). Standardized score through logistic function.

5,600 estimations (7 overlap × 2 surfaces × 2 learners × 200 reps).

Results archived: `output/mc_results_v4_highdim.csv`

### v5: Threshold propensity (COMPLETED)

Propensity model: Step function in birthweight (low bw → high treatment, high bw → low, middle → neutral), modulated by marriage status, with smooth nnhealth component. Standardized score through logistic function.

5,600 estimations (7 overlap × 2 surfaces × 2 learners × 200 reps).

Results archived: `output/mc_results_v5_threshold.csv`

## R_D Threshold Analysis (SUPERSEDED)

Earlier analysis identified R_D < 0.70 (from a flexible learner) as a candidate threshold for flagging unreliable DML inference. However, this approach has a fundamental limitation: **R_D is blind to the outcome side**. The same R_D value produces 94% coverage (linear surface) and 73% coverage (nonlinear surface). A single R_D threshold cannot work because coverage depends on both propensity overlap and outcome model quality.

This motivated the joint (R_D, R_Y) diagnostic — see below.

## Outcome-Side Diagnostic Attempt (COMPLETED — NEGATIVE RESULT)

### Motivation

At similar R_D values (~0.15–0.19), coverage ranges from 24% to 94% depending on the outcome model. R_D correctly reports overlap quality but cannot predict whether DML will actually succeed, because that also depends on outcome model quality.

### R_Y = cross-fitted outcome R² (FAILED)

Definition: R_Y = 1 − SS_res(Y, ĝ(X)) / SS_tot(Y), using out-of-fold predictions from doubleml.

**v7 MC results (8,400 estimations, structural DGP, completed):**

AUC for predicting coverage (nonlinear surface):
- R_D alone: 0.649
- R_Y alone: 0.522 (essentially random)
- R_D + R_Y: 0.645 (slightly worse than R_D alone)

**Why R_Y fails:** In PLR, ĝ(X) estimates E[Y|X] = g₀(X) + τ·m₀(X). As overlap strengthens, m₀(X) gets more variable, so E[Y|X] has more variance, so R_Y mechanically rises. XGBoost R_Y goes from 0.36 (overlap=0) to 0.72 (overlap=5) — the diagnostic moves in the wrong direction.

### Within-group residual variance (ALSO FAILED)

Attempted fix: compute Var(Y - ĝ(X) | D=d) within each treatment group, pooled. This removes between-group treatment signal.

Result: separates learners (XGBoost ~6-8, Lasso ~12-17) but adds no predictive power beyond R_D (AUC goes from 0.744 to 0.745). The metric captures "which learner you used" but doesn't vary within a learner across overlap levels.

### Why outcome-side diagnostics are hard in PLR

The outcome model estimates E[Y|X] = g₀(X) + τ·m₀(X), conflating outcome quality with treatment predictability. Any metric based on Y − ĝ(X) inherits this contamination. Conditioning on D doesn't fully fix it because ĝ(X) still targets E[Y|X], not E[Y|X,D].

This is a structural limitation of PLR, not a failure of any particular metric.

## Current Findings (PLR)

1. R_D works well as a within-learner diagnostic — coverage tracks R_D cleanly for a given learner
2. R_D is robust to structural propensity misspecification under correct functional form (+0.055 inflation over oracle R_D* = 0.139)
3. Outcome model quality is the dominant source of coverage variation, and R_D is blind to it
4. No simple outcome-side diagnostic works in PLR due to treatment channel contamination
5. The inflation identity holds exactly: R_D = R_D* + Var(δ)/Var(D)

## IRM Extension (COMPLETED)

### Motivation

PLR estimates E[Y|X] = g₀(X) + τ·m₀(X), conflating outcome quality with treatment predictability. IRM (`DoubleMLIRM`) estimates E[Y|X, D=0] and E[Y|X, D=1] separately, removing the treatment channel. We tested whether arm-specific R² (R_Y0, R_Y1) could serve as an uncontaminated outcome-side diagnostic.

### Implementation

- Added `estimate_ate_irm()` to `src/dml_simulation.py` using `DoubleMLIRM`
- Added `run_monte_carlo_irm()` and `--framework irm` CLI flag to `src/monte_carlo.py`
- IRM requires classifier propensity models — compatible learners: `lasso_logistic`, `xgboost`, `ridge`, `rf`
- Computes R_Y0 (cross-fitted R² among controls), R_Y1 (among treated), R_D (same as PLR)

### MC Runs

**v8:** 3 overlap strengths (0, 2, 5) × 2 surfaces × 2 learners (xgboost, lasso_logistic) × 100 reps = 1,200 estimations. Structural DGP. Results: `output/mc_results_v8_irm_structural.csv` (superseded by v8c)

**v8b:** overlap=5 × 2 surfaces × 2 learners (ridge, rf) × 100 reps = 400 estimations. Structural DGP. Results: `output/mc_results_v8b_irm_rf_ridge.csv` (superseded by v8c)

**v8c (balanced, primary):** 3 overlap strengths (0, 2, 5) × 2 surfaces × 4 learners × 100 reps = 2,400 estimations. Structural DGP. Results: `output/mc_results_v8c_irm_balanced.csv`. All 4 learners appear at every overlap × surface cell, enabling honest cross-learner AUC comparisons.

### Results (balanced v8c)

**R_Y0 contamination is gone:** XGBoost R_Y0 stays flat across overlap (0.844 → 0.832 → 0.832), unlike PLR's R_Y which nearly doubles (0.36 → 0.72).

**R_Y0 measures outcome quality accurately:** separates learners (XGBoost 0.832 vs Lasso+Logistic 0.675 on linear at overlap=5) and surfaces (linear 0.832 vs nonlinear 0.770 for XGBoost).

**R_D alone becomes a sharper diagnostic under IRM:** AUC = 0.804 (linear) / 0.734 (nonlinear), vs PLR's 0.566 / 0.649. IRM's doubly-robust score is more sensitive to propensity quality, so R_D carries more information about coverage.

**R_Y0 adds modest predictive value:** AUC for R_D + R_Y0 = 0.826 (linear) / 0.767 (nonlinear), a 2–3 point improvement over R_D alone. PLR's R_Y added nothing (0.564 / 0.645 vs 0.566 / 0.649).

**R_Y0 does not rank learners by coverage:** Ridge has the best R_Y0 (0.920) but only 62% coverage; XGBoost has lower R_Y0 (0.832) but 68%; RF has decent R_Y0 (0.724) but 30%. The ranking mismatch is the averaging barrier.

### Local R_Y0 experiment (v9/v10, COMPLETED)

**Motivation:** test the averaging hypothesis by computing R_Y0 restricted to controls in thin-overlap regions.

**Implementation:** Added to `estimate_ate_irm()`: IPW-weighted R² (R_Y0_weighted), local R² at multiple thresholds (m̂ > 0.5, 0.7, 0.8), top-k R² (30 most extreme controls), local MSE, global-local gap, and IF-weighted MSE. Added `--start-seed` CLI flag to `monte_carlo.py` for extending runs with non-overlapping seeds.

**MC runs:**
- v9: 2,400 estimations (100 reps, seeds 1-100) with weighted/local R_Y0 at threshold 0.7. Results: `output/mc_results_v9_irm_weighted_merged.csv`
- v10: 2,400 estimations (100 reps, seeds 1-100) with full variant sweep (thresholds 0.5/0.7/0.8, top-k, MSE, gap, IF-weighted). Results: `output/mc_results_v10_merged.csv`

**Key result:** R_D + R_Y0_local (m̂ > 0.5) achieves AUC 0.842 (linear) / 0.794 (nonlinear), vs R_D + R_Y0 (global) at 0.821 / 0.761. The +2–3 point improvement is concentrated in cross-learner comparisons (e.g., Ridge vs RF: 0.786 → 0.828). Threshold 0.5 is the sweet spot — stricter thresholds produce too few units for stable R². Top-k (30 most extreme controls) also works.

**Averaging hypothesis confirmed:** global R_Y0 hides local fit failures. Local R_Y0 captures coverage-relevant information the global metric misses. The gap between global and local R_Y0 is itself a diagnostic signal.

### Current Findings (Combined)

1. **Three-part structural finding:**
   - PLR's outcome target E[Y|X] = g₀(X) + τ·m₀(X) is contaminated — R_Y moves in the wrong direction with overlap. Fixable by switching to IRM.
   - IRM removes the contamination, and this upgrades R_D from poor cross-learner predictor (AUC ~0.6) to good (AUC ~0.8), plus R_Y0 adds modest further value.
   - Global outcome metrics miss local failures in thin-overlap regions. Partially fixable by restricting to "unnatural" units in both arms simultaneously — controls with m̂ > 0.5 and treated with m̂ < 0.5. Local R_Y adds +2–3 AUC points and improves cross-learner discrimination.
2. **Diagnostic staircase:** PLR R_D+R_Y AUC ≈ 0.60 → IRM R_D+R_Y0 AUC ≈ 0.79 → IRM R_D+R_Y0_local AUC ≈ 0.82. Each step improvement is statistically significant (bootstrap 95% CIs exclude zero, p < 0.01, 2,000 replicates).
3. **Include both arms:** compute R_Y0_local (controls with m̂ > 0.5) and R_Y1_local (treated with m̂ < 0.5) and include both alongside R_D. In our IHDP structural DGP, R_Y0_local carries the signal (control arm is larger and more heterogeneous) while R_Y1 has negligible cross-learner variance (0.067 to 0.115) and is naturally downweighted. In a DGP where the treated arm dominates, the roles would reverse. Including both is cheap, avoids a decision point, and ensures the diagnostic works regardless of which arm is informative.
4. **Practical recommendation:** use IRM over PLR when diagnostic observability matters. Compute the diagnostic triple: R_D + R_Y0_local (controls with m̂ > 0.5) + R_Y1_local (treated with m̂ < 0.5). Optionally compare global vs local R² for each arm — a large gap signals the outcome model struggles where it matters most.
5. **Paper framing:** the contribution is the diagnostic-observability perspective — choice of DML framework has underappreciated consequences for what practitioners can see about their estimates.

## Validation

Pre-MC validation (`src/validate_before_mc.py`) confirmed:
- ATE calibration: both surfaces produce exactly ATE = 4.0
- Propensity distributions behave as expected at each overlap level
- Seed independence: different seeds produce different datasets, same seed reproduces exactly
- Baseline coverage: all cells at overlap=0 achieve 93–99% coverage

## Analysis Outputs

**IMPORTANT:** The existing figures in `output/figures/` were generated from v3 results (Lasso vs XGBoost only, no R_Y). They predate the propensity isolation experiment. They show the Lasso R_D=0.66 inflation without the functional form confound caveat. **Do not use these figures for the current narrative.**

`src/analyze_results.py` generates (HISTORICAL, from v3):

Figures (in `output/figures/`):
- `coverage_vs_overlap.png`, `bias_vs_overlap.png`, `rmse_vs_overlap.png`
- `r_d_vs_overlap.png`, `r_d_vs_coverage.png`, `r_d_vs_bias.png`
- `se_vs_overlap.png`, `reversal_plot_*.png`, `decomposition_verification.png`

`src/analyze_joint_diagnostic.py` generates (from v7, documents R_Y failure):

Figures (in `output/figures/joint/`):
- `joint_2d_scatter.png`, `joint_coverage_heatmap.png`
- `joint_ry_marginal_value.png`, `joint_predictive_comparison.png`
- `joint_decision_boundary.png`

Tables:
- `output/tables/joint/joint_summary.csv` (v7 full summary with R_D, R_Y, coverage)
- `output/tables/joint/predictive_comparison.csv` (AUC comparison showing R_Y failure)

## Environment

- Python 3.14, venv at `.venv/`
- Key packages: numpy, pandas, scikit-learn, xgboost, doubleml, matplotlib, seaborn
- Dependencies listed in `requirements.txt`
