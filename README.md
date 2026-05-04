# Double Trouble: Diagnostic Observability in Double Machine Learning Under Weak Overlap

## Project Overview

This project studies the finite-sample reliability of Double Machine Learning (DML) for causal inference when the overlap assumption is weak.

The empirical foundation is the IHDP data bundled in this repository. We use IHDP covariates to build semi-synthetic experiments that deliberately weaken overlap while preserving realistic dependence across covariates.

The main objective is to evaluate whether an observable diagnostic can warn researchers when DML inference is becoming unstable because treatment assignment is too predictable from observed covariates.

The project combines:

- semi-synthetic simulation grounded in IHDP covariates
- Monte Carlo evaluation of DML estimators
- overlap diagnostics tied to finite-sample inferential failure

## Key Findings

### R_D alone is necessary but not sufficient

R_D = Var(D̃)/Var(D) correctly measures propensity-side overlap quality, but it tells you nothing about the outcome model. At similar R_D values (~0.15–0.19), coverage ranges from 24% to 94% depending on the outcome model and surface complexity. R_D is blind to this — we call this **outcome-side blindness**.

### R_D works well within a learner

Despite outcome-side blindness, R_D reliably tracks coverage *within* a given learner configuration. For lasso_logistic on the nonlinear surface, coverage drops from 94.5% (R_D=1.03) to 51.5% (R_D=0.20) in a clean monotonic relationship. The problem is only cross-learner.

### Cross-fitted outcome R² (R_Y) fails as an outcome diagnostic

We attempted to pair R_D with R_Y (cross-fitted outcome R²) as an outcome-side companion. **R_Y adds no predictive value** — AUC for predicting coverage is 0.649 with R_D alone and 0.645 with R_D + R_Y. R_Y is contaminated by treatment predictability: as overlap weakens, E[Y|X] = g₀(X) + τ·m₀(X) has more variance because m₀(X) is more variable, so R_Y mechanically rises even as coverage falls. This is a structural limitation of PLR, not fixable by simple normalization. Within-group residual variance (conditioning on D) also fails for related reasons.

### Prior findings on R_D inflation

The inflation identity R_D = R_D* + Var(δ)/Var(D) shows misspecification always inflates R_D upward (anti-conservative direction). However, with correct functional form for binary treatment, the practical magnitude is small (+0.055 relative to the true oracle R_D* = 0.139, computed directly from the known m₀(X) over 200 MC replications). Most of the dramatic inflation initially observed (+0.519) was from using linear regression on a binary outcome — a basic functional form error, not structural misspecification. XGBoost's R_D (0.150) carries only +0.010 inflation over oracle.

### Robustness

- 25,000+ estimations across 4 DGPs, 3 sample sizes (500, 985, 2000), 5 learner configurations
- Inflation does not shrink with sample size (population-level property)
- Finding confirmed with both XGBoost and Random Forest as flexible baselines
- Propensity isolation experiment (Lasso+Logistic vs Lasso+Lasso) cleanly separates functional form from structural misspecification

## Key Research Questions

How does weak overlap affect DML performance in finite samples?

We evaluate performance using:

- bias
- RMSE
- confidence interval coverage

Can an observable diagnostic predict when DML inference becomes unreliable?

Our proposed diagnostic is the residualized treatment variance ratio:

`R_D = Var(D - m_hat(X)) / Var(D)`

where `m_hat(X)` is the estimated propensity score.

Interpretation:

- large `R_D` means substantial unexplained treatment variation remains
- small `R_D` means treatment is highly predictable from covariates, indicating weak overlap
- **caveat**: R_D is only as trustworthy as the propensity model that produced it

## Causal Framework

We work in the potential outcomes framework:

- `Y_i(1)` is the outcome under treatment
- `Y_i(0)` is the outcome under control
- `Y_i = D_i Y_i(1) + (1 - D_i) Y_i(0)` is the observed outcome

The target estimand is:

`tau = E[Y(1) - Y(0)]`

Identification assumptions:

1. Ignorability: `(Y(0), Y(1)) ⟂ D | X`
2. SUTVA
3. Overlap: `0 < P(D = 1 | X) < 1`

Weak overlap arises when propensity scores are close to 0 or 1 for many units.

## Method: Double Machine Learning (PLR)

We estimate treatment effects using the partially linear regression DML setup:

`Y = tau D + g(X) + epsilon`

Workflow:

1. Estimate nuisance functions with machine learning:
   `g(X) = E[Y | X]`
   `m(X) = E[D | X]`
2. Residualize:
   `Y_tilde = Y - g_hat(X)`
   `D_tilde = D - m_hat(X)`
3. Regress `Y_tilde` on `D_tilde`

Cross-fitting (5 folds) is used to reduce overfitting bias from the nuisance stages.

## Why IHDP

The repository includes IHDP example and simulation data in `hill_data/`. IHDP is a natural fit because:

- the data are already local and documented
- Hill (2011) provides well-studied response surfaces for semi-synthetic outcomes
- Hill's original work used structural subgroup removal to create overlap violations, which directly inspires our design

## Simulation Design

We use the observed IHDP covariate matrix (985 rows, 28 covariates) as the base distribution.

### Treatment assignment

Four propensity models are available (selectable via `--propensity-model`), each creating different types of overlap violations:

**Structural model** (primary, inspired by Hill 2011):

Treatment depends on an interaction between race and birthweight:

- Non-white mothers: propensity is a steep function of birthweight (sicker infants → more likely treated)
- White mothers: propensity is low and flat

This creates an interpretable structural overlap violation. The interaction between race and birthweight is the key challenge — Lasso cannot capture it, XGBoost can.

**Logistic model** (legacy):

Treatment depends on a linear combination of 4 clinical covariates (birthweight, neonatal health, mother's age, marital status) passed through a logistic function. Creates diffuse, gradual overlap violations. Both learners can capture this.

**Highdim model** (robustness):

Treatment depends on all 6 continuous covariates plus 3 pairwise interactions (bw×nnhealth, preterm×momage, head×birth.o). Main effects are linear (Lasso-friendly), interactions are multiplicative (require flexibility). Tests threshold robustness with higher-dimensional confounding.

**Threshold model** (robustness):

Treatment depends on step functions in birthweight (low → high treatment, high → low), modulated by marriage status, with a smooth neonatal health component. Creates discontinuous propensity — like a policy eligibility cutoff. Tests threshold robustness with a different nonlinearity flavor.

All models: `overlap_strength` controls severity. At 0, treatment is random (p = 0.38 for all). At higher values, treatment becomes increasingly predictable.

Canonical overlap grid: `overlap_strength in {0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0}`

### Outcome model

Potential outcomes follow the response surfaces from Hill (2011):

**Surface A** (linear, constant treatment effect):

- `Y(0) ~ N(X @ beta, 1)`
- `Y(1) ~ N(X @ beta + 4, 1)`
- beta drawn from `{0, 1, 2, 3, 4}` with probabilities `{0.5, 0.2, 0.15, 0.1, 0.05}`
- Re-drawn each Monte Carlo iteration
- ATE = 4 by construction

**Surface B** (nonlinear control, linear treated, heterogeneous treatment effect):

- `Y(0) ~ N(exp((X + 0.5) @ beta), 1)`
- `Y(1) ~ N((X + 0.5) @ beta - omega, 1)`
- beta drawn from `{0, 0.1, 0.2, 0.3, 0.4}` with probabilities `{0.6, 0.1, 0.1, 0.1, 0.1}`
- Re-drawn each Monte Carlo iteration
- omega calibrated so ATE = 4 each run

Both surfaces use 25 covariates (ethnicity excluded) from the IHDP sim schema, standardizing the 6 continuous covariates and leaving the 19 binary covariates as-is. An intercept column is prepended, giving a 26-column design matrix.

## DML Learners

Five learner configurations, testing different propensity and outcome model combinations:

- **Lasso**: LassoCV for both outcome and propensity (regression on binary treatment — functional form mismatch for propensity)
- **Lasso+Logistic**: LassoCV outcome + LogisticRegression propensity (correct functional form, still linear — isolates structural misspecification)
- **Ridge**: RidgeCV outcome + LogisticRegression propensity
- **XGBoost**: XGBRegressor outcome + XGBClassifier propensity. Conservative hyperparameters (n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8). These are fixed, not tuned — results depend on XGBoost being flexible enough, which is not guaranteed in all settings.
- **Random Forest**: RandomForestRegressor/Classifier (n_estimators=200, max_depth=5). Alternative flexible learner to confirm findings aren't XGBoost-specific.

## Monte Carlo Design

Primary grid: 7 overlap strengths × 2 surfaces × 2 learners × 200 reps = 5,600 estimations per DGP. Additional robustness runs with extra learners, sample sizes, and the propensity isolation experiment bring the total to 25,000+ estimations.

For each setting:

1. construct treatment from IHDP covariates using the structural propensity model
2. generate potential outcomes using Hill (2011) response surfaces
3. form observed outcomes
4. estimate treatment effects with DML (PLR, 5-fold cross-fitting)
5. compute R_D and R_Y from the cross-fitted predictions
6. record point estimate, SE, CI, coverage, R_D, R_Y, and outcome RMSE

## Results Summary

### Structural propensity model (primary results)

**Linear surface:**

| strength | learner | bias | coverage | R_D |
|----------|---------|------|----------|-----|
| 0.0 | Lasso | +0.00 | 94.0% | 1.00 |
| 0.0 | XGBoost | -0.03 | 93.5% | 1.04 |
| 2.0 | Lasso | -0.43 | 39.0% | 0.73 |
| 2.0 | XGBoost | -0.03 | 94.5% | 0.42 |
| 5.0 | Lasso | -0.64 | 27.0% | 0.66 |
| 5.0 | XGBoost | -0.11 | 94.0% | 0.15 |

**Nonlinear surface:**

| strength | learner | bias | coverage | R_D |
|----------|---------|------|----------|-----|
| 0.0 | Lasso | -0.01 | 98.5% | 1.00 |
| 0.0 | XGBoost | -0.04 | 96.5% | 1.04 |
| 2.0 | Lasso | -0.21 | 46.0% | 0.73 |
| 2.0 | XGBoost | -0.27 | 74.0% | 0.42 |
| 5.0 | Lasso | -0.38 | 33.5% | 0.66 |
| 5.0 | XGBoost | -0.31 | 73.5% | 0.15 |

### Key takeaways

1. **Lasso coverage collapses** from 94% to 27% (linear) and 98.5% to 33.5% (nonlinear) as overlap weakens
2. **XGBoost on linear surface barely degrades** — 94% coverage even at strength=5
3. **Lasso R_D appears misleading** (0.66 with 27% coverage), but this is primarily driven by using linear regression on a binary outcome — a functional form error. With logistic regression propensity (same linear features), R_D drops to 0.19, close to XGBoost's 0.15.
4. **Learner choice matters enormously** under structural overlap violations — but the dominant factor is outcome model quality, not propensity diagnostics. The full propensity isolation experiment is in `src/robustness_analysis.py`.

## Repository Structure

- `README.md`: project framing and results
- `hill_data/`: bundled IHDP source data and metadata
- `processed/`: cleaned data — IHDP baseline CSV and LaLonde covariates
- `src/clean_ihdp.py`: IHDP data cleaning
- `src/generate_ihdp_synthetic.py`: semi-synthetic data generator (4 propensity models: structural, logistic, highdim, threshold)
- `src/dml_simulation.py`: DML estimation — PLR (`estimate_ate`) and IRM (`estimate_ate_irm`) with R_D, R_Y, R_Y0, R_Y1 diagnostics
- `src/monte_carlo.py`: Monte Carlo grid runner (supports `--framework plr|irm`)
- `src/analyze_results.py`: result aggregation and plotting
- `src/validate_before_mc.py`: pre-MC validation checks
- `src/find_threshold.py`: initial R_D threshold analysis
- `src/threshold_deep_dive.py`: detailed threshold analysis by surface and learner
- `src/robustness_analysis.py`: pooled threshold robustness test across all DGPs
- `src/robustness_sample_size_and_rf.py`: sample size and alternative learner robustness checks
- `src/plot_reversal.py`: reversal plots (bias and R_D vs overlap strength)
- `src/decomposition_analysis.py`: empirical verification of the inflation identity
- `src/theorem_anticonservative.py`: formal statement of the inflation identity
- `src/analyze_joint_diagnostic.py`: joint (R_D, R_Y) diagnostic analysis with 2D heatmaps and AUC comparison
- `src/validate_external_dataset.py`: external validation of IRM local-R² diagnostics on LaLonde covariates (semi-synthetic outcomes, known ATE)
- `output/figures/`: generated plots (including `output/figures/paper/` for final paper figures)
- `output/tables/`: summary tables
- `output/archive/`: archived MC results from all runs
- `hill_code/`: imported R simulation and example scripts
- `requirements.txt`: Python dependencies

## Goal

Develop a practical, computable diagnostic for DML reliability under weak overlap.

**Main findings:**

*Propensity-side (R_D):*
1. R_D = Var(D̃)/Var(D) is a reliable within-learner diagnostic for overlap quality, robust to structural propensity misspecification under correct functional form.
2. R_D is blind to outcome model quality, which is the dominant source of coverage variation in our simulations.

*Outcome-side diagnostics — PLR:*
3. Cross-fitted outcome R² (R_Y) fails in PLR — it is contaminated by the treatment channel (E[Y|X] = g₀(X) + τ·m₀(X)) and moves in the wrong direction as overlap weakens.

*Outcome-side diagnostics — IRM:*
4. IRM (Interactive Regression Model) fixes the contamination by estimating E[Y|X,D=0] and E[Y|X,D=1] separately. Arm-specific R_Y0 is stable across overlap levels, confirming the treatment channel contamination is gone.
5. Under IRM, R_D alone becomes a much sharper diagnostic (AUC 0.80 linear / 0.73 nonlinear, vs PLR's 0.57 / 0.65). Adding R_Y0 yields a further modest improvement (AUC 0.83 / 0.77).
6. R_Y0 does not rank learners by coverage. Ridge achieves the best R_Y0 (0.920) but only 62% coverage; XGBoost has lower R_Y0 (0.832) but 68%. Global outcome fit hides local failures in thin-overlap covariate regions.

*Local outcome diagnostics:*
7. The general diagnostic is **local arm-specific R² in the extrapolation region**, computed for both arms: R_Y0_local (controls with m̂ > 0.5) and R_Y1_local (treated with m̂ < 0.5). Include both alongside R_D — the practitioner does not need to choose which arm to inspect. The uninformative arm's local R² will have low variance and be naturally downweighted. In our IHDP DGP, R_Y0_local carries the signal (control arm is larger); in a DGP where the treated arm dominates, R_Y1_local would be informative instead. R_D + R_Y0_local achieves AUC 0.842 (linear) / 0.794 (nonlinear), a +2–3 point improvement over global R_Y0.
8. The gap between global and local R² is a useful qualitative check (RF drops from 0.671 → 0.528, the worst-coverage learner) but does not add predictive power as a formal predictor beyond R_Y0_local itself (gap alone AUC: 0.561 / 0.647; R_D + gap: 0.763 / 0.740). Its value is as a human-readable interpretation aid, not a model feature.

*The diagnostic staircase:*
```
PLR:   R_D + R_Y        → AUC ≈ 0.60  (contaminated)
IRM:   R_D + R_Y0       → AUC ≈ 0.79  (clean but global)
IRM:   R_D + R_Y0_local → AUC ≈ 0.82  (clean and local)
```

Each step fixes one structural barrier to diagnostic observability. All improvements are statistically significant (bootstrap p < 0.01).

*Practical recommendation:* Use IRM. Compute the diagnostic triple: R_D + R_Y0_local (controls with m̂ > 0.5) + R_Y1_local (treated with m̂ < 0.5). No need to choose which arm — include both. Optionally compare global vs local R² for each arm as a qualitative check. Empirical confirmation on a reversed-propensity DGP (where R_Y1_local is informative) is noted as future work.

See `src/` for the complete analysis pipeline, including the inflation identity (`theorem_anticonservative.py`, `decomposition_analysis.py`), propensity isolation experiment (`robustness_analysis.py`), PLR robustness checks (`robustness_sample_size_and_rf.py`), and IRM with local R_Y0 diagnostics (`analyze_joint_diagnostic.py`).
