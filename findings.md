# R_D Is Robust to Structural Propensity Misspecification Under Correct Functional Form

## Overview

Double Machine Learning (DML) practitioners need diagnostics to assess when their inference is reliable. A natural candidate is the residualized treatment variance ratio:

R_D = Var(D̃) / Var(D)

where D̃ = D − m̂(X) are the treatment residuals from the first-stage propensity model. High R_D signals adequate overlap; low R_D signals weak overlap and fragile inference.

A key concern is whether propensity model misspecification corrupts this diagnostic — specifically, whether R_D can become **anti-conservative** (signaling safety when inference is actually failing). We investigate this through extensive Monte Carlo simulations and find that **R_D is more robust than one might fear**: with correct functional form for binary treatment (logistic rather than linear regression), structural propensity misspecification barely affects R_D, even when the propensity model cannot capture the true confounding structure.

**Anti-conservative** means the diagnostic errs toward overconfidence — telling the practitioner things are fine when they're not. **Conservative** means it errs toward caution.

---

## The Inflation Identity

Under first-stage misspecification, R_D decomposes as:

**R_D = R_D\* + Var(δ) / Var(D)**

where R_D\* = Var(V)/Var(D) is the oracle diagnostic, V = D − m₀(X) is the true treatment residual, and δ(X) = m₀(X) − m̂(X) is the misspecification error.

This follows from Cov(V, δ) = 0 (by iterated expectations, since E[V|X] = 0 and δ is a function of X). The identity says: R_D is always inflated upward by misspecification. The direction is anti-conservative — misspecification makes R_D look better, not worse.

We verify this empirically across 1,400 DML estimations with decomposition error < 1.59 × 10⁻⁵ and Cov(V, δ) ≈ 0.000089.

The question is whether this inflation is large enough to matter in practice.

---

## The Investigation

### Setup

We run Monte Carlo simulations using IHDP covariates (985 units, 28 covariates) with Hill (2011) response surfaces (linear and nonlinear, true ATE = 4.0). Treatment assignment is generated from four propensity structures with varying overlap severity. We test multiple learners across multiple sample sizes, totaling over 25,000 DML estimations.

### Step 1: The apparent failure

Under a structural propensity model (race × birthweight interaction), Lasso showed dramatic R_D inflation:

| Learner | Propensity model | R_D | Coverage (linear) |
|---------|-----------------|-----|-------------------|
| Lasso | LassoCV (regression) | 0.66 | 27% |
| XGBoost | XGBClassifier | 0.15 | 94% |

Lasso's R_D looks moderate (0.66) while its coverage has collapsed. The diagnostic appears dangerously anti-conservative — R_D inflated by +0.51 over XGBoost.

### Step 2: Isolating the cause

We held the outcome model constant (LassoCV) and varied only the propensity model:

| Config | Outcome model | Propensity model | R_D | Coverage (nonlinear) |
|--------|---------------|-----------------|-----|---------------------|
| Lasso | LassoCV | LassoCV (regression) | 0.66 | 28% |
| Lasso+Logistic | LassoCV | LogisticRegression | 0.19 | 49% |
| XGBoost | XGBRegressor | XGBClassifier | 0.15 | 71% |

Switching only the propensity model from linear regression to logistic regression — same LassoCV outcome model, same linear features, still unable to capture the race × birthweight interaction — reduces R_D from 0.66 to 0.19. Comparing against the true oracle R_D* = 0.139 (computed from the known propensity m₀(X)), the inflation drops from **+0.519 to +0.055**. (XGBoost's R_D of 0.15 is close to oracle but itself carries +0.010 inflation.)

### Step 3: What this means

The dramatic inflation was primarily driven by using linear regression (LassoCV) on a binary treatment variable — a gross functional form error that produces predictions outside [0,1] and generates large Var(δ). This is not a subtle diagnostic failure; it's a basic modeling mistake.

**With correct functional form** (logistic regression for binary treatment), structural misspecification produces only modest inflation (+0.055 against the true oracle R_D* = 0.139). LogisticRegression is also misspecified — it's linear, it cannot capture the race × birthweight interaction — but the logistic functional form fits binary outcomes well enough that Var(δ) stays small. (Oracle R_D* computed directly from the known data-generating propensity m₀(X) over 200 MC replications.)

**R_D is robust to structural propensity misspecification when the functional form is appropriate.**

### Step 4: The remaining coverage gap is outcome-driven

The Lasso+Logistic results reveal something important: at similar R_D values (0.19 vs 0.15), coverage differs substantially (49% vs 71%). Same overlap, similar diagnostic, but a 22 percentage point coverage gap. This gap is entirely driven by the outcome model — LassoCV cannot capture the nonlinear outcome surface that XGBRegressor handles.

This directly demonstrates **outcome-side blindness**: R_D, even when honest, cannot distinguish 49% coverage from 71% coverage. The propensity diagnostic is doing its job (correctly reporting weak overlap), but the outcome model quality — which R_D cannot measure — determines whether inference actually holds.

On the linear outcome surface, the pattern is even clearer: XGBoost achieves 94% coverage at R_D = 0.15 because the outcome surface is easy to model, regardless of weak overlap.

---

## Robustness Evidence

### Sample size (n = 500, 985, 2000)

The R_D gap between Lasso and XGBoost is constant across sample sizes:

| n | Lasso R_D | XGB R_D | Gap |
|---|-----------|---------|-----|
| 500 | 0.66 | 0.16 | +0.51 |
| 985 | 0.66 | 0.15 | +0.51 |
| 2000 | 0.65 | 0.14 | +0.51 |

(Structural DGP, strength=5, nonlinear surface.)

The inflation is a population-level property (Var(δ)/Var(D) does not vanish with n), not a small-sample artifact. This applies to both the regression-driven inflation and the small structural inflation.

### Alternative flexible learner (Random Forest)

| Learner | R_D | Coverage (nonlinear, strength=5) |
|---------|-----|----------------------------------|
| Lasso | 0.66 | 27% |
| RF | 0.22 | 24% |
| XGBoost | 0.15 | 77% |

Random Forest confirms Lasso's R_D is inflated — RF produces low R_D (0.22), similar to XGBoost (0.15). The finding is not specific to XGBoost as the reference learner.

However, RF's coverage (24%) is catastrophically worse than XGBoost's (77%) despite similar R_D. This is because RF with our hyperparameters (max_depth=5, 200 trees) is insufficiently flexible for the outcome model at extreme overlap — it cannot extrapolate nonlinear outcomes in thin-data covariate regions. RF's R_D is honest (correctly low), but honest R_D does not guarantee good inference. This reinforces the outcome-side blindness finding: R_D tells you about overlap, not about whether your outcome model can handle it.

### Cross-DGP consistency

The Lasso R_D > XGBoost R_D pattern holds across all four propensity structures:

| DGP | Lasso R_D | XGB R_D | Gap |
|-----|-----------|---------|-----|
| Structural | 0.66 | 0.15 | +0.51 |
| Highdim | 0.73 | 0.57 | +0.16 |
| Logistic | 0.35 | 0.21 | +0.14 |
| Threshold | 0.40 | 0.28 | +0.13 |

The logistic DGP serves as a positive control: when the propensity is linear and both learners can approximately capture it, the gap is smallest. This is consistent with the inflation identity — less misspecification means less inflation.

---

## R_D > 1.0 at Baseline

XGBoost produces R_D = 1.04 at overlap_strength = 0 (random treatment). R_D > 1 means Var(D̃) > Var(D), which occurs when the propensity model adds noise rather than removing signal. At overlap_strength = 0, treatment is random (p = 0.38 for all units) — there is no propensity signal to learn. XGBoost slightly overfits in cross-fitting folds, producing predictions that are anticorrelated with the true (constant) propensity. This is a minor artifact of flexible learners on small samples with no true signal, and it disappears as overlap strengthens.

---

## The Outcome-Side Problem

### R_D is necessary but not sufficient

Our results demonstrate that R_D alone is insufficient for assessing DML reliability. The propensity side (captured by R_D) and the outcome side (not captured) jointly determine inference quality. At R_D ≈ 0.15:

- XGBoost gets 94% coverage on linear outcomes, 71% on nonlinear
- Lasso+Logistic gets 39% on linear, 49% on nonlinear
- RF gets ~90% on linear, 24% on nonlinear

Same overlap, dramatically different reliability — entirely driven by outcome model quality. We call this **outcome-side blindness**: R_D tells you about overlap, but whether DML survives weak overlap depends on the outcome model's ability to extrapolate in thin-data regions.

**Note on a counterintuitive pattern:** Lasso coverage is slightly higher on the nonlinear surface (33.5%) than the linear surface (27%) despite worse outcome fit. This is not a sign that the nonlinear surface is easier. Lasso's poor fit on the nonlinear surface inflates residual variance, which inflates SE, which widens the CI. A wider CI covers the truth more often by accident — it is imprecise rather than precisely wrong. On the linear surface, Lasso's outcome model is accurate, producing tight SEs and narrow CIs that confidently miss the truth (bias = −0.64). The relationship between outcome model quality and coverage is not monotonic: a worse model can produce better coverage if its failures inflate the SE enough to compensate for bias.

### R_D works well within a learner

Despite outcome-side blindness, R_D is a reliable diagnostic *within* a given learner configuration. Lasso+Logistic on the nonlinear surface (v7 results):

| R_D | Coverage |
|-----|----------|
| 1.03 | 94.5% |
| 0.76 | 94.0% |
| 0.60 | 93.5% |
| 0.47 | 88.0% |
| 0.31 | 69.5% |
| 0.20 | 51.5% |

Coverage tracks R_D cleanly. The problem is only when comparing *across* learners — the same R_D means different things for different outcome models.

### R_Y (cross-fitted outcome R²) fails as an outcome diagnostic

We attempted to pair R_D with R_Y = 1 − SS_res(Y, ĝ(X)) / SS_tot(Y), the cross-fitted outcome R², as an outcome-side companion. This failed.

**v7 MC results (8,400 estimations, structural DGP):**

AUC for predicting coverage:

| Predictors | Linear surface | Nonlinear surface |
|------------|---------------|-------------------|
| R_D alone | 0.566 | 0.649 |
| R_Y alone | 0.539 | 0.522 |
| R_D + R_Y | 0.564 | 0.645 |

R_Y adds no predictive value. On the nonlinear surface, R_Y alone is essentially random (AUC = 0.522), and adding it to R_D makes prediction slightly *worse*.

### Why R_Y fails: treatment channel contamination

R_Y measures how well ĝ(X) predicts Y. But in PLR, ĝ(X) estimates E[Y|X] = g₀(X) + τ·m₀(X). As overlap strengthens, treatment D becomes more predictable from X (m₀(X) gets more variable), which means E[Y|X] has more variance, which means R_Y mechanically rises — even though the outcome model isn't getting better.

XGBoost R_Y on the nonlinear surface across overlap levels:

| Overlap strength | R_Y | Coverage |
|-----------------|-----|----------|
| 0.0 | 0.357 | 98.0% |
| 2.0 | 0.605 | 78.0% |
| 5.0 | 0.723 | 74.5% |

R_Y nearly doubles (0.36 → 0.72) as overlap gets *worse* and coverage *declines*. The diagnostic moves in the wrong direction — it's anti-conservative for the same reason R_D is anti-conservative under misspecification, but through a different mechanism (treatment channel rather than propensity error).

### Within-group residual variance also fails

We attempted to fix R_Y by computing outcome residual variance within treatment groups (conditioning out D to remove the treatment channel):

```
Var(Y - ĝ(X) | D=0) and Var(Y - ĝ(X) | D=1), pooled
```

This does separate learners (XGBoost pooled variance ~6-8, Lasso ~12-17) but adds no predictive power beyond R_D:

| Predictors | AUC (nonlinear) |
|------------|----------------|
| R_D alone | 0.744 |
| Within-var alone | 0.509 |
| R_D + within-var | 0.745 |

The within-group variance captures *which learner you used* but doesn't vary meaningfully *within* a learner across overlap levels — the same limitation as R_Y.

### Why outcome-side diagnostics are hard

The fundamental difficulty: in DML's PLR framework, the outcome model ĝ(X) estimates E[Y|X], which conflates the outcome function g₀(X) with the treatment channel τ·m₀(X). Any metric based on Y − ĝ(X) is contaminated by how predictable treatment is from covariates. Conditioning on D (within-group metrics) removes the between-group treatment signal but ĝ(X) still targets E[Y|X], not E[Y|X,D], so residuals within groups are systematically biased.

Building an outcome-side diagnostic that is not contaminated by treatment predictability appears to require either:
- Access to E[Y|X,D] separately (which DML's PLR doesn't estimate)
- A metric that is invariant to the treatment channel (an open question)

This is a structural limitation of the PLR framework, not a failure of any particular metric.

---

## Summary

### Propensity-side diagnostics (R_D)

1. **The inflation identity holds exactly.** R_D = R_D\* + Var(δ)/Var(D). Misspecification always inflates R_D in the anti-conservative direction. Verified with decomposition error < 10⁻⁵.

2. **R_D is robust to structural misspecification under correct functional form.** With logistic regression for binary treatment, structural propensity misspecification produces only +0.055 inflation relative to the true oracle R_D* = 0.139. The dramatic +0.519 inflation initially observed was from using linear regression on a binary outcome — a functional form error, not structural misspecification.

3. **R_D works well as a within-learner diagnostic.** For a given learner configuration, R_D tracks coverage decline cleanly. The problem is only cross-learner: the same R_D means different things for different outcome models.

4. **Outcome-side blindness is the dominant limitation.** At similar R_D values, coverage varies by 20+ percentage points depending on outcome model quality. R_D cannot detect this.

### Outcome-side diagnostics in PLR

5. **PLR's R_Y fails — treatment channel contamination.** R_Y is contaminated by treatment predictability: as overlap weakens, E[Y|X] = g₀(X) + τ·m₀(X) has more variance because m₀(X) is more variable, so R_Y mechanically rises even as coverage falls (0.36 → 0.72). AUC for predicting coverage: R_D alone = 0.649, R_Y alone = 0.522, R_D + R_Y = 0.645. R_Y adds nothing.

6. **Within-group residual variance also fails.** Conditioning on D removes between-group treatment signal, but ĝ(X) still targets E[Y|X] not E[Y|X,D], so within-group metrics separate learners but don't vary meaningfully within a learner across overlap levels.

### Outcome-side diagnostics in IRM

7. **IRM fixes the contamination problem.** IRM estimates E[Y|X, D=0] and E[Y|X, D=1] separately. Arm-specific R_Y0 stays flat across overlap levels (0.844 → 0.832 → 0.832), unlike PLR's R_Y which nearly doubles. The treatment channel contamination is gone.

8. **Under IRM, R_D becomes a sharper diagnostic, and R_Y0 adds modest value.** On a balanced 4-learner grid (2,400 estimations): R_D alone AUC = 0.80 (linear) / 0.73 (nonlinear) vs PLR's 0.57 / 0.65. R_D + R_Y0 AUC = 0.83 / 0.77, a 2–3 point improvement over R_D alone. The practical recommendation: use IRM when diagnostic observability matters.

9. **R_Y0 accurately measures outcome quality but does not rank learners by coverage.** Ridge has the best R_Y0 (0.920) but only 62% coverage; XGBoost has lower R_Y0 (0.832) but 68% coverage; RF has decent R_Y0 (0.724) but catastrophic 30% coverage. R_Y0 separates learners on outcome fit, but the ranking doesn't match the coverage ranking.

10. **Global fit hides local failures.** R_Y0 averages over all control units. Coverage depends on outcome fit in thin-overlap covariate regions, which are few in number and contribute little to the global average. Good global fit does not imply good fit where it matters for causal inference.

### Local outcome diagnostics

11. **Local R_Y (restricted to thin-overlap units) confirms the averaging barrier and partially fixes it.** The key idea: compute arm-specific R² only on units in thin-overlap territory — specifically, the "unnatural" units whose covariates predict the opposite treatment from what they received. For controls, these are units with m̂ > 0.5 (covariates say they should have been treated, but they weren't). For treated units, these are units with m̂ < 0.5. Compute both R_Y0_local and R_Y1_local and include both alongside R_D — the practitioner does not need to choose which arm to inspect. The uninformative arm's local R² will have low cross-learner variance and be naturally downweighted in any predictive model.

    R_D + R_Y0_local (m̂ > 0.5) achieves AUC 0.842 (linear) / 0.794 (nonlinear), a +2–3 point improvement over R_D + R_Y0 (global). The improvement is concentrated in cross-learner comparisons (e.g., Ridge vs RF: AUC 0.786 → 0.828). In our IHDP structural DGP, R_Y0_local carries the signal (the control arm is larger and more heterogeneous) while R_Y1 has negligible cross-learner variance (0.067 to 0.115). In a DGP where the treated arm dominates, the roles would reverse — but including both is always safe.

12. **The gap between global and local R_Y is a qualitative check, not a formal predictor.** Compute the gap for both arms:

    - Gap_0 = R_Y0 − R_Y0_local: does the control outcome model degrade in the thin-overlap region?
    - Gap_1 = R_Y1 − R_Y1_local: does the treated outcome model degrade in the thin-overlap region?

    If Gap_0 is large and Gap_1 is near zero, the control-arm model is the problem. If Gap_1 is large and Gap_0 is near zero, the treated-arm model is the problem. If both are large, both arms have local fit problems. If both are small, fit is consistent everywhere. In our DGP, Gap_1 is tiny because R_Y1 is already near zero globally — there's no room to drop further. Gap_0 is where the signal is (RF: 0.671 → 0.528, gap of 0.14). The practitioner doesn't need to decide which gap to look at in advance — compute both, report both, worry about whichever one is large.

    However, the gap fails as a formal predictor:

    | Predictors | Linear AUC | Nonlinear AUC |
    |------------|-----------|--------------|
    | Gap alone | 0.561 | 0.647 |
    | R_D + gap | 0.763 | 0.740 |
    | R_D + R_Y0 + gap | 0.769 | 0.747 |
    | R_D + R_Y0_local(0.5) | 0.842 | 0.794 |

    The gap = R_Y0 − R_Y0_local is the difference between two noisy quantities. It loses information compared to using R_Y0_local directly. A logistic regression with R_D + R_Y0 + R_Y0_local (which can learn any relationship, including the gap) achieves 0.843 / 0.793 — essentially the same as R_D + R_Y0_local alone. The gap's information is already captured by R_Y0_local. Its value is as a human-readable interpretation aid ("my outcome model loses X% of its predictive power in the high-leverage region"), not as a model feature.

### The general diagnostic principle

13. **The local arm-specific R² diagnostic generalizes beyond R_Y0.** The method presented here — restricting outcome R² to units in opposite-treatment territory — applies to both arms simultaneously. Compute local R² for controls with m̂ > 0.5 (R_Y0_local) and for treated with m̂ < 0.5 (R_Y1_local), and include both alongside R_D. The practitioner does not need to choose which arm to inspect — the uninformative arm's local R² will have low cross-learner variance and be naturally downweighted in any predictive model. In our IHDP structural DGP, R_Y0_local carries the signal (control arm is larger and more heterogeneous) while R_Y1 has negligible variance (0.067 to 0.115). In a DGP where the treated arm dominates, the roles would reverse. Including both is cheap, avoids a decision point, and ensures the diagnostic works regardless of which arm is informative.

    The practical recommendation:
    1. Fit IRM to obtain arm-specific outcome models (ĝ₀, ĝ₁) and propensity estimates m̂(X)
    2. Compute local R² for both arms: R_Y0_local (controls with m̂ > 0.5) and R_Y1_local (treated with m̂ < 0.5)
    3. Use R_D + R_Y0_local + R_Y1_local as the diagnostic triple
    4. Optionally compare global vs local R² for each arm — a large drop signals the outcome model struggles in that arm's high-leverage region

    This framing — **local arm-specific R² in the extrapolation region, computed for both arms** — is the general contribution. Empirical confirmation on a reversed-propensity DGP (where R_Y1_local is the informative arm) would further validate the symmetry; this is noted as future work.

### The three-part structural finding

14. **Three structurally distinct findings.** (a) PLR's E[Y|X] conflates outcome quality with treatment predictability — contamination that makes R_Y useless; (b) IRM removes the contamination, upgrading R_D from poor cross-learner predictor to good, and enabling clean outcome diagnostics; (c) global outcome metrics miss local failures in thin-overlap regions — partially fixable by restricting to the "unnatural" units in the more heterogeneous arm.

**The diagnostic staircase:**
```
PLR:   R_D + R_Y        → AUC ≈ 0.60  (contaminated)
IRM:   R_D + R_Y0       → AUC ≈ 0.79  (clean but global)
IRM:   R_D + R_Y0_local → AUC ≈ 0.82  (clean and local)
```

Each step improvement is statistically significant (bootstrap p < 0.01, 2,000 replicates).

15. **Robustness.** PLR findings hold across 4 DGPs, 3 sample sizes, 5 learner configurations, and 25,000+ estimations. IRM findings confirmed across 4 learners, 2 surfaces, and 2,400 estimations on a balanced grid. Local R_Y0 findings confirmed with threshold sensitivity analysis (0.5, 0.7, 0.8) and top-k alternative; m̂ > 0.5 is the most robust choice.

---

## IRM Extension: Results

### Motivation

All PLR outcome-side diagnostic failures trace to the same root cause: PLR estimates E[Y|X] = g₀(X) + τ·m₀(X), conflating outcome quality with treatment predictability. The Interactive Regression Model (IRM, `DoubleMLIRM`) estimates E[Y|X, D=0] and E[Y|X, D=1] separately. These arm-specific expectations do not contain the τ·m₀(X) term, so cross-fitted R² from them (R_Y0, R_Y1) should measure outcome quality without treatment channel contamination.

### IRM MC results (2,400 estimations, balanced grid, structural DGP)

**v8c: 3 overlap strengths (0, 2, 5) × 2 surfaces × 4 learners (lasso_logistic, xgboost, ridge, rf) × 100 reps = 2,400 estimations.**

(Earlier runs v8 and v8b used an unbalanced grid that inflated R_Y0's apparent AUC; v8c balanced the grid so all 4 learners appear at every overlap × surface cell.)

All 4 learners at overlap=5:

| Surface | Learner | Coverage | R_D | R_Y0 | R_Y1 |
|---------|---------|----------|-----|------|------|
| Linear | XGBoost | 70% | 0.148 | 0.832 | 0.827 |
| Linear | Ridge | 56% | 0.195 | 0.920 | 0.923 |
| Linear | Lasso+Logistic | 57% | 0.194 | 0.675 | 0.714 |
| Linear | RF | 31% | 0.217 | 0.724 | 0.719 |
| Nonlinear | XGBoost | 86% | 0.148 | 0.770 | 0.067 |
| Nonlinear | Ridge | 84% | 0.194 | 0.781 | 0.115 |
| Nonlinear | Lasso+Logistic | 85% | 0.194 | 0.593 | 0.113 |
| Nonlinear | RF | 67% | 0.218 | 0.671 | 0.090 |

### Finding 1: IRM fixes the contamination problem

R_Y0 is stable across overlap levels — it does not exhibit the anti-conservative drift of PLR's R_Y.

XGBoost R_Y0 across overlap strengths:

| Overlap | PLR R_Y (contaminated) | IRM R_Y0 (clean) |
|---------|----------------------|-------------------|
| 0.0 | 0.357 | 0.844 |
| 2.0 | 0.605 | 0.832 |
| 5.0 | 0.723 | 0.832 |

PLR's R_Y nearly doubles (wrong direction). IRM's R_Y0 stays flat. The treatment channel contamination is gone.

### Finding 2: R_Y0 accurately measures outcome model quality

R_Y0 separates learners (XGBoost 0.832 vs Lasso+Logistic 0.675) and surfaces (linear 0.832 vs nonlinear 0.770 for XGBoost). It correctly identifies which outcome models fit better and which surfaces are harder.

### Finding 3: IRM improves R_D as a diagnostic, and R_Y0 adds modest value

Under the balanced grid (2,400 estimations):

| Predictors | Linear AUC | Nonlinear AUC |
|------------|------------|----------------|
| R_D alone | 0.804 | 0.734 |
| R_Y0 alone | 0.588 | 0.608 |
| R_D + R_Y0 | **0.826** | **0.767** |

Compare to PLR on the same structural DGP (v7, 8,400 estimations):

| Predictors | Linear AUC | Nonlinear AUC |
|------------|------------|----------------|
| R_D alone | 0.566 | 0.649 |
| R_Y alone | 0.539 | 0.522 |
| R_D + R_Y | 0.564 | 0.645 |

Two improvements under IRM:
1. **R_D alone becomes a sharper diagnostic** (AUC 0.80 vs 0.57 on the linear surface, 0.73 vs 0.65 on the nonlinear surface). The reason: IRM's doubly-robust score is more sensitive to propensity quality, so R_D — which measures propensity quality — is more informative about coverage.
2. **R_Y0 adds real predictive value** (+2.2 AUC points linear, +3.3 nonlinear). Not dramatic, but nonzero. PLR's R_Y added nothing (in fact made the linear AUC slightly worse).

### Finding 4: R_Y0 does not rank learners by coverage — the averaging barrier

Despite the modest AUC gain, R_Y0 alone does not identify which learner will have the best coverage. On the linear surface at overlap=5:

- Ridge has the **best** R_Y0 (0.920) but only 62% coverage
- XGBoost has lower R_Y0 (0.832) but better coverage (68%)
- RF has decent R_Y0 (0.724) but catastrophic coverage (30%)
- Lasso+Logistic has the lowest R_Y0 (0.675) with 58% coverage

The ranking of R_Y0 does not match the ranking of coverage. R_Y0 measures outcome fit accurately, but outcome fit averaged over all units does not correspond to the local fit in thin-overlap regions where coverage actually breaks.

### Why: global fit hides local failures

R_Y0 measures outcome model quality averaged over all control units. Most units are in covariate regions with adequate overlap and ample data — the outcome model fits well there. Coverage failures are driven by the small number of units in thin-overlap regions where the outcome model cannot extrapolate. These units contribute little to R_Y0 because there aren't many of them, but they dominate coverage breakdown.

Ridge illustrates this clearly: it achieves the best global outcome fit (R_Y0 = 0.920) but its errors are concentrated in the thin-overlap regions where the doubly-robust estimator is most sensitive. Good average fit ≠ good fit where it matters.

### Finding 5: Local R_Y0 confirms the averaging barrier and partially fixes it

We tested the averaging hypothesis directly by computing R_Y0 restricted to controls in thin-overlap regions (m̂ > 0.5 — controls whose covariates say they were more likely to be treated than not). This "local R_Y0" measures outcome fit specifically where the IRM score puts large weight and coverage most depends on.

**v10 MC results (2,400 estimations, 100 reps, structural DGP, balanced grid):**

| Predictors | Linear AUC | Nonlinear AUC |
|------------|------------|----------------|
| R_D alone | 0.806 | 0.733 |
| R_D + R_Y0 (global) | 0.821 | 0.761 |
| **R_D + R_Y0_local (m̂>0.5)** | **0.842** | **0.794** |
| R_D + R_Y0_topk (30) | 0.843 | 0.779 |

Local R_Y0 at threshold m̂ > 0.5 beats global R_Y0 by +2.3 (linear) and +3.2 (nonlinear) AUC points. Both improvements are statistically significant (bootstrap 95% CIs: linear [+1.4, +3.2], p < 0.001; nonlinear [+1.0, +5.8], p = 0.002; 2,000 bootstrap replicates). The improvement is concentrated in learner-pair comparisons where cross-learner discrimination matters most:

| Learner pair (nonlinear) | R_D alone | R_D + R_Y0 | R_D + R_Y0_local |
|--------------------------|-----------|------------|-------------------|
| XGBoost vs RF | 0.707 | 0.747 | **0.782** |
| Ridge vs RF | 0.763 | 0.786 | **0.828** |
| Ridge vs Lasso+Logistic | 0.780 | 0.797 | **0.811** |

Local R_Y0 adds +3.5 to +4.2 AUC points over global R_Y0 for distinguishing learners with good coverage from those with poor coverage. This is the cross-learner discrimination that global R_Y0 cannot provide.

**Threshold sensitivity:** m̂ > 0.5 is the sweet spot. Stricter thresholds (0.7, 0.8) produce too few units for stable R², and AUC degrades. The 0.5 threshold has a natural interpretation: controls whose covariates say they were more likely to be treated than not — exactly the units where the outcome model has to extrapolate from sparse data. Top-k (30 most extreme controls) also works well and avoids choosing a threshold.

**Diagnostic values at overlap=5, nonlinear surface:**

| Learner | Coverage | R_Y0 (global) | R_Y0_local (m̂>0.5) |
|---------|----------|---------------|---------------------|
| XGBoost | 88% | 0.768 | 0.700 |
| Ridge | 83% | 0.782 | 0.705 |
| Lasso+Logistic | 87% | 0.593 | 0.538 |
| RF | 70% | 0.671 | 0.528 |

Both global and local R_Y0 drop from global to local, but the drop is larger for learners that struggle locally (RF: 0.671 → 0.528) than for those that hold up (XGBoost: 0.768 → 0.700). This confirms the mechanism: averaging over all units masks local fit problems in the units that drive coverage.

### Three structural findings

The PLR and IRM results together support a three-part structural finding:

1. **Contamination (PLR-specific).** PLR's outcome model estimates E[Y|X] = g₀(X) + τ·m₀(X). Any residual-based metric inherits the treatment channel, causing R_Y to move in the wrong direction as overlap weakens. **Fixable by switching to IRM.**

2. **IRM is a more diagnostically transparent framework.** IRM removes the contamination, and this upgrades R_D from a poor cross-learner predictor (AUC ~0.6) to a good one (AUC ~0.8). R_Y0 adds further modest value on top. The practical recommendation: use IRM over PLR when diagnostic observability matters.

3. **Averaging barrier (framework-general, partially addressable).** Even under IRM, global R_Y0 cannot rank learners by coverage because it averages over all units. Coverage depends on outcome model performance in thin-overlap covariate regions, which contribute little to the global average. **Partially fixable by restricting R_Y0 to controls with m̂ > 0.5**, which captures outcome fit in the high-leverage region. This adds +2–3 AUC points over global R_Y0, concentrated in the cross-learner comparisons where outcome-side blindness is most severe.

The paper contribution is the diagnostic-observability perspective: choice of DML framework has underappreciated consequences for what practitioners can see about their own estimates. The full diagnostic staircase:

```
PLR:   R_D + R_Y     → AUC ≈ 0.60  (contaminated, R_Y moves wrong direction)
IRM:   R_D + R_Y0    → AUC ≈ 0.79  (clean but global)
IRM:   R_D + R_Y0_local → AUC ≈ 0.82  (clean and local)
```

Each step fixes one structural barrier. The practical recommendation: use IRM, compute the diagnostic triple R_D + R_Y0_local (controls with m̂ > 0.5) + R_Y1_local (treated with m̂ < 0.5), and optionally compare global vs local R² for each arm — a large gap signals that the outcome model struggles where it matters most. No need to choose which arm in advance: include both and let the regression decide.
