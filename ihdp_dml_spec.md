# IHDP DML Simulation Specification

## Purpose

This note defines the recommended canonical IHDP variable set for the new DML overlap project.

The project goal is not to reuse the legacy section 4 or section 6 analyses directly. Instead, it is to use the empirical distribution of IHDP covariates as the base for a new semi-synthetic simulation engine with explicit knobs for:

- overlap strength
- model complexity

Monte Carlo seeds are used to index repeated simulation draws within each design setting rather than to define separate settings.

## Source Files

- `hill_data/example.data` loads an R data frame called `ihdp`
- `hill_data/sim.data` loads an R object called `imp1`
- `hill_code/example1.code.R` and `hill_code/example2.code.R` show how the IHDP example data were used in older analyses
- `hill_data/imp1.vars.sim.data.doc` and `hill_data/ihdp.vars.exmp.data.doc` document the variables

## Recommended Active Dataset

Use `hill_data/ihdp_sim.csv` as the active source for the current project pipeline.

Reason:

- it matches the schema now used by the processed and synthetic files in this repo
- it keeps the active workflow focused on a compact, stable covariate set
- it avoids carrying the extra 34-covariate example-based representation that the repo no longer uses as its default

The active cleaning step should load `hill_data/ihdp_sim.csv` and export `processed/ihdp_sim_processed.csv`.

## Role Of Variables In The New Project

### Variables to keep as metadata or benchmarks

- `treat`

The active sim-based path keeps `treat` as the observed treatment indicator from `ihdp_sim.csv`. The current pipeline does not depend on an observed outcome variable in the source data because synthetic outcomes are generated later.

### Active covariate set `X`

The active pipeline uses the original `ihdp_sim` covariate schema:

- `bw`
- `b.head`
- `preterm`
- `birth.o`
- `nnhealth`
- `momage`
- `sex`
- `twin`
- `b.marr`
- `mom.lths`
- `mom.hs`
- `mom.scoll`
- `cig`
- `first`
- `booze`
- `drugs`
- `work.dur`
- `prenatal`
- `ark`
- `ein`
- `har`
- `mia`
- `pen`
- `tex`
- `was`
- `momwhite`
- `momblack`
- `momhisp`

This gives 28 baseline covariates plus `treat` in the processed sim-based file.

## Why This Variable Set

This specification keeps the active workflow aligned with the compact simulation dataset already bundled in the Hill materials and avoids carrying multiple competing processed schemas inside the repo.

The richer `example.data` file is still useful as reference material, but it is no longer the active default because:

1. It introduces an additional 34-covariate representation that the current repo is not using.
2. It includes legacy analysis-specific columns and encodings that are not needed for the active sim-based path.

## Recommended Variable Roles In The New Engine

### Base empirical distribution

Use only the active sim-based covariate set `X` to define the empirical covariate distribution.

### Synthetic treatment

Generate a new treatment variable:

`D ~ Bernoulli(logit^{-1}(intercept + alpha * s(X)))`

where `alpha` controls overlap strength. The overlap mechanism is an open design question. The current implementation uses 15 covariates with hardcoded weights. It is being redesigned to use a smaller set of clinically interpretable covariates (e.g., birthweight, neonatal health, prematurity) so that the story of why overlap weakens is clear.

### Synthetic outcomes

Potential outcomes follow the response surfaces from Hill (2011), using 25 covariates (ethnicity excluded), with the 6 continuous covariates standardized and the 19 binary covariates left as-is. An intercept column is prepended, giving a 26-column design matrix.

**Surface A** (linear, constant ATE = 4):

- `Y(0) ~ N([1|X] @ beta, 1)`, `Y(1) ~ N([1|X] @ beta + 4, 1)`
- beta drawn from `{0,1,2,3,4}` with probs `{0.5,0.2,0.15,0.1,0.05}`, re-drawn each run

**Surface B** (nonlinear Y(0), linear Y(1), heterogeneous effects, ATE calibrated to 4):

- `Y(0) ~ N(exp([1|X+0.5] @ beta), 1)`, `Y(1) ~ N([1|X+0.5] @ beta - omega, 1)`
- beta drawn from `{0,0.1,0.2,0.3,0.4}` with probs `{0.6,0.1,0.1,0.1,0.1}`, re-drawn each run
- omega calibrated so sample ATE = 4 each run

## Initial Design Recommendation

Use the following project defaults:

- active source dataset: `hill_data/ihdp_sim.csv`
- processed baseline dataset: `processed/ihdp_sim_processed.csv`
- treatment benchmark variable: `treat`
- active covariate matrix: the 28-variable `ihdp_sim` covariate set
- sample size fixed at `985`
- overlap grid `{0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}`
- complexity grid `{linear, nonlinear}`
- `200` Monte Carlo iterations per setting
- total dataset count `7 x 2 x 200 = 2,800`
- seeds used as Monte Carlo iteration indices rather than as a separate substantive design parameter

## Current State

Completed:

- `hill_data/ihdp_sim.csv` has 985 rows and 29 columns (28 covariates + `treat`)
- `processed/ihdp_sim_processed.csv` keeps `treat` plus the 28 sim covariates
- Response surfaces A and B from Hill (2011) are implemented in `src/generate_ihdp_synthetic.py`
- ATE calibrated to 4 for both surfaces (constant for A, via omega offset for B)

Open:

- Overlap mechanism needs redesign for interpretability
- DML estimation runner, overlap diagnostics (`R_D`), Monte Carlo grid runner not yet implemented
