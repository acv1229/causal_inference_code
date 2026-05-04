"""Validate IRM local-R2 diagnostics on the LaLonde covariate dataset.

The validation uses LaLonde data only as an external covariate distribution.
Treatment and outcomes are generated semi-synthetically so the true ATE is
known and coverage can be measured.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from dml_simulation import IRM_LEARNER_FACTORIES, LEARNER_FACTORIES, estimate_ate, estimate_ate_irm


TRUE_ATE = 4.0
DEFAULT_OVERLAP_STRENGTHS = [0.0, 2.0, 4.0]
DEFAULT_SURFACES = ["linear", "localized_nonlinear"]
DEFAULT_LEARNERS = ["ridge", "lasso_logistic", "rf"]
LALONDE_COVARIATES = [
    "age",
    "educ",
    "black",
    "hispan",
    "married",
    "nodegree",
    "re74",
    "re75",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _zscore(values: pd.Series) -> np.ndarray:
    x = values.to_numpy(dtype=float)
    sd = x.std()
    if sd == 0:
        return np.zeros(len(values))
    return (x - x.mean()) / sd


def load_lalonde_covariates(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    missing = sorted(set(LALONDE_COVARIATES).difference(raw.columns))
    if missing:
        raise ValueError(f"LaLonde file is missing covariates: {missing}")

    out = raw[LALONDE_COVARIATES].copy()
    for col in ["age", "educ", "re74", "re75"]:
        out[col] = _zscore(out[col])
    return out.reset_index(drop=True)


def _base_score(x: np.ndarray) -> np.ndarray:
    age, educ, black, hispan, married, nodegree, re74, re75 = range(x.shape[1])
    score = (
        -0.60 * x[:, age]
        - 0.35 * x[:, educ]
        + 0.90 * x[:, black]
        + 0.45 * x[:, hispan]
        - 0.35 * x[:, married]
        + 0.40 * x[:, nodegree]
        - 0.60 * x[:, re74]
        - 0.50 * x[:, re75]
        + 0.35 * x[:, black] * (x[:, re74] < 0).astype(float)
    )
    return (score - score.mean()) / (score.std() + 1e-8)


def _propensity(score: np.ndarray, overlap_strength: float) -> np.ndarray:
    baseline_logit = np.log(0.30 / 0.70)
    logits = baseline_logit + overlap_strength * score
    return np.clip(_sigmoid(logits), 0.001, 0.999)


def _outcome_mean(x: np.ndarray, score: np.ndarray, surface: str) -> np.ndarray:
    age, educ, black, hispan, married, nodegree, re74, re75 = range(x.shape[1])
    linear = (
        -0.40 * x[:, age]
        + 0.70 * x[:, educ]
        - 0.55 * x[:, black]
        - 0.35 * x[:, hispan]
        + 0.30 * x[:, married]
        - 0.40 * x[:, nodegree]
        + 0.80 * x[:, re74]
        + 0.90 * x[:, re75]
    )
    if surface == "linear":
        return linear
    if surface == "localized_nonlinear":
        local_bump = _sigmoid(4.0 * (score - 0.6))
        nonlinear = 2.75 * local_bump * (
            x[:, re74] ** 2
            + 0.5 * np.sin(x[:, re75])
            + 0.75 * x[:, black] * x[:, nodegree]
        )
        return linear + nonlinear
    raise ValueError(f"Unknown surface: {surface}")


def generate_external_dataset(
    base_df: pd.DataFrame,
    overlap_strength: float,
    surface: str,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = base_df.to_numpy(dtype=float)
    score = _base_score(x)
    propensity = _propensity(score, overlap_strength)
    treatment = rng.binomial(1, propensity)

    mu0 = _outcome_mean(x, score, surface)
    y0 = mu0 + rng.normal(0.0, 1.0, size=len(base_df))
    y1 = y0 + TRUE_ATE
    y = np.where(treatment == 1, y1, y0)

    out = base_df.reset_index(drop=True).copy()
    out["sim_treat"] = treatment
    out["propensity"] = propensity
    out["y0"] = y0
    out["y1"] = y1
    out["true_tau"] = TRUE_ATE
    out["y"] = y
    out["sample_size"] = len(out)
    out["overlap_strength"] = overlap_strength
    out["complexity"] = surface
    out["seed"] = seed
    out["dataset"] = "lalonde"
    return out


def run_validation(
    n_reps: int,
    n_folds: int,
    overlap_strengths: list[float],
    surfaces: list[str],
    learners: list[str],
    start_seed: int,
    input_path: str | Path,
) -> pd.DataFrame:
    base_df = load_lalonde_covariates(input_path)
    rows = []
    total = len(overlap_strengths) * len(surfaces) * n_reps * len(learners)
    done = 0

    for alpha in overlap_strengths:
        for surface in surfaces:
            for seed in range(start_seed, start_seed + n_reps):
                data = generate_external_dataset(base_df, alpha, surface, seed)
                for learner in learners:
                    t0 = time.time()
                    result = estimate_ate_irm(
                        data,
                        learner=learner,
                        n_folds=n_folds,
                        true_ate=TRUE_ATE,
                    )
                    result["elapsed"] = time.time() - t0
                    result["dataset"] = "lalonde"
                    rows.append(result)
                    done += 1
                    if done % 25 == 0 or done == total:
                        print(
                            f"[{done}/{total}] alpha={alpha}, "
                            f"surface={surface}, learner={learner}, seed={seed}"
                        )

    return pd.DataFrame(rows)


def run_plr_validation(
    n_reps: int,
    n_folds: int,
    overlap_strengths: list[float],
    surfaces: list[str],
    learners: list[str],
    start_seed: int,
    input_path: str | Path,
) -> pd.DataFrame:
    base_df = load_lalonde_covariates(input_path)
    rows = []
    total = len(overlap_strengths) * len(surfaces) * n_reps * len(learners)
    done = 0

    for alpha in overlap_strengths:
        for surface in surfaces:
            for seed in range(start_seed, start_seed + n_reps):
                data = generate_external_dataset(base_df, alpha, surface, seed)
                for learner in learners:
                    t0 = time.time()
                    result = estimate_ate(
                        data,
                        learner=learner,
                        n_folds=n_folds,
                        true_ate=TRUE_ATE,
                    )
                    result["elapsed"] = time.time() - t0
                    result["dataset"] = "lalonde"
                    result["framework"] = "plr"
                    rows.append(result)
                    done += 1
                    if done % 25 == 0 or done == total:
                        print(
                            f"[PLR {done}/{total}] alpha={alpha}, "
                            f"surface={surface}, learner={learner}, seed={seed}"
                        )

    return pd.DataFrame(rows)


def _compute_auc(df: pd.DataFrame, features: list[str]) -> float:
    sub = df.dropna(subset=features + ["covers"])
    if len(sub) < 20 or sub["covers"].nunique() < 2:
        return np.nan
    model = LogisticRegression(max_iter=5000, random_state=0)
    model.fit(sub[features].to_numpy(), sub["covers"].to_numpy())
    proba = model.predict_proba(sub[features].to_numpy())[:, 1]
    return roc_auc_score(sub["covers"], proba)


def _bootstrap_auc_diff(
    df: pd.DataFrame,
    features_a: list[str],
    features_b: list[str],
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict:
    """Bootstrap CI for AUC(b) - AUC(a), paired on the same resampled rows."""
    all_features = list(dict.fromkeys(features_a + features_b + ["covers"]))
    sub = df.dropna(subset=all_features)
    if len(sub) < 20 or sub["covers"].nunique() < 2:
        return {"diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_gt_zero": np.nan}

    rng = np.random.default_rng(seed)
    diffs = []
    idx = np.arange(len(sub))
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(idx, size=len(idx), replace=True)
        boot = sub.iloc[boot_idx]
        if boot["covers"].nunique() < 2:
            continue
        y = boot["covers"].to_numpy()

        def _auc(feats: list[str]) -> float:
            X = boot[feats].to_numpy()
            m = LogisticRegression(max_iter=5000, random_state=0)
            m.fit(X, y)
            return roc_auc_score(y, m.predict_proba(X)[:, 1])

        try:
            diffs.append(_auc(features_b) - _auc(features_a))
        except Exception:
            continue

    diffs = np.array(diffs)
    point = _compute_auc(sub, features_b) - _compute_auc(sub, features_a)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_gt_zero = float((diffs > 0).mean())
    return {"diff": point, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_gt_zero": p_gt_zero}


def build_plr_auc_summary(plr_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for surface, group in plr_results.groupby("complexity"):
        rows.extend([
            {
                "dataset": "lalonde",
                "framework": "plr",
                "complexity": surface,
                "diagnostic": "R_D alone",
                "auc": _compute_auc(group, ["r_d"]),
            },
            {
                "dataset": "lalonde",
                "framework": "plr",
                "complexity": surface,
                "diagnostic": "R_D + R_Y (contaminated)",
                "auc": _compute_auc(group, ["r_d", "r_y"]),
            },
        ])
    return pd.DataFrame(rows)


def build_summary(
    results: pd.DataFrame, n_bootstrap: int = 2000
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    auc_rows = []
    staircase_rows = []

    # Staircase steps to test: (label, features_a, features_b)
    staircase_steps = [
        (
            "global R_Y adds to R_D",
            ["r_d"],
            ["r_d", "r_y0", "r_y1"],
        ),
        (
            "local(0.5) R_Y0+R_Y1 beats global R_Y",
            ["r_d", "r_y0", "r_y1"],
            ["r_d", "r_y0_local_05", "r_y1_local_05"],
        ),
        (
            "local(2x) R_Y0+R_Y1 beats global R_Y",
            ["r_d", "r_y0", "r_y1"],
            ["r_d", "r_y0_local_2x", "r_y1_local_2x"],
        ),
        (
            "local(2x) beats local(0.5)",
            ["r_d", "r_y0_local_05", "r_y1_local_05"],
            ["r_d", "r_y0_local_2x", "r_y1_local_2x"],
        ),
        (
            "local(0.5) adds to R_D (combined step)",
            ["r_d"],
            ["r_d", "r_y0_local_05", "r_y1_local_05"],
        ),
        (
            "local(2x) adds to R_D (combined step)",
            ["r_d"],
            ["r_d", "r_y0_local_2x", "r_y1_local_2x"],
        ),
    ]

    for surface, group in results.groupby("complexity"):
        auc_rows.extend([
            {
                "dataset": "lalonde",
                "framework": "irm",
                "complexity": surface,
                "diagnostic": "R_D alone",
                "auc": _compute_auc(group, ["r_d"]),
            },
            {
                "dataset": "lalonde",
                "framework": "irm",
                "complexity": surface,
                "diagnostic": "R_D + global R_Y0/R_Y1",
                "auc": _compute_auc(group, ["r_d", "r_y0", "r_y1"]),
            },
            {
                "dataset": "lalonde",
                "framework": "irm",
                "complexity": surface,
                "diagnostic": "R_D + local R_Y0 at 0.5",
                "auc": _compute_auc(group, ["r_d", "r_y0_local_05"]),
            },
            {
                "dataset": "lalonde",
                "framework": "irm",
                "complexity": surface,
                "diagnostic": "R_D + global R_Y0 + local R_Y0 at 0.5",
                "auc": _compute_auc(group, ["r_d", "r_y0", "r_y0_local_05"]),
            },
            {
                "dataset": "lalonde",
                "framework": "irm",
                "complexity": surface,
                "diagnostic": "R_D + local R_Y0/R_Y1 at 0.5",
                "auc": _compute_auc(group, ["r_d", "r_y0_local_05", "r_y1_local_05"]),
            },
            {
                "dataset": "lalonde",
                "framework": "irm",
                "complexity": surface,
                "diagnostic": "R_D + local R_Y0/R_Y1 2x-odds",
                "auc": _compute_auc(group, ["r_d", "r_y0_local_2x", "r_y1_local_2x"]),
            },
        ])

        for label, feat_a, feat_b in staircase_steps:
            boot = _bootstrap_auc_diff(group, feat_a, feat_b, n_bootstrap=n_bootstrap)
            staircase_rows.append({
                "dataset": "lalonde",
                "complexity": surface,
                "comparison": label,
                "auc_diff": boot["diff"],
                "ci_lo": boot["ci_lo"],
                "ci_hi": boot["ci_hi"],
                "p_gt_zero": boot["p_gt_zero"],
            })

    auc_summary = pd.DataFrame(auc_rows)
    staircase_summary = pd.DataFrame(staircase_rows)
    cell_summary = (
        results.assign(bias=results["estimate"] - TRUE_ATE)
        .groupby(["complexity", "overlap_strength", "learner"])
        .agg(
            coverage=("covers", "mean"),
            bias=("bias", "mean"),
            rmse=("estimate", lambda x: np.sqrt(((x - TRUE_ATE) ** 2).mean())),
            r_d=("r_d", "mean"),
            r_y0=("r_y0", "mean"),
            r_y1=("r_y1", "mean"),
            r_y0_local_05=("r_y0_local_05", "mean"),
            r_y1_local_05=("r_y1_local_05", "mean"),
            n_local_control_05=("n_local_control_05", "mean"),
            n_local_treated_05=("n_local_treated_05", "mean"),
            n=("estimate", "count"),
        )
        .reset_index()
    )
    return auc_summary, staircase_summary, cell_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate IRM local-R2 diagnostics on the LaLonde covariate dataset."
    )
    parser.add_argument("--input", default="processed/lalonde.csv")
    parser.add_argument("--n-reps", type=int, default=20)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--overlap-strengths",
        type=float,
        nargs="+",
        default=DEFAULT_OVERLAP_STRENGTHS,
    )
    parser.add_argument(
        "--surfaces",
        nargs="+",
        choices=DEFAULT_SURFACES,
        default=DEFAULT_SURFACES,
    )
    parser.add_argument(
        "--learners",
        nargs="+",
        choices=list(IRM_LEARNER_FACTORIES.keys()),
        default=DEFAULT_LEARNERS,
    )
    parser.add_argument(
        "--plr-learners",
        nargs="+",
        choices=list(LEARNER_FACTORIES.keys()),
        default=DEFAULT_LEARNERS,
    )
    parser.add_argument(
        "--output",
        default="output/external_validation/lalonde_irm_local_results.csv",
    )
    parser.add_argument(
        "--plr-output",
        default="output/external_validation/lalonde_plr_results.csv",
    )
    parser.add_argument(
        "--auc-output",
        default="output/external_validation/lalonde_irm_local_auc.csv",
    )
    parser.add_argument(
        "--staircase-output",
        default="output/external_validation/lalonde_irm_staircase_tests.csv",
    )
    parser.add_argument(
        "--cell-output",
        default="output/external_validation/lalonde_irm_local_cells.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    irm_results = run_validation(
        n_reps=args.n_reps,
        n_folds=args.n_folds,
        overlap_strengths=args.overlap_strengths,
        surfaces=args.surfaces,
        learners=args.learners,
        start_seed=args.start_seed,
        input_path=args.input,
    )

    plr_results = run_plr_validation(
        n_reps=args.n_reps,
        n_folds=args.n_folds,
        overlap_strengths=args.overlap_strengths,
        surfaces=args.surfaces,
        learners=args.plr_learners,
        start_seed=args.start_seed,
        input_path=args.input,
    )

    output_path = Path(args.output)
    plr_output_path = Path(args.plr_output)
    auc_path = Path(args.auc_output)
    staircase_path = Path(args.staircase_output)
    cell_path = Path(args.cell_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    irm_results.to_csv(output_path, index=False)
    plr_results.to_csv(plr_output_path, index=False)

    auc_summary, staircase_summary, cell_summary = build_summary(
        irm_results, n_bootstrap=args.n_bootstrap
    )
    plr_auc_summary = build_plr_auc_summary(plr_results)
    auc_summary = pd.concat([plr_auc_summary, auc_summary], ignore_index=True)

    auc_summary.to_csv(auc_path, index=False)
    staircase_summary.to_csv(staircase_path, index=False)
    cell_summary.to_csv(cell_path, index=False)

    print(f"\nWrote IRM results to {output_path}")
    print(f"Wrote PLR results to {plr_output_path}")
    print(f"Wrote AUC summary to {auc_path}")
    print(f"Wrote staircase tests to {staircase_path}")
    print(f"Wrote cell summary to {cell_path}")
    print("\nAUC summary (PLR then IRM):")
    print(auc_summary.round(4).to_string(index=False))
    print("\nStaircase bootstrap tests (2000 resamples, 95% CI):")
    print(staircase_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
