"""Generate IHDP-based semi-synthetic datasets for the DML overlap project."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from clean_ihdp import EXAMPLE_COVARIATE_COLUMNS, SIM_COVARIATE_COLUMNS


# ---------------------------------------------------------------------------
# Hill (2011) response-surface covariate classification
# Response surfaces use 25 covariates (ethnicity excluded), split into
# 6 continuous (standardized) and 19 binary (left as-is).
# ---------------------------------------------------------------------------
SIM_CONTINUOUS_COVARIATES = [
    "bw", "b.head", "preterm", "birth.o", "nnhealth", "momage",
]
SIM_BINARY_COVARIATES = [
    "sex", "twin", "b.marr", "mom.lths", "mom.hs", "mom.scoll",
    "cig", "first", "booze", "drugs", "work.dur", "prenatal",
    "ark", "ein", "har", "mia", "pen", "tex", "was",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def zscore(series: pd.Series) -> np.ndarray:
    std = series.std(ddof=1)
    if std == 0:
        return np.zeros(len(series))
    return ((series - series.mean()) / std).to_numpy()


def detect_schema(df: pd.DataFrame) -> str:
    columns = set(df.columns)
    if set(EXAMPLE_COVARIATE_COLUMNS).issubset(columns):
        return "example"
    if set(SIM_COVARIATE_COLUMNS).issubset(columns):
        return "sim"
    raise ValueError("Input file does not match the supported example or sim IHDP schemas.")


def covariate_columns_for_schema(schema: str) -> list[str]:
    if schema == "example":
        return EXAMPLE_COVARIATE_COLUMNS
    if schema == "sim":
        return SIM_COVARIATE_COLUMNS
    raise ValueError(f"Unknown schema: {schema}")


def build_design_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build the [1 | X] design matrix following Hill (2011) Section 4.1.

    Continuous covariates are standardized to mean 0, sd 1 (sample sd with
    ddof=1 to match R's default).  Binary covariates are left as-is.
    An intercept column of 1s is prepended.  Ethnicity columns (momwhite,
    momblack, momhisp) are excluded, giving 1 + 25 = 26 columns.
    """
    n = len(df)
    parts = [np.ones((n, 1))]

    for col in SIM_CONTINUOUS_COVARIATES:
        x = df[col].to_numpy(dtype=float)
        std = x.std(ddof=1)
        if std == 0:
            parts.append(np.zeros((n, 1)))
        else:
            parts.append(((x - x.mean()) / std).reshape(-1, 1))

    for col in SIM_BINARY_COVARIATES:
        parts.append(df[col].to_numpy(dtype=float).reshape(-1, 1))

    return np.hstack(parts)


def build_treatment_probability(
    df: pd.DataFrame,
    overlap_strength: float,
    schema: str,
    propensity_model: str = "logistic",
) -> np.ndarray:
    if schema != "sim":
        raise ValueError(
            "Propensity model requires the sim schema. "
            "Use hill_data/ihdp_sim.csv or processed/ihdp_sim_processed.csv."
        )
    if propensity_model == "logistic":
        return _propensity_logistic(df, overlap_strength)
    if propensity_model == "structural":
        return _propensity_structural(df, overlap_strength)
    if propensity_model == "highdim":
        return _propensity_highdim(df, overlap_strength)
    if propensity_model == "threshold":
        return _propensity_threshold(df, overlap_strength)
    raise ValueError(f"Unknown propensity_model: {propensity_model}")


def _propensity_logistic(
    df: pd.DataFrame,
    overlap_strength: float,
) -> np.ndarray:
    """Original logistic propensity model using 4 clinical covariates."""
    score = (
        -0.75 * zscore(df["bw"])
        - 0.35 * zscore(df["nnhealth"])
        - 0.20 * zscore(df["momage"])
        + 0.30 * df["b.marr"].to_numpy()
    )
    score = (score - score.mean()) / score.std(ddof=0)
    intercept = np.log(0.38 / (1.0 - 0.38))
    logits = intercept + overlap_strength * score
    probs = sigmoid(logits)
    return np.clip(probs, 0.001, 0.999)


def _propensity_structural(
    df: pd.DataFrame,
    overlap_strength: float,
) -> np.ndarray:
    """Structural propensity model inspired by Hill (2011).

    Non-white mothers: propensity depends on birthweight via a logistic
    function with intercept and slope both scaled by overlap_strength.
    Sicker infants (lower bw) get higher propensity.

    White mothers: propensity is low and flat, dropping with
    overlap_strength.

    At overlap_strength = 0, all units get p = 0.38 (random).
    As overlap_strength increases, treatment becomes increasingly
    determined by race and birthweight.
    """
    n = len(df)
    bw_z = zscore(df["bw"])
    nonwhite = (df["momwhite"] == 0).to_numpy()
    white = ~nonwhite

    baseline_logit = np.log(0.38 / (1.0 - 0.38))

    # Non-white: logit shifts up (higher base propensity) and gets a
    # negative birthweight slope (lower bw → higher propensity).
    # Both intercept shift and slope scale with overlap_strength.
    nonwhite_logits = baseline_logit + overlap_strength * (0.5 - 1.5 * bw_z[nonwhite])

    # White: logit shifts down (lower base propensity).
    white_logits = baseline_logit - overlap_strength * 1.0

    probs = np.empty(n)
    probs[nonwhite] = sigmoid(nonwhite_logits)
    probs[white] = sigmoid(white_logits)

    return np.clip(probs, 0.001, 0.999)


def _propensity_highdim(
    df: pd.DataFrame,
    overlap_strength: float,
) -> np.ndarray:
    """High-dimensional propensity with interactions across 6 covariates.

    Uses all 6 continuous covariates plus 3 pairwise interactions.
    The main effects are linear (Lasso-friendly) but the interactions
    are multiplicative (require flexibility to capture).

    At overlap_strength = 0, all units get p = 0.38 (random).
    """
    bw_z = zscore(df["bw"])
    head_z = zscore(df["b.head"])
    preterm_z = zscore(df["preterm"])
    nnhealth_z = zscore(df["nnhealth"])
    momage_z = zscore(df["momage"])
    birtho_z = zscore(df["birth.o"])

    # Main effects (linear)
    main = -0.4 * bw_z - 0.3 * nnhealth_z - 0.2 * momage_z + 0.15 * preterm_z

    # Interactions (nonlinear — Lasso can't capture these)
    interactions = (
        0.5 * bw_z * nnhealth_z
        + 0.4 * preterm_z * momage_z
        + 0.3 * head_z * birtho_z
    )

    score = main + interactions
    score = (score - score.mean()) / (score.std(ddof=0) + 1e-8)

    baseline_logit = np.log(0.38 / (1.0 - 0.38))
    logits = baseline_logit + overlap_strength * score
    probs = sigmoid(logits)
    return np.clip(probs, 0.001, 0.999)


def _propensity_threshold(
    df: pd.DataFrame,
    overlap_strength: float,
) -> np.ndarray:
    """Threshold/step-function propensity based on birthweight and marriage.

    Treatment probability jumps at birthweight thresholds, modulated by
    marital status. This creates a discontinuous propensity surface that
    neither Lasso nor simple tree models capture perfectly.

    At overlap_strength = 0, all units get p = 0.38 (random).
    """
    bw_z = zscore(df["bw"])
    married = df["b.marr"].to_numpy(dtype=float)
    nnhealth_z = zscore(df["nnhealth"])

    # Step function: low bw gets high propensity, medium gets moderate, high gets low
    score = np.zeros(len(df))
    score[bw_z < -0.5] = 1.0   # low birthweight → high treatment prob
    score[bw_z > 0.5] = -1.0   # high birthweight → low treatment prob
    # middle stays at 0

    # Married modulates: married + low bw gets even higher propensity
    score += 0.5 * married * (bw_z < 0).astype(float)

    # Continuous component from nnhealth adds some smoothness
    score -= 0.3 * nnhealth_z

    score = (score - score.mean()) / (score.std(ddof=0) + 1e-8)

    baseline_logit = np.log(0.38 / (1.0 - 0.38))
    logits = baseline_logit + overlap_strength * score
    probs = sigmoid(logits)
    return np.clip(probs, 0.001, 0.999)


def generate_surface_A(
    design: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Hill (2011) Response Surface A: linear, parallel, constant tau = 4.

    Y(0) ~ N(X @ beta, 1)
    Y(1) ~ N(X @ beta + 4, 1)

    beta drawn from {0, 1, 2, 3, 4} with probabilities {0.5, 0.2, 0.15, 0.1, 0.05}.
    Coefficients are re-drawn each simulation run.
    """
    n, p = design.shape
    beta = rng.choice(
        [0, 1, 2, 3, 4],
        size=p,
        p=[0.50, 0.20, 0.15, 0.10, 0.05],
    ).astype(float)
    mu = design @ beta
    y0 = rng.normal(mu, 1.0)
    tau = np.full(n, 4.0)
    return y0, tau


def generate_surface_B(
    design: np.ndarray,
    rng: np.random.Generator,
    treatment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Hill (2011) Response Surface B: nonlinear Y(0), linear Y(1).

    Y(0) ~ N(exp([1 | X+0.5] @ beta), 1)
    Y(1) ~ N([1 | X+0.5] @ beta - omega, 1)

    beta drawn from {0, 0.1, 0.2, 0.3, 0.4} with probabilities
    {0.6, 0.1, 0.1, 0.1, 0.1}.  omega is calibrated so that the sample
    average treatment effect (ATE) equals 4, matching the DML estimand.
    Coefficients are re-drawn each simulation run.
    """
    n, p = design.shape
    beta = rng.choice(
        [0.0, 0.1, 0.2, 0.3, 0.4],
        size=p,
        p=[0.6, 0.1, 0.1, 0.1, 0.1],
    )

    # Shift covariate columns by 0.5 (intercept column stays as-is)
    design_shifted = design.copy()
    design_shifted[:, 1:] += 0.5

    linear_term = design_shifted @ beta
    mu_y0 = np.exp(linear_term)
    mu_y1_raw = linear_term

    # Calibrate omega so ATE = 4 over all units
    omega = (mu_y1_raw - mu_y0).mean() - 4.0

    mu_y1 = mu_y1_raw - omega
    y0 = rng.normal(mu_y0, 1.0)
    tau = mu_y1 - mu_y0
    return y0, tau


def sample_base_data(
    df: pd.DataFrame,
    sample_size: int | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if sample_size is None:
        sample_size = len(df)
    if sample_size < 2:
        raise ValueError("sample_size must be at least 2.")
    if sample_size == len(df):
        sampled = df.reset_index(drop=True).copy()
        sampled["source_row"] = df.index.to_numpy()
        return sampled

    replace = sample_size > len(df)
    indices = rng.choice(df.index.to_numpy(), size=sample_size, replace=replace)
    sampled = df.loc[indices].reset_index(drop=True).copy()
    sampled["source_row"] = indices
    return sampled


def generate_dataset(
    df: pd.DataFrame,
    sample_size: int | None,
    overlap_strength: float,
    complexity: str,
    seed: int,
    propensity_model: str = "logistic",
) -> pd.DataFrame:
    schema = detect_schema(df)
    if schema != "sim":
        raise ValueError(
            "Hill (2011) response surfaces require the sim schema. "
            "Use hill_data/ihdp_sim.csv or processed/ihdp_sim_processed.csv."
        )
    rng = np.random.default_rng(seed)
    sampled = sample_base_data(df, sample_size=sample_size, rng=rng)
    propensity = build_treatment_probability(
        sampled,
        overlap_strength=overlap_strength,
        schema=schema,
        propensity_model=propensity_model,
    )
    treatment = rng.binomial(1, propensity)

    design = build_design_matrix(sampled)
    if complexity == "linear":
        y0, tau = generate_surface_A(design, rng)
    elif complexity == "nonlinear":
        y0, tau = generate_surface_B(design, rng, treatment)
    else:
        raise ValueError(f"Unknown complexity: {complexity}")

    y1 = y0 + tau
    y = np.where(treatment == 1, y1, y0)

    out = sampled.copy()
    out["sim_treat"] = treatment
    out["propensity"] = propensity
    out["y0"] = y0
    out["y1"] = y1
    out["true_tau"] = tau
    out["y"] = y
    out["sample_size"] = len(sampled)
    out["overlap_strength"] = overlap_strength
    out["complexity"] = complexity
    out["seed"] = seed
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="processed/ihdp_sim_processed.csv",
        help="Path to the cleaned IHDP CSV.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to a parameterized CSV in processed/.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Rows to sample without replacement. Defaults to using all rows in the cleaned IHDP file.",
    )
    parser.add_argument("--overlap-strength", type=float, default=0.0)
    parser.add_argument(
        "--complexity",
        choices=["linear", "nonlinear"],
        default="linear",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--propensity-model",
        choices=["logistic", "structural", "highdim", "threshold"],
        default="logistic",
    )
    return parser.parse_args()


def default_output_path(args: argparse.Namespace) -> Path:
    sample_size_slug = "full" if args.sample_size is None else str(args.sample_size)
    overlap_slug = str(args.overlap_strength).replace(".", "p")
    name = (
        f"ihdp_synth_n{sample_size_slug}_a{overlap_slug}_"
        f"{args.complexity}_seed{args.seed}.csv"
    )
    return Path("processed") / name


def main() -> None:
    args = parse_args()
    base_df = pd.read_csv(args.input)
    schema = detect_schema(base_df)
    required = set(covariate_columns_for_schema(schema))
    missing = sorted(required.difference(base_df.columns))
    if missing:
        raise ValueError(f"Cleaned IHDP file is missing covariates: {missing}")

    output_path = Path(args.output) if args.output else default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    synthetic = generate_dataset(
        base_df,
        sample_size=args.sample_size,
        overlap_strength=args.overlap_strength,
        complexity=args.complexity,
        seed=args.seed,
        propensity_model=args.propensity_model,
    )
    synthetic.to_csv(output_path, index=False)

    print(f"Wrote synthetic dataset to {output_path}")
    print(f"Shape: {synthetic.shape}")
    print(f"Mean synthetic treatment: {synthetic['sim_treat'].mean():.3f}")
    print(f"Mean propensity: {synthetic['propensity'].mean():.3f}")
    print(f"Min/max propensity: {synthetic['propensity'].min():.3f}, {synthetic['propensity'].max():.3f}")
    print(f"Mean true tau: {synthetic['true_tau'].mean():.3f}")


if __name__ == "__main__":
    main()
