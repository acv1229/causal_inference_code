"""Monte Carlo grid runner for the IHDP DML overlap simulation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from generate_ihdp_synthetic import generate_dataset
from dml_simulation import estimate_ate, LEARNER_FACTORIES, estimate_ate_irm, IRM_LEARNER_FACTORIES


# ---------------------------------------------------------------------------
# Default grid
# ---------------------------------------------------------------------------
DEFAULT_OVERLAP_STRENGTHS = [0.0, 2.0, 3.0, 5.0, 7.0, 10.0]
DEFAULT_COMPLEXITIES = ["linear", "nonlinear"]
DEFAULT_LEARNERS = list(LEARNER_FACTORIES.keys())
DEFAULT_N_REPS = 200

# Fixed per-learner seed offsets so fold assignment is stable regardless of
# which subset of learners is passed to a given run.
_LEARNER_SEED_OFFSET: dict[str, int] = {
    "lasso": 0,
    "lasso_logistic": 1,
    "xgboost": 2,
    "ridge": 3,
    "rf": 4,
}


def run_monte_carlo(
    base_df: pd.DataFrame,
    overlap_strengths: list[float] = DEFAULT_OVERLAP_STRENGTHS,
    complexities: list[str] = DEFAULT_COMPLEXITIES,
    learners: list[str] = DEFAULT_LEARNERS,
    n_reps: int = DEFAULT_N_REPS,
    n_folds: int = 5,
    propensity_model: str = "logistic",
    start_seed: int = 1,
) -> pd.DataFrame:
    """Run the full Monte Carlo grid and return a results DataFrame."""
    results = []
    total = len(overlap_strengths) * len(complexities) * len(learners) * n_reps
    done = 0

    for alpha in overlap_strengths:
        for complexity in complexities:
            for seed in range(start_seed, start_seed + n_reps):
                synth = generate_dataset(
                    base_df,
                    sample_size=None,
                    overlap_strength=alpha,
                    complexity=complexity,
                    seed=seed,
                    propensity_model=propensity_model,
                )
                for learner in learners:
                    np.random.seed(seed * 10 + _LEARNER_SEED_OFFSET.get(learner, 0))
                    t0 = time.time()
                    result = estimate_ate(
                        synth,
                        learner=learner,
                        n_folds=n_folds,
                    )
                    elapsed = time.time() - t0
                    result["elapsed"] = elapsed
                    results.append(result)
                    done += 1
                    if done % 50 == 0 or done == total:
                        print(f"[{done}/{total}] alpha={alpha}, {complexity}, {learner}, seed={seed} ({elapsed:.1f}s)")

    return pd.DataFrame(results)


def run_monte_carlo_irm(
    base_df: pd.DataFrame,
    overlap_strengths: list[float] = DEFAULT_OVERLAP_STRENGTHS,
    complexities: list[str] = DEFAULT_COMPLEXITIES,
    learners: list[str] | None = None,
    n_reps: int = DEFAULT_N_REPS,
    n_folds: int = 5,
    propensity_model: str = "logistic",
    start_seed: int = 1,
) -> pd.DataFrame:
    """Run the full Monte Carlo grid using IRM and return a results DataFrame."""
    if learners is None:
        learners = list(IRM_LEARNER_FACTORIES.keys())
    results = []
    total = len(overlap_strengths) * len(complexities) * len(learners) * n_reps
    done = 0

    for alpha in overlap_strengths:
        for complexity in complexities:
            for seed in range(start_seed, start_seed + n_reps):
                synth = generate_dataset(
                    base_df,
                    sample_size=None,
                    overlap_strength=alpha,
                    complexity=complexity,
                    seed=seed,
                    propensity_model=propensity_model,
                )
                for learner in learners:
                    np.random.seed(seed * 10 + _LEARNER_SEED_OFFSET.get(learner, 0))
                    t0 = time.time()
                    result = estimate_ate_irm(
                        synth,
                        learner=learner,
                        n_folds=n_folds,
                    )
                    elapsed = time.time() - t0
                    result["elapsed"] = elapsed
                    results.append(result)
                    done += 1
                    if done % 50 == 0 or done == total:
                        print(f"[{done}/{total}] alpha={alpha}, {complexity}, {learner}, seed={seed} ({elapsed:.1f}s)")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo DML simulations over the overlap grid."
    )
    parser.add_argument(
        "--input",
        default="processed/ihdp_sim_processed.csv",
        help="Path to the cleaned IHDP CSV.",
    )
    parser.add_argument(
        "--output",
        default="output/mc_results.csv",
        help="Path to write the results CSV.",
    )
    parser.add_argument("--n-reps", type=int, default=DEFAULT_N_REPS)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--start-seed",
        type=int,
        default=1,
        help="Starting seed for MC reps (default 1). Use to extend prior runs with non-overlapping seeds.",
    )
    parser.add_argument(
        "--overlap-strengths",
        type=float,
        nargs="+",
        default=DEFAULT_OVERLAP_STRENGTHS,
    )
    parser.add_argument(
        "--complexities",
        nargs="+",
        choices=["linear", "nonlinear"],
        default=DEFAULT_COMPLEXITIES,
    )
    parser.add_argument(
        "--learners",
        nargs="+",
        choices=list(LEARNER_FACTORIES.keys()),
        default=DEFAULT_LEARNERS,
    )
    parser.add_argument(
        "--propensity-model",
        choices=["logistic", "structural", "highdim", "threshold"],
        default="logistic",
    )
    parser.add_argument(
        "--framework",
        choices=["plr", "irm"],
        default="plr",
        help="DML framework: PLR (partially linear regression) or IRM (interactive regression model).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_df = pd.read_csv(args.input)

    # Validate learners against framework
    if args.framework == "irm":
        valid_learners = list(IRM_LEARNER_FACTORIES.keys())
        for l in args.learners:
            if l not in valid_learners:
                raise ValueError(
                    f"Learner '{l}' not supported for IRM (requires classifier propensity). "
                    f"Use one of {valid_learners}."
                )

    total = len(args.overlap_strengths) * len(args.complexities) * len(args.learners) * args.n_reps
    print(f"Running {total} estimations...")
    print(f"  Framework: {args.framework.upper()}")
    print(f"  Overlap strengths: {args.overlap_strengths}")
    print(f"  Complexities: {args.complexities}")
    print(f"  Learners: {args.learners}")
    print(f"  Reps per setting: {args.n_reps}")
    print(f"  Propensity model: {args.propensity_model}")
    print()

    if args.framework == "irm":
        results = run_monte_carlo_irm(
            base_df,
            overlap_strengths=args.overlap_strengths,
            complexities=args.complexities,
            learners=args.learners,
            n_reps=args.n_reps,
            n_folds=args.n_folds,
            propensity_model=args.propensity_model,
            start_seed=args.start_seed,
        )
    else:
        results = run_monte_carlo(
            base_df,
            overlap_strengths=args.overlap_strengths,
            complexities=args.complexities,
            learners=args.learners,
            n_reps=args.n_reps,
            n_folds=args.n_folds,
            propensity_model=args.propensity_model,
            start_seed=args.start_seed,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nWrote {len(results)} results to {output_path}")

    # Print summary
    summary = results.groupby(["overlap_strength", "complexity", "learner"]).agg(
        mean_est=("estimate", "mean"),
        mean_se=("se", "mean"),
        bias=("estimate", lambda x: x.mean() - 4.0),
        rmse=("estimate", lambda x: np.sqrt(((x - 4.0) ** 2).mean())),
        coverage=("covers", "mean"),
    ).round(4)
    print("\n" + summary.to_string())


if __name__ == "__main__":
    main()
