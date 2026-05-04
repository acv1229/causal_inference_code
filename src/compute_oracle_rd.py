"""Compute oracle R_D* and compare to estimated R_D across learners.

Uses the known true propensity m_0(X) from the data-generating process
to compute R_D* = Var(D - m_0(X)) / Var(D), then compares to the
estimated R_D from each learner's fitted propensity model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from generate_ihdp_synthetic import generate_dataset
from dml_simulation import estimate_ate

def compute_oracle_rd(df: pd.DataFrame) -> float:
    """Compute R_D* using the true propensity."""
    d = df["sim_treat"].to_numpy()
    m0 = df["propensity"].to_numpy()
    v = d - m0  # true treatment residual
    var_d = d.var()
    return v.var() / var_d if var_d > 0 else np.nan


def run_comparison(
    n_reps: int = 200,
    overlap_strength: float = 5.0,
    complexity: str = "nonlinear",
    propensity_model: str = "structural",
):
    base_df = pd.read_csv("processed/ihdp_sim_processed.csv")

    learners = ["xgboost", "lasso", "lasso_logistic"]
    results = []

    for rep in range(n_reps):
        seed = 1000 + rep
        df = generate_dataset(
            base_df,
            sample_size=None,
            overlap_strength=overlap_strength,
            complexity=complexity,
            seed=seed,
            propensity_model=propensity_model,
        )
        oracle_rd = compute_oracle_rd(df)

        for learner in learners:
            try:
                res = estimate_ate(df, learner=learner, n_folds=5)
                results.append({
                    "rep": rep,
                    "seed": seed,
                    "learner": learner,
                    "r_d": res["r_d"],
                    "r_d_oracle": oracle_rd,
                    "inflation": res["r_d"] - oracle_rd,
                    "coverage": res["covers"],
                    "estimate": res["estimate"],
                    "se": res["se"],
                })
            except Exception as e:
                print(f"  Failed: rep={rep}, learner={learner}: {e}")

        if (rep + 1) % 10 == 0:
            print(f"Completed {rep + 1}/{n_reps} reps")

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print(f"Oracle R_D* comparison: overlap={overlap_strength}, "
          f"surface={complexity}, propensity={propensity_model}")
    print(f"Reps: {n_reps}")
    print("=" * 70)

    summary = results_df.groupby("learner").agg(
        r_d_mean=("r_d", "mean"),
        r_d_oracle_mean=("r_d_oracle", "mean"),
        inflation_mean=("inflation", "mean"),
        inflation_std=("inflation", "std"),
        coverage=("coverage", "mean"),
    ).round(4)

    print("\n", summary.to_string())

    out_path = Path("output") / "oracle_rd_comparison.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    return results_df


if __name__ == "__main__":
    run_comparison(n_reps=200, overlap_strength=5.0, complexity="nonlinear")
