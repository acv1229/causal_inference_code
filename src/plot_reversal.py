"""
The Reversal Plot: R_D is anti-conservative under propensity misspecification.

Shows |Bias| and R_D both increasing with overlap strength for the misspecified
learner (Lasso), while R_D correctly decreases for the flexible learner (XGBoost).

This is the centerpiece figure for the anti-conservative diagnostic finding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

TRUE_ATE = 4.0

LEARNER_COLORS = {
    "lasso": "#d62728",
    "xgboost": "#1f77b4",
}
LEARNER_LABELS = {
    "lasso": "Lasso (misspecified)",
    "xgboost": "XGBoost (flexible)",
}


def load_and_summarize(path: str, dgp_name: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["abs_bias"] = (df["estimate"] - TRUE_ATE).abs()
    df["bias"] = df["estimate"] - TRUE_ATE
    summary = df.groupby(["overlap_strength", "complexity", "learner"]).agg(
        mean_bias=("bias", "mean"),
        mean_abs_bias=("abs_bias", "mean"),
        mean_r_d=("r_d", "mean"),
        coverage=("covers", "mean"),
        rmse=("estimate", lambda x: np.sqrt(((x - TRUE_ATE) ** 2).mean())),
    ).reset_index()
    if dgp_name:
        summary["dgp"] = dgp_name
    return summary


def plot_reversal_single(summary: pd.DataFrame, output_path: Path, title: str = ""):
    """Two-panel reversal plot (linear / nonlinear surface)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    for ax, surface in zip(axes, ["linear", "nonlinear"]):
        sub = summary[summary["complexity"] == surface]
        ax2 = ax.twinx()

        for learner in ["lasso", "xgboost"]:
            s = sub[sub["learner"] == learner].sort_values("overlap_strength")
            color = LEARNER_COLORS[learner]
            label = LEARNER_LABELS[learner]

            # |Bias| on left axis (solid)
            ax.plot(
                s["overlap_strength"], s["mean_abs_bias"],
                color=color, marker="o", linewidth=2, markersize=6,
                label=f"|Bias| — {label}",
            )

            # R_D on right axis (dashed)
            ax2.plot(
                s["overlap_strength"], s["mean_r_d"],
                color=color, marker="s", linewidth=2, markersize=5,
                linestyle="--", alpha=0.8,
                label=f"R_D — {label}",
            )

        ax.set_xlabel("Overlap strength", fontsize=12)
        ax.set_ylabel("|Bias|", fontsize=12)
        ax2.set_ylabel("R_D", fontsize=12, rotation=270, labelpad=15)
        ax.set_title(f"{surface.capitalize()} outcome surface", fontsize=13)
        ax.set_ylim(bottom=0)
        ax2.set_ylim(0, 1.15)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2, labels1 + labels2,
            loc="upper left", fontsize=8, framealpha=0.9,
        )

        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_reversal_four_panel(summaries: dict[str, pd.DataFrame], output_path: Path):
    """Four-DGP reversal plot (2x2 grid), nonlinear surface only."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for ax, (dgp_name, summary) in zip(axes.flat, summaries.items()):
        sub = summary[summary["complexity"] == "nonlinear"]
        ax2 = ax.twinx()

        for learner in ["lasso", "xgboost"]:
            s = sub[sub["learner"] == learner].sort_values("overlap_strength")
            color = LEARNER_COLORS[learner]
            label = LEARNER_LABELS[learner]

            ax.plot(
                s["overlap_strength"], s["mean_abs_bias"],
                color=color, marker="o", linewidth=2, markersize=5,
                label=f"|Bias| — {label}",
            )
            ax2.plot(
                s["overlap_strength"], s["mean_r_d"],
                color=color, marker="s", linewidth=2, markersize=4,
                linestyle="--", alpha=0.8,
                label=f"R_D — {label}",
            )

        ax.set_xlabel("Overlap strength", fontsize=10)
        ax.set_ylabel("|Bias|", fontsize=10)
        ax2.set_ylabel("R_D", fontsize=10, rotation=270, labelpad=12)
        ax.set_title(f"{dgp_name.capitalize()} propensity", fontsize=12, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax2.set_ylim(0, 1.15)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2, labels1 + labels2,
            loc="upper left", fontsize=7, framealpha=0.9,
        )
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Anti-Conservative Diagnostic: |Bias| and R_D vs Overlap Strength\n(Nonlinear outcome surface)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    # Single DGP reversal plot (structural — the clearest case)
    structural = load_and_summarize("output/archive/mc_results_v3_structural.csv", "structural")
    plot_reversal_single(
        structural,
        Path("output/figures/reversal_plot_structural.png"),
        title="Anti-Conservative Diagnostic: Structural Propensity (Race × Birthweight)",
    )

    # Cross-DGP four-panel plot
    datasets = {
        "structural": "output/archive/mc_results_v3_structural.csv",
        "highdim": "output/mc_results_v4_highdim.csv",
        "logistic": "output/archive/mc_results_v2_clip001_overlap0-10.csv",
        "threshold": "output/mc_results_v5_threshold.csv",
    }
    summaries = {}
    for dgp_name, path in datasets.items():
        if Path(path).exists():
            summaries[dgp_name] = load_and_summarize(path, dgp_name)
        else:
            print(f"  SKIPPING {dgp_name}: {path} not found")

    if len(summaries) > 1:
        plot_reversal_four_panel(
            summaries,
            Path("output/figures/reversal_plot_cross_dgp.png"),
        )

    # Print summary table for quick reference
    print("\nAnti-conservative signature by DGP (max overlap strength, nonlinear surface):")
    print(f"  {'DGP':<15} | {'Lasso |Bias|':>12} | {'Lasso R_D':>10} | {'XGB |Bias|':>11} | {'XGB R_D':>8} | {'R_D Gap':>8}")
    print(f"  {'-'*15} | {'-'*12} | {'-'*10} | {'-'*11} | {'-'*8} | {'-'*8}")
    for dgp_name, s in summaries.items():
        nl = s[s["complexity"] == "nonlinear"]
        max_alpha = nl["overlap_strength"].max()
        row = nl[nl["overlap_strength"] == max_alpha]
        lasso = row[row["learner"] == "lasso"].iloc[0]
        xgb = row[row["learner"] == "xgboost"].iloc[0]
        gap = lasso["mean_r_d"] - xgb["mean_r_d"]
        print(f"  {dgp_name:<15} | {lasso['mean_abs_bias']:>12.3f} | {lasso['mean_r_d']:>10.3f} | "
              f"{xgb['mean_abs_bias']:>11.3f} | {xgb['mean_r_d']:>8.3f} | {gap:>+8.3f}")


if __name__ == "__main__":
    main()
