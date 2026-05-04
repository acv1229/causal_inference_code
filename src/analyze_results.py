"""Analyze and plot Monte Carlo results for the IHDP DML overlap project."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["bias"] = df["estimate"] - 4.0
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby(["overlap_strength", "complexity", "learner"]).agg(
        mean_est=("estimate", "mean"),
        bias=("bias", "mean"),
        rmse=("estimate", lambda x: np.sqrt(((x - 4.0) ** 2).mean())),
        mean_se=("se", "mean"),
        coverage=("covers", "mean"),
        mean_r_d=("r_d", "mean"),
        std_r_d=("r_d", "std"),
        n_reps=("estimate", "count"),
    ).reset_index()
    return summary


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
LEARNER_STYLES = {
    "lasso": {"color": "#d62728", "marker": "o", "label": "Lasso"},
    "xgboost": {"color": "#1f77b4", "marker": "s", "label": "XGBoost"},
}
SURFACE_TITLES = {"linear": "Surface A (linear)", "nonlinear": "Surface B (nonlinear)"}


def _setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["savefig.bbox"] = "tight"


def plot_metric_vs_overlap(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    hline: float | None = None,
    hline_label: str | None = None,
):
    """Line plot of a metric vs overlap_strength, paneled by surface."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, complexity in zip(axes, ["linear", "nonlinear"]):
        sub = summary[summary["complexity"] == complexity]
        for learner, style in LEARNER_STYLES.items():
            data = sub[sub["learner"] == learner]
            ax.plot(
                data["overlap_strength"], data[metric],
                color=style["color"], marker=style["marker"],
                label=style["label"], linewidth=2, markersize=7,
            )
        if hline is not None:
            ax.axhline(hline, color="gray", linestyle="--", linewidth=1, label=hline_label)
        ax.set_xlabel("Overlap strength")
        ax.set_title(SURFACE_TITLES[complexity])
        ax.legend()

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_coverage_vs_overlap(summary: pd.DataFrame, output_path: Path):
    plot_metric_vs_overlap(
        summary, metric="coverage", ylabel="Coverage",
        title="95% CI Coverage vs Overlap Strength",
        output_path=output_path,
        hline=0.95, hline_label="Nominal 95%",
    )


def plot_bias_vs_overlap(summary: pd.DataFrame, output_path: Path):
    plot_metric_vs_overlap(
        summary, metric="bias", ylabel="Bias (estimate − 4.0)",
        title="Bias vs Overlap Strength",
        output_path=output_path,
        hline=0.0, hline_label="No bias",
    )


def plot_rmse_vs_overlap(summary: pd.DataFrame, output_path: Path):
    plot_metric_vs_overlap(
        summary, metric="rmse", ylabel="RMSE",
        title="RMSE vs Overlap Strength",
        output_path=output_path,
    )


def plot_r_d_vs_overlap(summary: pd.DataFrame, output_path: Path):
    plot_metric_vs_overlap(
        summary, metric="mean_r_d", ylabel="$R_D$",
        title="$R_D$ vs Overlap Strength",
        output_path=output_path,
    )


def plot_r_d_vs_coverage(summary: pd.DataFrame, output_path: Path):
    """The central plot: R_D on x-axis, coverage on y-axis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, complexity in zip(axes, ["linear", "nonlinear"]):
        sub = summary[summary["complexity"] == complexity]
        for learner, style in LEARNER_STYLES.items():
            data = sub[sub["learner"] == learner]
            ax.plot(
                data["mean_r_d"], data["coverage"],
                color=style["color"], marker=style["marker"],
                label=style["label"], linewidth=2, markersize=7,
            )
            # Annotate extreme points with overlap strength
            for _, row in data.iterrows():
                if row["overlap_strength"] in [0.0, 5.0]:
                    ax.annotate(
                        f'α={row["overlap_strength"]:.0f}',
                        (row["mean_r_d"], row["coverage"]),
                        textcoords="offset points", xytext=(8, -4),
                        fontsize=8, color=style["color"],
                    )
        ax.axhline(0.95, color="gray", linestyle="--", linewidth=1, label="Nominal 95%")
        ax.set_xlabel("$R_D$")
        ax.set_title(SURFACE_TITLES[complexity])
        ax.legend()

    axes[0].set_ylabel("Coverage")
    fig.suptitle(
        "$R_D$ vs Coverage: Same Diagnostic, Different Outcomes",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_r_d_vs_bias(summary: pd.DataFrame, output_path: Path):
    """R_D on x-axis, absolute bias on y-axis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, complexity in zip(axes, ["linear", "nonlinear"]):
        sub = summary[summary["complexity"] == complexity]
        for learner, style in LEARNER_STYLES.items():
            data = sub[sub["learner"] == learner]
            ax.plot(
                data["mean_r_d"], data["bias"].abs(),
                color=style["color"], marker=style["marker"],
                label=style["label"], linewidth=2, markersize=7,
            )
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("$R_D$")
        ax.set_title(SURFACE_TITLES[complexity])
        ax.legend()

    axes[0].set_ylabel("|Bias|")
    fig.suptitle("$R_D$ vs |Bias|", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_se_vs_overlap(summary: pd.DataFrame, output_path: Path):
    """Mean reported SE vs overlap strength — shows SE inflation."""
    plot_metric_vs_overlap(
        summary, metric="mean_se", ylabel="Mean reported SE",
        title="SE Inflation vs Overlap Strength",
        output_path=output_path,
    )


def save_summary_table(summary: pd.DataFrame, output_path: Path):
    """Save the summary table as CSV."""
    cols = [
        "overlap_strength", "complexity", "learner",
        "bias", "rmse", "mean_se", "coverage", "mean_r_d",
    ]
    table = summary[cols].copy()
    table = table.round(4)
    table.to_csv(output_path, index=False)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Monte Carlo results and produce figures/tables."
    )
    parser.add_argument(
        "--input",
        default="output/mc_results_v3_structural.csv",
        help="Path to the MC results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for figures/ and tables/ subdirectories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {args.input}...")
    df = load_results(args.input)
    summary = build_summary(df)

    _setup_style()

    print("Generating figures...")
    plot_coverage_vs_overlap(summary, fig_dir / "coverage_vs_overlap.png")
    plot_bias_vs_overlap(summary, fig_dir / "bias_vs_overlap.png")
    plot_rmse_vs_overlap(summary, fig_dir / "rmse_vs_overlap.png")
    plot_r_d_vs_overlap(summary, fig_dir / "r_d_vs_overlap.png")
    plot_r_d_vs_coverage(summary, fig_dir / "r_d_vs_coverage.png")
    plot_r_d_vs_bias(summary, fig_dir / "r_d_vs_bias.png")
    plot_se_vs_overlap(summary, fig_dir / "se_vs_overlap.png")

    print("Generating tables...")
    save_summary_table(summary, table_dir / "mc_summary.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
