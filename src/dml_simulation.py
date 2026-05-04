"""DML estimation runner for the IHDP overlap simulation project."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegression

import doubleml as dml

from clean_ihdp import SIM_COVARIATE_COLUMNS


# ---------------------------------------------------------------------------
# Learner factories
# ---------------------------------------------------------------------------

def make_lasso_learners() -> tuple:
    """Return (ml_l, ml_m) for PLR using LassoCV."""
    ml_l = LassoCV(cv=5, max_iter=5000, random_state=0)
    ml_m = LassoCV(cv=5, max_iter=5000, random_state=0)
    return ml_l, ml_m


def make_xgboost_learners() -> tuple:
    """Return (ml_l, ml_m) for PLR using XGBoost."""
    from xgboost import XGBClassifier, XGBRegressor

    ml_l = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=0,
        verbosity=0,
    )
    ml_m = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=0,
        verbosity=0,
        eval_metric="logloss",
    )
    return ml_l, ml_m


def make_ridge_learners() -> tuple:
    """Return (ml_l, ml_m) for PLR using Ridge/LogisticRegression."""
    ml_l = RidgeCV(cv=5)
    ml_m = LogisticRegression(max_iter=10000, random_state=0)
    return ml_l, ml_m


def make_lasso_logistic_learners() -> tuple:
    """Return (ml_l, ml_m) for PLR using Lasso outcome + Logistic propensity.

    Same outcome model as 'lasso', but with a logistic propensity model.
    Allows isolating the effect of propensity specification while holding
    the outcome model constant.
    """
    ml_l = LassoCV(cv=5, max_iter=5000, random_state=0)
    ml_m = LogisticRegression(max_iter=10000, random_state=0)
    return ml_l, ml_m


def make_rf_learners() -> tuple:
    """Return (ml_l, ml_m) for PLR using Random Forest."""
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    ml_l = RandomForestRegressor(
        n_estimators=200, max_depth=5, min_samples_leaf=5,
        random_state=0, n_jobs=-1,
    )
    ml_m = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=5,
        random_state=0, n_jobs=-1,
    )
    return ml_l, ml_m


LEARNER_FACTORIES = {
    "lasso": make_lasso_learners,
    "xgboost": make_xgboost_learners,
    "ridge": make_ridge_learners,
    "rf": make_rf_learners,
    "lasso_logistic": make_lasso_logistic_learners,
}


NON_COVARIATE_COLUMNS = {
    "y",
    "sim_treat",
    "treat",
    "propensity",
    "true_propensity",
    "y0",
    "y1",
    "true_tau",
    "sample_size",
    "overlap_strength",
    "complexity",
    "seed",
    "source_row",
    "dataset",
}


def infer_covariate_columns(df: pd.DataFrame) -> list[str]:
    """Return covariates for IHDP data, with a numeric fallback for external bases."""
    ihdp_cols = [c for c in SIM_COVARIATE_COLUMNS if c in df.columns and c != "treat"]
    if ihdp_cols:
        return ihdp_cols

    return [
        c for c in df.columns
        if c not in NON_COVARIATE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]


# ---------------------------------------------------------------------------
# DML estimation
# ---------------------------------------------------------------------------

def estimate_ate(
    df: pd.DataFrame,
    learner: str = "lasso",
    n_folds: int = 5,
    true_ate: float = 4.0,
) -> dict:
    """Run PLR-style DML on a single synthetic dataset.

    Parameters
    ----------
    df : pd.DataFrame
        A generated synthetic dataset with covariates, ``sim_treat``, and ``y``.
    learner : str
        One of ``"lasso"`` or ``"xgboost"``.
    n_folds : int
        Number of cross-fitting folds.
    true_ate : float
        The true ATE for computing coverage.

    Returns
    -------
    dict with keys: estimate, se, ci_lower, ci_upper, covers, r_d,
    learner, overlap_strength, complexity, seed.
    """
    covariate_cols = infer_covariate_columns(df)

    dml_data = dml.DoubleMLData(
        df,
        y_col="y",
        d_cols="sim_treat",
        x_cols=covariate_cols,
    )

    ml_l, ml_m = LEARNER_FACTORIES[learner]()

    model = dml.DoubleMLPLR(
        dml_data,
        ml_l=ml_l,
        ml_m=ml_m,
        n_folds=n_folds,
        score="partialling out",
    )
    model.fit()

    estimate = model.coef[0]
    se = model.se[0]
    ci = model.confint(level=0.95)
    ci_lower = ci.iloc[0, 0]
    ci_upper = ci.iloc[0, 1]
    covers = int(ci_lower <= true_ate <= ci_upper)

    # Overlap diagnostic: R_D = Var(D_tilde) / Var(D)
    d = df["sim_treat"].to_numpy()
    m_hat = model.predictions["ml_m"].flatten()
    d_tilde = d - m_hat
    var_d = d.var()
    r_d = d_tilde.var() / var_d if var_d > 0 else np.nan

    # Outcome-side diagnostic: R_Y = cross-fitted outcome R²
    y = df["y"].to_numpy()
    l_hat = model.predictions["ml_l"].flatten()
    ss_res_y = np.sum((y - l_hat) ** 2)
    ss_tot_y = np.sum((y - y.mean()) ** 2)
    r_y = 1.0 - ss_res_y / ss_tot_y if ss_tot_y > 0 else np.nan

    # Within-group outcome residual variance (pooled)
    # R_Y is contaminated: E[Y|X] = g0(X) + tau*m(X), so R_Y mechanically
    # rises when treatment is more predictable from X.
    # Within-group residual variance avoids this by conditioning on D.
    # Var(Y - g_hat | D=d) captures outcome model fit within each group.
    treated = d == 1
    control = d == 0
    outcome_resid_var_within = np.nan
    if treated.sum() > 1 and control.sum() > 1:
        resid = y - l_hat
        var_0 = resid[control].var()
        var_1 = resid[treated].var()
        n0, n1 = control.sum(), treated.sum()
        outcome_resid_var_within = (n0 * var_0 + n1 * var_1) / (n0 + n1)

    outcome_rmse = np.sqrt(np.mean((y - l_hat) ** 2))
    outcome_resid_std = (y - l_hat).std()
    propensity_rmse = np.sqrt(np.mean((d - m_hat) ** 2))

    return {
        "estimate": estimate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "covers": covers,
        "r_d": r_d,
        "r_y": r_y,
        "outcome_resid_var_within": outcome_resid_var_within,
        "outcome_rmse": outcome_rmse,
        "outcome_resid_std": outcome_resid_std,
        "propensity_rmse": propensity_rmse,
        "learner": learner,
        "overlap_strength": df["overlap_strength"].iloc[0] if "overlap_strength" in df.columns else None,
        "complexity": df["complexity"].iloc[0] if "complexity" in df.columns else None,
        "seed": df["seed"].iloc[0] if "seed" in df.columns else None,
    }


# ---------------------------------------------------------------------------
# IRM estimation
# ---------------------------------------------------------------------------

# Learners compatible with IRM (ml_m must be a classifier).
IRM_LEARNER_FACTORIES = {
    "lasso_logistic": make_lasso_logistic_learners,
    "xgboost": make_xgboost_learners,
    "ridge": make_ridge_learners,
    "rf": make_rf_learners,
}


def estimate_ate_irm(
    df: pd.DataFrame,
    learner: str = "xgboost",
    n_folds: int = 5,
    true_ate: float = 4.0,
) -> dict:
    """Run IRM-style DML on a single synthetic dataset.

    Unlike PLR, IRM estimates E[Y|X,D=0] and E[Y|X,D=1] separately,
    enabling arm-specific outcome diagnostics (R_Y0, R_Y1) that are
    not contaminated by the treatment channel.

    Parameters
    ----------
    df : pd.DataFrame
        A generated synthetic dataset with covariates, ``sim_treat``, and ``y``.
    learner : str
        One of the keys in ``IRM_LEARNER_FACTORIES``.
    n_folds : int
        Number of cross-fitting folds.
    true_ate : float
        The true ATE for computing coverage.

    Returns
    -------
    dict with keys: estimate, se, ci_lower, ci_upper, covers, r_d,
    r_y0, r_y1, r_y_irm_pooled, learner, overlap_strength, complexity,
    seed, framework.
    """
    if learner not in IRM_LEARNER_FACTORIES:
        raise ValueError(
            f"Learner '{learner}' not supported for IRM. "
            f"Use one of {list(IRM_LEARNER_FACTORIES.keys())}."
        )

    covariate_cols = infer_covariate_columns(df)

    dml_data = dml.DoubleMLData(
        df,
        y_col="y",
        d_cols="sim_treat",
        x_cols=covariate_cols,
    )

    ml_g, ml_m = IRM_LEARNER_FACTORIES[learner]()

    model = dml.DoubleMLIRM(
        dml_data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=n_folds,
        score="ATE",
    )
    model.fit()

    estimate = model.coef[0]
    se = model.se[0]
    ci = model.confint(level=0.95)
    ci_lower = ci.iloc[0, 0]
    ci_upper = ci.iloc[0, 1]
    covers = int(ci_lower <= true_ate <= ci_upper)

    # Overlap diagnostic: R_D (same as PLR)
    d = df["sim_treat"].to_numpy()
    m_hat = model.predictions["ml_m"].flatten()
    d_tilde = d - m_hat
    var_d = d.var()
    r_d = d_tilde.var() / var_d if var_d > 0 else np.nan

    # Arm-specific outcome diagnostics
    g0_hat = model.predictions["ml_g0"].flatten()
    g1_hat = model.predictions["ml_g1"].flatten()
    y = df["y"].to_numpy()
    treated = d == 1
    control = d == 0

    # R_Y0: cross-fitted R² among controls
    r_y0 = np.nan
    if control.sum() > 1:
        ss_res_0 = np.sum((y[control] - g0_hat[control]) ** 2)
        ss_tot_0 = np.sum((y[control] - y[control].mean()) ** 2)
        r_y0 = 1.0 - ss_res_0 / ss_tot_0 if ss_tot_0 > 0 else np.nan

    # R_Y1: cross-fitted R² among treated
    r_y1 = np.nan
    if treated.sum() > 1:
        ss_res_1 = np.sum((y[treated] - g1_hat[treated]) ** 2)
        ss_tot_1 = np.sum((y[treated] - y[treated].mean()) ** 2)
        r_y1 = 1.0 - ss_res_1 / ss_tot_1 if ss_tot_1 > 0 else np.nan

    # Pooled R_Y (sum of squares across arms, not weighted average of R²)
    r_y_irm_pooled = np.nan
    if control.sum() > 1 and treated.sum() > 1 and ss_tot_0 + ss_tot_1 > 0:
        r_y_irm_pooled = 1.0 - (ss_res_0 + ss_res_1) / (ss_tot_0 + ss_tot_1)

    # PLR-style R_Y for comparison (E[Y|X] fit quality)
    # IRM doesn't estimate E[Y|X] directly, so approximate:
    # g_hat(X) = D * g1_hat + (1-D) * g0_hat gives the fitted value
    g_hat_combined = np.where(treated, g1_hat, g0_hat)
    ss_res_y = np.sum((y - g_hat_combined) ** 2)
    ss_tot_y = np.sum((y - y.mean()) ** 2)
    r_y_combined = 1.0 - ss_res_y / ss_tot_y if ss_tot_y > 0 else np.nan

    # Clip m_hat away from 0/1 for inverse-probability weighting
    m_hat_clipped = np.clip(m_hat, 0.01, 0.99)

    # IPW-weighted R² — weights match the IRM score's per-unit weighting.
    # R_Y0_weighted: among controls, weight by 1/(1 - m_hat) (the control IPW weight)
    r_y0_weighted = np.nan
    if control.sum() > 1:
        w0 = 1.0 / (1.0 - m_hat_clipped[control])
        y0 = y[control]
        g0c = g0_hat[control]
        w0_mean_y = np.sum(w0 * y0) / np.sum(w0)
        ss_res_0w = np.sum(w0 * (y0 - g0c) ** 2)
        ss_tot_0w = np.sum(w0 * (y0 - w0_mean_y) ** 2)
        r_y0_weighted = 1.0 - ss_res_0w / ss_tot_0w if ss_tot_0w > 0 else np.nan

    # R_Y1_weighted: among treated, weight by 1/m_hat (the treated IPW weight)
    r_y1_weighted = np.nan
    if treated.sum() > 1:
        w1 = 1.0 / m_hat_clipped[treated]
        y1 = y[treated]
        g1t = g1_hat[treated]
        w1_mean_y = np.sum(w1 * y1) / np.sum(w1)
        ss_res_1w = np.sum(w1 * (y1 - g1t) ** 2)
        ss_tot_1w = np.sum(w1 * (y1 - w1_mean_y) ** 2)
        r_y1_weighted = 1.0 - ss_res_1w / ss_tot_1w if ss_tot_1w > 0 else np.nan

    # Local R² — only on units in thin-overlap regions.
    # R_Y0_local: R² among controls with m_hat > 0.7 (the "rare control" units)
    r_y0_local = np.nan
    local_control_mask = control & (m_hat_clipped > 0.7)
    n_local_c = local_control_mask.sum()
    if n_local_c > 5:
        yl = y[local_control_mask]
        gl = g0_hat[local_control_mask]
        ss_res_0l = np.sum((yl - gl) ** 2)
        ss_tot_0l = np.sum((yl - yl.mean()) ** 2)
        r_y0_local = 1.0 - ss_res_0l / ss_tot_0l if ss_tot_0l > 0 else np.nan

    # R_Y1_local: R² among treated with m_hat < 0.3 (the "rare treated" units)
    r_y1_local = np.nan
    local_treated_mask = treated & (m_hat_clipped < 0.3)
    n_local_t = local_treated_mask.sum()
    if n_local_t > 5:
        yl = y[local_treated_mask]
        gl = g1_hat[local_treated_mask]
        ss_res_1l = np.sum((yl - gl) ** 2)
        ss_tot_1l = np.sum((yl - yl.mean()) ** 2)
        r_y1_local = 1.0 - ss_res_1l / ss_tot_1l if ss_tot_1l > 0 else np.nan

    # Alternative local variants -----------------------------------------
    def _local_r2(mask, g_hat):
        if mask.sum() <= 5:
            return np.nan
        yl = y[mask]
        gl = g_hat[mask]
        ss_res = np.sum((yl - gl) ** 2)
        ss_tot = np.sum((yl - yl.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Threshold sweep for both arms
    local_control_mask_05 = control & (m_hat_clipped > 0.5)
    local_treated_mask_05 = treated & (m_hat_clipped < 0.5)
    r_y0_local_05 = _local_r2(local_control_mask_05, g0_hat)
    r_y1_local_05 = _local_r2(local_treated_mask_05, g1_hat)
    r_y0_local_08 = _local_r2(control & (m_hat_clipped > 0.8), g0_hat)

    # Adaptive "2x odds-ratio" threshold: units whose odds of treatment are
    # at least 2x the baseline odds. Scales below 0.5 for low-baseline datasets
    # (e.g. LaLonde at 30% → control thresh=0.46, treated thresh=0.18).
    p_bar = d.mean()
    thresh_2x_c = (2 * p_bar) / (1 + p_bar)      # control: odds(m) >= 2 * odds(p_bar)
    thresh_2x_t = p_bar / (2 - p_bar)             # treated: odds(1-m) >= 2 * odds(1-p_bar)
    local_control_mask_2x = control & (m_hat_clipped > thresh_2x_c)
    local_treated_mask_2x = treated & (m_hat_clipped < thresh_2x_t)
    r_y0_local_2x = _local_r2(local_control_mask_2x, g0_hat)
    r_y1_local_2x = _local_r2(local_treated_mask_2x, g1_hat)

    # Top-k: R² among the 30 controls with highest m_hat (most extreme)
    r_y0_topk = np.nan
    if control.sum() > 30:
        ctrl_idx = np.where(control)[0]
        top_idx = ctrl_idx[np.argsort(-m_hat_clipped[ctrl_idx])[:30]]
        top_mask = np.zeros_like(control)
        top_mask[top_idx] = True
        r_y0_topk = _local_r2(top_mask, g0_hat)

    # Mean squared residual in the local region (not R², just MSE — lower=better fit)
    # Some times MSE is a cleaner signal than R² when local y-variance is small
    r_y0_local_mse = np.nan
    if n_local_c > 5:
        yl = y[local_control_mask]
        gl = g0_hat[local_control_mask]
        r_y0_local_mse = np.mean((yl - gl) ** 2)

    # Gap between global and local R² at 0.5 threshold — apples-to-apples signal
    r_y0_gap = (r_y0 - r_y0_local_05) if not (np.isnan(r_y0) or np.isnan(r_y0_local_05)) else np.nan
    r_y1_gap = (r_y1 - r_y1_local_05) if not (np.isnan(r_y1) or np.isnan(r_y1_local_05)) else np.nan

    # IF-weighted squared residual — weight each control residual by (1/(1-m))^2
    # This matches the IRM influence function's variance contribution more exactly
    r_y0_ifw_mse = np.nan
    if control.sum() > 1:
        w_ifw = 1.0 / (1.0 - m_hat_clipped[control]) ** 2
        resid_sq = (y[control] - g0_hat[control]) ** 2
        r_y0_ifw_mse = np.sum(w_ifw * resid_sq) / np.sum(w_ifw)

    propensity_rmse = np.sqrt(np.mean((d - m_hat) ** 2))

    return {
        "estimate": estimate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "covers": covers,
        "r_d": r_d,
        "r_y0": r_y0,
        "r_y1": r_y1,
        "r_y_irm_pooled": r_y_irm_pooled,
        "r_y_combined": r_y_combined,
        "r_y0_weighted": r_y0_weighted,
        "r_y1_weighted": r_y1_weighted,
        "r_y0_local": r_y0_local,
        "r_y1_local": r_y1_local,
        "r_y0_local_05": r_y0_local_05,
        "r_y1_local_05": r_y1_local_05,
        "r_y0_local_08": r_y0_local_08,
        "r_y0_topk": r_y0_topk,
        "r_y0_local_mse": r_y0_local_mse,
        "r_y0_gap": r_y0_gap,
        "r_y1_gap": r_y1_gap,
        "r_y0_ifw_mse": r_y0_ifw_mse,
        "n_local_control": int(n_local_c),
        "n_local_treated": int(n_local_t),
        "n_local_control_05": int(local_control_mask_05.sum()),
        "n_local_treated_05": int(local_treated_mask_05.sum()),
        "r_y0_local_2x": r_y0_local_2x,
        "r_y1_local_2x": r_y1_local_2x,
        "n_local_control_2x": int(local_control_mask_2x.sum()),
        "n_local_treated_2x": int(local_treated_mask_2x.sum()),
        "thresh_2x_c": thresh_2x_c,
        "thresh_2x_t": thresh_2x_t,
        "propensity_rmse": propensity_rmse,
        "learner": learner,
        "framework": "IRM",
        "overlap_strength": df["overlap_strength"].iloc[0] if "overlap_strength" in df.columns else None,
        "complexity": df["complexity"].iloc[0] if "complexity" in df.columns else None,
        "seed": df["seed"].iloc[0] if "seed" in df.columns else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DML estimation on a synthetic IHDP dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a generated synthetic CSV.",
    )
    parser.add_argument(
        "--learner",
        choices=list(LEARNER_FACTORIES.keys()),
        default="lasso",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    result = estimate_ate(df, learner=args.learner, n_folds=args.n_folds)

    print(f"Learner:    {result['learner']}")
    print(f"Estimate:   {result['estimate']:.4f}")
    print(f"SE:         {result['se']:.4f}")
    print(f"95% CI:     [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"Covers 4.0: {bool(result['covers'])}")
    print(f"R_D:        {result['r_d']:.4f}")


if __name__ == "__main__":
    main()
