"""Generate a three-panel drift monitoring chart."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ["generate_report"]

PREDICTION_LOG_PATH = Path("data/predictions.db")
FEATURES_PATH = Path("data/processed/features.parquet")
DEFAULT_OUTPUT = Path("reports/drift_report.png")

_RESIDUAL_THRESHOLD_MULTIPLIER = 1.5


def _load_prediction_log(db_path: Path) -> pd.DataFrame:
    import sqlite_utils
    db = sqlite_utils.Database(str(db_path))
    if "predictions" not in db.table_names():
        return pd.DataFrame()
    rows = list(db["predictions"].rows)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["predicted_values"] = df["predicted_values"].apply(
        lambda v: ast.literal_eval(v) if isinstance(v, str) else v
    )
    df["actual_values"] = df["actual_values"].apply(
        lambda v: ast.literal_eval(v) if isinstance(v, str) else None
    )
    return df


def generate_report(
    baseline_mae: float,
    output_path: Path = DEFAULT_OUTPUT,
    db_path: Path = PREDICTION_LOG_PATH,
    features_path: Path = FEATURES_PATH,
) -> Path:
    """Render and save a three-panel drift monitoring chart.

    Panel 1: Predicted vs actual approvals (national aggregate)
    Panel 2: Rolling MAE over time with alert threshold
    Panel 3: Cash rate vs training distribution bands
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(features_path) if features_path.exists() else pd.DataFrame()
    log_df = _load_prediction_log(db_path)

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle("Housing Approvals Forecaster: Drift Monitoring Report", fontsize=14, fontweight="bold")

    # Panel 1: Predicted vs actual (requires actuals in log)
    ax1 = axes[0]
    if not log_df.empty and "actual_values" in log_df.columns:
        with_actuals = log_df.dropna(subset=["actual_values"]).sort_values("timestamp")
        if not with_actuals.empty:
            agg_pred = with_actuals["predicted_values"].apply(lambda v: np.mean(v) if v else np.nan)
            agg_actual = with_actuals["actual_values"].apply(lambda v: np.mean(v) if v else np.nan)
            ax1.plot(with_actuals["timestamp"], agg_actual, label="Actual", marker="o", color="steelblue")
            ax1.plot(with_actuals["timestamp"], agg_pred, label="Predicted", linestyle="--", marker="x", color="darkorange")
            ax1.set_title("Predicted vs Actual Approvals (mean across logged LGAs)")
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, "No actuals logged yet", ha="center", va="center", transform=ax1.transAxes)
            ax1.set_title("Predicted vs Actual Approvals")
    else:
        ax1.text(0.5, 0.5, "No prediction log data", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Predicted vs Actual Approvals")
    ax1.set_ylabel("Dwellings approved")

    # Panel 2: Rolling MAE over time
    ax2 = axes[1]
    if not log_df.empty and "actual_values" in log_df.columns:
        with_actuals = log_df.dropna(subset=["actual_values"]).sort_values("timestamp").copy()
        if not with_actuals.empty:
            with_actuals["mae"] = with_actuals.apply(
                lambda r: float(np.mean(np.abs(np.array(r["predicted_values"][:len(r["actual_values"])]) -
                                              np.array(r["actual_values"][:len(r["predicted_values"])]))))
                if r["predicted_values"] and r["actual_values"] else np.nan,
                axis=1,
            )
            rolling_mae = with_actuals.set_index("timestamp")["mae"].rolling("30D").mean()
            ax2.plot(rolling_mae.index, rolling_mae.values, color="steelblue", label="Rolling 30D MAE")
            threshold = baseline_mae * _RESIDUAL_THRESHOLD_MULTIPLIER
            ax2.axhline(threshold, color="red", linestyle="--", label=f"Alert threshold ({threshold:.0f})")
            ax2.axhline(baseline_mae, color="green", linestyle=":", label=f"Training MAE ({baseline_mae:.0f})")
            ax2.set_title("Residual Drift: Rolling MAE Over Time")
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "No actuals available for MAE computation", ha="center", va="center", transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, "No prediction log data", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_ylabel("Mean Absolute Error")

    # Panel 3: Cash rate vs training distribution
    ax3 = axes[2]
    if not df.empty and "cash_rate" in df.columns and "quarter" in df.columns:
        df_sorted = df.drop_duplicates("quarter").sort_values("quarter").copy()
        df_sorted["quarter_dt"] = pd.to_datetime(df_sorted["quarter"])
        train_rates = df_sorted[df_sorted["post_rate_hike"] == 0]["cash_rate"]
        mean = train_rates.mean()
        std = train_rates.std()
        ax3.plot(df_sorted["quarter_dt"], df_sorted["cash_rate"], color="steelblue", label="Cash rate")
        ax3.axhline(mean + _RESIDUAL_THRESHOLD_MULTIPLIER * std, color="red", linestyle="--", label="Drift upper bound")
        ax3.axhline(mean - _RESIDUAL_THRESHOLD_MULTIPLIER * std, color="red", linestyle="--", label="Drift lower bound")
        ax3.fill_between(df_sorted["quarter_dt"], mean - std, mean + std, alpha=0.1, color="green", label="Training mean +/-1 std")
        rate_hike_start = df_sorted[df_sorted["post_rate_hike"] == 1]["quarter_dt"].min()
        if pd.notna(rate_hike_start):
            ax3.axvline(rate_hike_start, color="darkorange", linestyle=":", linewidth=2, label="Rate-hike structural break (Q3 2022)")
        ax3.set_title("Feature Drift: Cash Rate vs Training Distribution")
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "Feature data not available", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Feature Drift: Cash Rate vs Training Distribution")
    ax3.set_ylabel("RBA Cash Rate (%)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Drift report saved to {output_path}")
    return output_path


if __name__ == "__main__":
    generate_report(baseline_mae=50.0)
