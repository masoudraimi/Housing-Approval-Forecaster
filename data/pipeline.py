"""Feature engineering pipeline: merges ABS + RBA data into features.parquet."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"
OUTPUT_PATH = PROCESSED_DIR / "features.parquet"

# Quarters from which the RBA rate-hike structural break is flagged
_RATE_HIKE_START = pd.Period("2022Q3", freq="Q")


def _load_approvals(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the cleaned building approvals parquet or CSV."""
    if path is None:
        path = RAW_DIR / "approvals_clean.parquet"
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_cash_rate(path: Optional[Path] = None) -> pd.DataFrame:
    """Load quarterly RBA cash rate series."""
    if path is None:
        path = RAW_DIR / "rba_cash_rate.csv"
    from data.download import load_rba_cash_rate
    return load_rba_cash_rate(path)


def build_features(
    approvals_df: pd.DataFrame,
    cash_rate_df: pd.DataFrame,
    n_lags: int = 4,
) -> pd.DataFrame:
    """Construct the modelling feature set from raw input frames.

    Parameters
    ----------
    approvals_df:
        Long-format frame with columns lga_code, lga_name, quarter (Period), dwellings_approved.
    cash_rate_df:
        Quarterly frame with columns quarter (Period), cash_rate.
    n_lags:
        Number of autoregressive lags for dwellings_approved (default 4 quarters).

    Returns
    -------
    pd.DataFrame with one row per (lga_code, quarter), feature columns documented below.
    """
    df = approvals_df.copy()
    df["quarter"] = df["quarter"].dt.to_timestamp()

    # Sort for lag computation
    df = df.sort_values(["lga_code", "quarter"]).reset_index(drop=True)

    # Autoregressive lags
    for lag in range(1, n_lags + 1):
        df[f"approvals_lag{lag}"] = df.groupby("lga_code")["dwellings_approved"].shift(lag)

    # Seasonal dummies (quarter of year)
    df["quarter_dt"] = pd.to_datetime(df["quarter"])
    df["quarter_num"] = df["quarter_dt"].dt.quarter
    for q in range(1, 5):
        df[f"season_q{q}"] = (df["quarter_num"] == q).astype(int)

    # YoY change in approvals (same quarter, prior year)
    df["approvals_yoy"] = df.groupby(["lga_code", "quarter_num"])["dwellings_approved"].pct_change()

    # Merge cash rate
    cash_rate_df["quarter"] = cash_rate_df["quarter"].dt.to_timestamp()
    df = df.merge(cash_rate_df.rename(columns={"quarter": "quarter_dt_merge"}),
                  left_on="quarter_dt", right_on="quarter_dt_merge", how="left")

    # Rate lags
    df = df.sort_values(["lga_code", "quarter_dt"])
    df["cash_rate_lag1"] = df.groupby("lga_code")["cash_rate"].shift(1)
    df["cash_rate_lag2"] = df.groupby("lga_code")["cash_rate"].shift(2)

    # Structural break indicator: 1 from Q3 2022 onward
    break_ts = _RATE_HIKE_START.to_timestamp()
    df["post_rate_hike"] = (df["quarter_dt"] >= break_ts).astype(int)

    # Construction cost YoY placeholder (set to 0 if PPI data not available)
    if "construction_cost_index" in df.columns:
        df["construction_cost_yoy"] = df["construction_cost_index"].pct_change(4)
    else:
        df["construction_cost_yoy"] = 0.0

    # Population growth YoY placeholder
    if "population" in df.columns:
        df["population_growth_yoy"] = df.groupby("lga_code")["population"].pct_change(4)
    else:
        df["population_growth_yoy"] = 0.0

    # Final column selection
    feature_cols = [
        "lga_code",
        "lga_name",
        "quarter",
        "dwellings_approved",
        "cash_rate",
        "cash_rate_lag1",
        "cash_rate_lag2",
        "construction_cost_yoy",
        "population_growth_yoy",
        "approvals_lag1",
        "approvals_lag2",
        "approvals_lag3",
        "approvals_lag4",
        "approvals_yoy",
        "season_q1",
        "season_q2",
        "season_q3",
        "season_q4",
        "post_rate_hike",
    ]
    available = [c for c in feature_cols if c in df.columns]
    df = df[available].dropna(subset=["approvals_lag4"]).reset_index(drop=True)
    return df


def describe(df: pd.DataFrame) -> None:
    """Print a summary of the feature dataset: time range, LGA count, class distribution."""
    print(f"Time range:     {df['quarter'].min()} to {df['quarter'].max()}")
    print(f"LGAs:           {df['lga_code'].nunique()}")
    print(f"Total rows:     {len(df)}")
    print(f"Target (dwellings_approved):")
    print(f"  mean={df['dwellings_approved'].mean():.1f}  "
          f"std={df['dwellings_approved'].std():.1f}  "
          f"min={df['dwellings_approved'].min():.0f}  "
          f"max={df['dwellings_approved'].max():.0f}")
    missing_pct = df.isnull().mean().mul(100).round(1)
    cols_with_missing = missing_pct[missing_pct > 0]
    if len(cols_with_missing):
        print("Missing (%):")
        print(cols_with_missing.to_string())
    else:
        print("Missing:        none")


def run_pipeline(
    approvals_path: Optional[Path] = None,
    cash_rate_path: Optional[Path] = None,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Execute the full feature pipeline and write features.parquet."""
    print("Loading approvals data...")
    approvals = _load_approvals(approvals_path)

    print("Loading RBA cash rate...")
    cash_rate = _load_cash_rate(cash_rate_path)

    print("Building features...")
    features = build_features(approvals, cash_rate)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    print(f"Features written to {output_path}")
    describe(features)
    return features


if __name__ == "__main__":
    run_pipeline()
