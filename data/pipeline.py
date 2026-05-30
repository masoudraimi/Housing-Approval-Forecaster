"""Feature engineering pipeline: merges ABS approvals, population (ERP), and PPI into features.parquet."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"
OUTPUT_PATH = PROCESSED_DIR / "features.parquet"

# National Housing Accord announced August 2022 — planning/governance policy break
_ACCORD_START = pd.Period("2022Q3", freq="Q")


def _load_approvals(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = RAW_DIR / "approvals_clean.parquet"
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _load_population(path: Optional[Path] = None) -> pd.DataFrame:
    """Load LGA-level population (ERP) parquet, or return empty frame if missing."""
    if path is None:
        path = RAW_DIR / "population_clean.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["lga_code", "quarter", "population"])
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _load_ppi(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load quarterly national construction PPI, or None if not available."""
    if path is None:
        path = RAW_DIR / "ppi_construction.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    except Exception:
        return None


def build_features(
    approvals_df: pd.DataFrame,
    population_df: pd.DataFrame,
    ppi_df: Optional[pd.DataFrame] = None,
    n_lags: int = 1,
) -> pd.DataFrame:
    """Construct the modelling feature set.

    Parameters
    ----------
    approvals_df:
        Long-format: lga_code, lga_name, quarter (Period), dwellings_approved.
    population_df:
        Long-format: lga_code, quarter (Period), population.
        Pass an empty DataFrame if unavailable; population_growth_yoy will be 0.0.
    ppi_df:
        National quarterly: quarter (Period), construction_cost_index. Optional.
        If None, construction_cost_yoy will be 0.0.
    n_lags:
        Number of autoregressive lags for dwellings_approved.
    """
    df = approvals_df.copy()
    df["quarter"] = df["quarter"].dt.to_timestamp()
    df = df.sort_values(["lga_code", "quarter"]).reset_index(drop=True)

    # Autoregressive lags
    for lag in range(1, n_lags + 1):
        df[f"approvals_lag{lag}"] = df.groupby("lga_code")["dwellings_approved"].shift(lag)

    # Seasonal dummies
    df["quarter_dt"] = pd.to_datetime(df["quarter"])
    df["quarter_num"] = df["quarter_dt"].dt.quarter
    for q in range(1, 5):
        df[f"season_q{q}"] = (df["quarter_num"] == q).astype(int)

    # YoY approvals change
    df["approvals_yoy"] = df.groupby(["lga_code", "quarter_num"])["dwellings_approved"].pct_change()

    # Population growth YoY — LGA-level from ABS ERP
    if not population_df.empty and "population" in population_df.columns:
        pop = population_df.copy()
        pop["quarter"] = pop["quarter"].dt.to_timestamp()
        df = df.merge(
            pop[["lga_code", "quarter", "population"]].rename(columns={"quarter": "pop_quarter"}),
            left_on=["lga_code", "quarter_dt"],
            right_on=["lga_code", "pop_quarter"],
            how="left",
        ).drop(columns=["pop_quarter"], errors="ignore")
        df["population_growth_yoy"] = (
            df.sort_values(["lga_code", "quarter"])
              .groupby("lga_code")["population"]
              .pct_change()
        )
    else:
        df["population_growth_yoy"] = 0.0

    # Construction cost YoY — national, from ABS PPI (broadcast to all LGAs)
    if ppi_df is not None and not ppi_df.empty and "construction_cost_index" in ppi_df.columns:
        ppi = ppi_df.copy()
        ppi["quarter"] = ppi["quarter"].dt.to_timestamp()
        # 4-period YoY on quarterly series; after merge with annual data this is YoY
        ppi["construction_cost_yoy"] = ppi["construction_cost_index"].pct_change(4)
        df = df.merge(
            ppi[["quarter", "construction_cost_yoy"]].rename(columns={"quarter": "ppi_quarter"}),
            left_on="quarter_dt",
            right_on="ppi_quarter",
            how="left",
        ).drop(columns=["ppi_quarter"], errors="ignore")
    else:
        df["construction_cost_yoy"] = 0.0

    # Planning policy break: National Housing Accord (Aug 2022 = 2022Q3)
    break_ts = _ACCORD_START.to_timestamp()
    df["post_accord_2022"] = (df["quarter_dt"] >= break_ts).astype(int)

    # Final column selection
    lag_cols = [f"approvals_lag{i}" for i in range(1, n_lags + 1)]
    feature_cols = [
        "lga_code",
        "lga_name",
        "quarter",
        "dwellings_approved",
        "population_growth_yoy",
        "construction_cost_yoy",
        *lag_cols,
        "approvals_yoy",
        "season_q1",
        "season_q2",
        "season_q3",
        "season_q4",
        "post_accord_2022",
    ]
    available = [c for c in feature_cols if c in df.columns]
    last_lag = f"approvals_lag{n_lags}"
    df = df[available].dropna(subset=[last_lag]).reset_index(drop=True)
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
    population_path: Optional[Path] = None,
    ppi_path: Optional[Path] = None,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Execute the full feature pipeline and write features.parquet."""
    print("Loading approvals data...")
    approvals = _load_approvals(approvals_path)

    print("Loading population data...")
    population = _load_population(population_path)
    if population.empty:
        print("  Warning: population_clean.parquet not found. Run data.download first.")

    print("Loading PPI data...")
    ppi = _load_ppi(ppi_path)
    if ppi is None:
        print("  Note: ppi_construction.parquet not found. construction_cost_yoy will be 0.0.")

    print("Building features...")
    features = build_features(approvals, population, ppi)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    print(f"Features written to {output_path}")
    describe(features)
    return features


if __name__ == "__main__":
    run_pipeline()
