"""Download ABS building approvals, RBA cash rate, PPI, and ERP data to data/raw/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).parent / "raw"

# ABS Regional datasets: LGA2020 covers FY2010-11→2019-20, LGA2021 covers FY2020-21→present
_ABS_REGIONAL_LGA2020_CSV_URL = "https://data.api.abs.gov.au/files/ABS_ABS_REGIONAL_LGA2020_1.2.0.csv"
_ABS_REGIONAL_LGA2021_CSV_URL = (
    "https://data.api.abs.gov.au/rest/data/ABS,ABS_REGIONAL_LGA2021,1.6.0/all?dimensionAtObservation=AllDimensions&format=csvfilewithlabels"
)


# Measure code for total dwelling units in ABS_REGIONAL_LGA2021.
_BUILDING_APPROVALS_MEASURE = "BUILDING_4"

_RBA_CASH_RATE_URL = "https://www.rba.gov.au/statistics/tables/csv/f1-data.csv"


def _get(url: str, timeout: int = 30) -> bytes:
    response = requests.get(url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    return response.content


def download_rba_cash_rate(out_dir: Path = RAW_DIR) -> Path:
    """Download RBA monthly cash rate target CSV to out_dir/rba_cash_rate.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rba_cash_rate.csv"
    out_path.write_bytes(_get(_RBA_CASH_RATE_URL))
    print(f"Downloaded RBA cash rate -> {out_path}")
    return out_path


def _download_abs_csv(url: str, filename: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_bytes(_get(url, timeout=120))
    print(f"Downloaded {filename} -> {out_path}")
    return out_path


def load_rba_cash_rate(path: Path) -> pd.DataFrame:
    """Parse downloaded RBA cash rate CSV into a quarterly mean series.

    Returns columns: quarter (pd.Period), cash_rate.
    """
    raw = pd.read_csv(path, skiprows=10, header=0)
    df = raw.iloc[:, :2].copy()
    df.columns = ["date", "cash_rate"]
    df = df.dropna(subset=["cash_rate"])
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df["cash_rate"] = pd.to_numeric(df["cash_rate"], errors="coerce")
    df = df.dropna(subset=["cash_rate"])
    df["quarter"] = df["date"].dt.to_period("Q")
    return df.groupby("quarter")["cash_rate"].mean().reset_index()


def load_abs_building_approvals(path: Path) -> pd.DataFrame:
    """Parse an ABS Regional LGA CSV (LGA2020 or LGA2021) into a long-format DataFrame.

    Filters to the building approvals measure (_BUILDING_APPROVALS_MEASURE) and
    returns columns: lga_code (str), lga_name (str), quarter (pd.Period Q-JUN),
    dwellings_approved (float).

    ABS Regional data is published by financial year (e.g. "2020-21").
    Each observation maps to Q2 of the ending calendar year to align with the
    Australian June financial year-end.
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]

    # Identify the LGA dimension column
    lga_col = next((c for c in df.columns if c.startswith("LGA")), None)
    if lga_col is None:
        raise ValueError(
            f"No LGA column found in {path}.\nAvailable columns: {list(df.columns)}"
        )

    # Filter to building approvals measure
    if "MEASURE" in df.columns:
        available = df["MEASURE"].unique()
        if _BUILDING_APPROVALS_MEASURE not in available:
            raise ValueError(
                f"Measure '{_BUILDING_APPROVALS_MEASURE}' not found.\n"
                f"Available measures: {sorted(available)}\n"
                f"Update _BUILDING_APPROVALS_MEASURE in download.py."
            )
        df = df[df["MEASURE"] == _BUILDING_APPROVALS_MEASURE]

    df = df[[lga_col, "TIME_PERIOD", "OBS_VALUE"]].copy()
    df.columns = ["lga_code", "period_str", "dwellings_approved"]
    df["dwellings_approved"] = pd.to_numeric(df["dwellings_approved"], errors="coerce")
    df = df.dropna(subset=["dwellings_approved", "lga_code"])

    # Financial year "2020-21" → Q2 of end year (2021Q2 = Jun 2021)
    # Plain year "2020" → 2020Q2
    def _to_period(s: str) -> pd.Period:
        s = str(s).strip()
        end_year = int(s.split("-")[0]) + 1 if "-" in s else int(s)
        return pd.Period(f"{end_year}Q2", freq="Q")

    df["quarter"] = df["period_str"].apply(_to_period)
    df["lga_code"] = df["lga_code"].astype(str).str.extract(r"(\d+)")[0]
    df["lga_name"] = df["lga_code"]

    return df[["lga_code", "lga_name", "quarter", "dwellings_approved"]]


def download_all(out_dir: Path = RAW_DIR) -> None:
    """Download all sources, parse building approvals, and save approvals_clean.parquet."""
    print("Downloading RBA cash rate...")
    download_rba_cash_rate(out_dir)

    print("Downloading ABS Regional LGA2020 CSV (FY2010-11 to FY2019-20)...")
    path_2020 = _download_abs_csv(_ABS_REGIONAL_LGA2020_CSV_URL, "abs_regional_lga2020.csv", out_dir)

    print("Downloading ABS Regional LGA2021 CSV (FY2020-21 to present)...")
    path_2021 = _download_abs_csv(_ABS_REGIONAL_LGA2021_CSV_URL, "abs_regional_lga2021.csv", out_dir)

    print("Parsing and consolidating building approvals...")
    approvals = pd.concat(
        [load_abs_building_approvals(path_2020), load_abs_building_approvals(path_2021)],
        ignore_index=True,
    )
    approvals = (
        approvals
        .drop_duplicates(subset=["lga_code", "quarter"])
        .sort_values(["lga_code", "quarter"])
        .reset_index(drop=True)
    )
    parquet_path = out_dir / "approvals_clean.parquet"
    approvals.to_parquet(parquet_path, index=False)
    print(f"Saved approvals_clean.parquet -> {parquet_path}  ({len(approvals):,} rows)")
    print("All downloads complete.")


if __name__ == "__main__":
    download_all()
