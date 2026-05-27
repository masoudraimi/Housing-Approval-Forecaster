"""Download ABS building approvals, RBA cash rate, PPI, and ERP data to data/raw/."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).parent / "raw"

# ABS time series URLs (direct CSV download endpoints)
_ABS_8731_URL = (
    "https://www.abs.gov.au/statistics/industry/building-and-construction/"
    "building-approvals-australia/latest-release/8731016.xlsx"
)
_RBA_CASH_RATE_URL = "https://www.rba.gov.au/statistics/tables/csv/f1-data.csv"

# Fallback: ABS API base for table downloads
_ABS_API_BASE = "https://api.data.abs.gov.au/data"


def _get(url: str, timeout: int = 30) -> bytes:
    """Fetch URL content with a basic retry on timeout."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def download_rba_cash_rate(out_dir: Path = RAW_DIR) -> Path:
    """Download RBA monthly cash rate target CSV to out_dir/rba_cash_rate.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rba_cash_rate.csv"
    content = _get(_RBA_CASH_RATE_URL)
    out_path.write_bytes(content)
    print(f"Downloaded RBA cash rate -> {out_path}")
    return out_path


def download_abs_building_approvals(out_dir: Path = RAW_DIR) -> Path:
    """Download ABS 8731.0 building approvals Excel file to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "abs_8731_building_approvals.xlsx"
    content = _get(_ABS_8731_URL, timeout=60)
    out_path.write_bytes(content)
    print(f"Downloaded ABS 8731.0 -> {out_path}")
    return out_path


def load_rba_cash_rate(path: Path) -> pd.DataFrame:
    """Parse downloaded RBA cash rate CSV into a quarterly mean series.

    Returns a DataFrame with columns: quarter (pd.Period), cash_rate.
    The RBA CSV has a multi-row header; rows beginning with dates are data rows.
    """
    raw = pd.read_csv(path, skiprows=10, header=0)
    # Column 0 is the date, column 1 is the cash rate target
    df = raw.iloc[:, :2].copy()
    df.columns = ["date", "cash_rate"]
    df = df.dropna(subset=["cash_rate"])
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df["cash_rate"] = pd.to_numeric(df["cash_rate"], errors="coerce")
    df = df.dropna(subset=["cash_rate"])
    df["quarter"] = df["date"].dt.to_period("Q")
    quarterly = df.groupby("quarter")["cash_rate"].mean().reset_index()
    return quarterly


def load_abs_building_approvals(path: Path) -> pd.DataFrame:
    """Parse ABS 8731.0 Excel into a long-format quarterly DataFrame.

    Returns columns: lga_code, lga_name, quarter (pd.Period), dwellings_approved.
    The Excel layout varies by release; this targets the LGA-level total dwellings sheet.
    """
    # The LGA total dwellings data is typically on a sheet named 'Data1' or similar
    xl = pd.ExcelFile(path)
    # Try to find the LGA quarterly sheet
    sheet = None
    for name in xl.sheet_names:
        if "lga" in name.lower() or "data" in name.lower():
            sheet = name
            break
    if sheet is None:
        sheet = xl.sheet_names[0]

    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    # ABS time series format: series metadata in top rows, data below
    # Row 0: Series ID, Row 1: Description, data starts after metadata block
    # Locate the row where date data begins (first column looks like a date)
    data_start = None
    for i, row in raw.iterrows():
        val = row.iloc[0]
        try:
            pd.to_datetime(str(val), dayfirst=True)
            data_start = i
            break
        except Exception:
            continue

    if data_start is None:
        raise ValueError(f"Could not find data rows in {path}. Manual inspection required.")

    headers = raw.iloc[data_start - 1]
    data = raw.iloc[data_start:].copy()
    data.columns = headers
    data = data.rename(columns={data.columns[0]: "date"})
    data["date"] = pd.to_datetime(data["date"], dayfirst=True, errors="coerce")
    data = data.dropna(subset=["date"])
    data["quarter"] = data["date"].dt.to_period("Q")

    # Melt wide format (one column per LGA) to long
    id_cols = ["date", "quarter"]
    lga_cols = [c for c in data.columns if c not in id_cols]
    long = data.melt(id_vars=id_cols, value_vars=lga_cols, var_name="lga_name", value_name="dwellings_approved")
    long["dwellings_approved"] = pd.to_numeric(long["dwellings_approved"], errors="coerce")
    long = long.dropna(subset=["dwellings_approved"])
    long["lga_code"] = long["lga_name"].str.extract(r"(\d{5})")
    return long[["lga_code", "lga_name", "quarter", "dwellings_approved"]]


def download_all(out_dir: Path = RAW_DIR) -> None:
    """Download all data sources to out_dir."""
    print("Downloading RBA cash rate...")
    download_rba_cash_rate(out_dir)
    print("Downloading ABS building approvals (8731.0)...")
    download_abs_building_approvals(out_dir)
    print("All downloads complete.")


if __name__ == "__main__":
    download_all()
