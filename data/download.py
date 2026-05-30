"""Download ABS building approvals, population (ERP), and ABS PPI data to data/raw/."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

RAW_DIR = Path(__file__).parent / "raw"

# ABS Regional datasets: LGA2020 covers FY2010-11→2019-20, LGA2021 covers FY2020-21→present
_ABS_REGIONAL_LGA2020_CSV_URL = "https://data.api.abs.gov.au/files/ABS_ABS_REGIONAL_LGA2020_1.2.0.csv"
_ABS_REGIONAL_LGA2021_CSV_URL = (
    "https://data.api.abs.gov.au/rest/data/ABS,ABS_REGIONAL_LGA2021,1.6.0/all"
    "?dimensionAtObservation=AllDimensions&format=csvfilewithlabels"
)

_BUILDING_APPROVALS_MEASURE = "BUILDING_4"
_POPULATION_MEASURE = "ERP_P_20"  # Estimated Resident Population: Persons (no.)

# ABS PPI Table 17: Output of Construction Industries (quarterly, national)
# URL resolves from the "latest-release" redirect; update the date segment if needed.
_ABS_PPI_CONSTRUCTION_URL = (
    "https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/"
    "producer-price-indexes-australia/latest-release/6427017.xlsx"
)


def _get(url: str, timeout: int = 30) -> bytes:
    response = requests.get(url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    return response.content


def _download_abs_csv(url: str, filename: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_bytes(_get(url, timeout=120))
    print(f"Downloaded {filename} -> {out_path}")
    return out_path


def _load_abs_regional(path: Path, measure: str, value_col: str) -> pd.DataFrame:
    """Generic parser for ABS Regional LGA CSVs.

    Filters to *measure* and returns lga_code, quarter (pd.Period Q-JUN), and value_col.
    Financial-year strings like "2020-21" map to Q2 of the end calendar year (2021Q2).
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]

    lga_col = next((c for c in df.columns if c.startswith("LGA")), None)
    if lga_col is None:
        raise ValueError(f"No LGA column in {path}. Columns: {list(df.columns)}")

    if "MEASURE" in df.columns:
        available = df["MEASURE"].unique()
        if measure not in available:
            raise ValueError(
                f"Measure '{measure}' not found in {path.name}. "
                f"Available: {sorted(available)}"
            )
        df = df[df["MEASURE"] == measure]

    # The REGION column carries the human-readable LGA name in both LGA2020 and LGA2021 CSVs.
    has_name = "REGION" in df.columns
    cols = [lga_col] + (["REGION"] if has_name else []) + ["TIME_PERIOD", "OBS_VALUE"]
    df = df[cols].copy()
    col_names = ["lga_code"] + (["lga_name"] if has_name else []) + ["period_str", value_col]
    df.columns = col_names

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col, "lga_code"])

    def _to_period(s: str) -> pd.Period:
        s = str(s).strip()
        end_year = int(s.split("-")[0]) + 1 if "-" in s else int(s)
        return pd.Period(f"{end_year}Q2", freq="Q")

    df["quarter"] = df["period_str"].apply(_to_period)
    df["lga_code"] = df["lga_code"].astype(str).str.extract(r"(\d+)")[0]

    out_cols = ["lga_code"] + (["lga_name"] if has_name else []) + ["quarter", value_col]
    return df[out_cols]


def load_abs_building_approvals(path: Path) -> pd.DataFrame:
    """Parse ABS Regional LGA CSV into long-format building approvals.

    Returns columns: lga_code (str), lga_name (str), quarter (pd.Period Q-JUN),
    dwellings_approved (float).  lga_name comes from the REGION column in the CSV
    (e.g. "Surf Coast", "Moree Plains (A)"); falls back to lga_code if absent.
    """
    df = _load_abs_regional(path, _BUILDING_APPROVALS_MEASURE, "dwellings_approved")
    if "lga_name" not in df.columns:
        df["lga_name"] = df["lga_code"]
    return df[["lga_code", "lga_name", "quarter", "dwellings_approved"]]


def load_abs_population(path: Path) -> pd.DataFrame:
    """Parse ABS Regional LGA CSV into long-format population (ERP_P_20).

    Returns columns: lga_code (str), quarter (pd.Period Q-JUN), population (float).
    """
    return _load_abs_regional(path, _POPULATION_MEASURE, "population")


def download_abs_ppi(out_dir: Path = RAW_DIR) -> Optional[Path]:
    """Download ABS PPI Table 17 (Output of Construction Industries) xlsx.

    Returns the saved path, or None if the download fails.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ppi_construction.xlsx"
    try:
        out_path.write_bytes(_get(_ABS_PPI_CONSTRUCTION_URL, timeout=30))
        print(f"Downloaded ABS PPI construction -> {out_path}")
        return out_path
    except Exception as e:
        print(f"Warning: ABS PPI download failed ({e}). construction_cost_yoy will be 0.0.")
        return None


def load_abs_ppi(path: Path) -> Optional[pd.DataFrame]:
    """Parse ABS PPI Table 17 xlsx (Data1 sheet) into a quarterly house construction index.

    The Data1 sheet has series descriptions in row 0, 9 metadata rows (1-9),
    then date rows from row 10 onward with datetime objects in column A.
    Column 10 contains "3011 House construction Australia".

    Returns columns: quarter (pd.Period), construction_cost_index (float).
    Returns None if parsing fails.
    """
    try:
        raw = pd.read_excel(path, sheet_name="Data1", header=None)

        # Identify data start: first row where column A is a datetime object
        data_start = None
        for i in range(len(raw)):
            val = raw.iloc[i, 0]
            if hasattr(val, "year"):  # datetime / Timestamp
                data_start = i
                break

        if data_start is None:
            print("Warning: No datetime rows found in PPI Data1 sheet. Skipping.")
            return None

        # Find the "House construction Australia" column (row 0 headers)
        headers = raw.iloc[0]
        house_col_idx = 1  # fallback: national building construction
        for j, h in enumerate(headers):
            if "house construction australia" in str(h).lower():
                house_col_idx = j
                break

        data = raw.iloc[data_start:, [0, house_col_idx]].copy()
        data.columns = ["period_dt", "construction_cost_index"]
        data["construction_cost_index"] = pd.to_numeric(
            data["construction_cost_index"], errors="coerce"
        )
        data["quarter"] = pd.to_datetime(data["period_dt"], errors="coerce").dt.to_period("Q")
        data = data.dropna(subset=["quarter", "construction_cost_index"])

        if data.empty:
            print("Warning: No valid data found in PPI Data1 sheet. Skipping.")
            return None

        print(
            f"Loaded ABS PPI house construction: {len(data)} quarters, "
            f"{data['quarter'].min()} to {data['quarter'].max()}"
        )
        return (
            data[["quarter", "construction_cost_index"]]
            .sort_values("quarter")
            .reset_index(drop=True)
        )

    except Exception as e:
        print(f"Warning: PPI xlsx parse failed ({e}). construction_cost_yoy will be 0.0.")
        return None


def download_all(out_dir: Path = RAW_DIR) -> None:
    """Download all data sources and save processed parquet files."""
    print("Downloading ABS Regional LGA2020 CSV (FY2010-11 to FY2019-20)...")
    path_2020 = _download_abs_csv(_ABS_REGIONAL_LGA2020_CSV_URL, "abs_regional_lga2020.csv", out_dir)

    print("Downloading ABS Regional LGA2021 CSV (FY2020-21 to present)...")
    path_2021 = _download_abs_csv(_ABS_REGIONAL_LGA2021_CSV_URL, "abs_regional_lga2021.csv", out_dir)

    # Building approvals
    print("Parsing and consolidating building approvals...")
    ap20 = load_abs_building_approvals(path_2020)
    ap21 = load_abs_building_approvals(path_2021)
    approvals = pd.concat([ap20, ap21], ignore_index=True)
    approvals = (
        approvals
        .drop_duplicates(subset=["lga_code", "quarter"])
        .sort_values(["lga_code", "quarter"])
        .reset_index(drop=True)
    )
    # LGA2021 uses cleaner names (no "(C)"/"(A)" type suffixes); prefer those for
    # any code that appears in both editions so the whole series uses one name.
    preferred_names = ap21.drop_duplicates("lga_code").set_index("lga_code")["lga_name"]
    approvals["lga_name"] = approvals["lga_code"].map(preferred_names).fillna(approvals["lga_name"])

    parquet_path = out_dir / "approvals_clean.parquet"
    approvals.to_parquet(parquet_path, index=False)
    print(f"Saved approvals_clean.parquet -> {parquet_path}  ({len(approvals):,} rows)")

    # Population (ERP_P_20) — extracted from the already-downloaded CSVs
    print("Extracting population (ERP) from ABS Regional CSVs...")
    pop = pd.concat(
        [load_abs_population(path_2020), load_abs_population(path_2021)],
        ignore_index=True,
    )
    pop = (
        pop
        .drop_duplicates(subset=["lga_code", "quarter"])
        .sort_values(["lga_code", "quarter"])
        .reset_index(drop=True)
    )
    pop_path = out_dir / "population_clean.parquet"
    pop.to_parquet(pop_path, index=False)
    print(f"Saved population_clean.parquet -> {pop_path}  ({len(pop):,} rows)")

    # PPI construction costs (best-effort)
    print("Downloading ABS PPI construction costs (best-effort)...")
    ppi_xlsx = download_abs_ppi(out_dir)
    if ppi_xlsx is not None:
        ppi_df = load_abs_ppi(ppi_xlsx)
        if ppi_df is not None:
            ppi_path = out_dir / "ppi_construction.parquet"
            ppi_df.to_parquet(ppi_path, index=False)
            print(f"Saved ppi_construction.parquet -> {ppi_path}  ({len(ppi_df)} rows)")

    print("All downloads complete.")


if __name__ == "__main__":
    download_all()
