"""
Data Loader and Validator Module for CloudWise.

Loads, validates, and cleans compute usage data from CSV sources.
Enforces strict schema constraints, validates physical bounds (0 <= usage_hours <= 24,
non-negative costs), calculates ground-truth history from dates, and produces
auditable validation reports without silently masking bad data.
"""

from dataclasses import dataclass, field
from io import StringIO, BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "date",
    "project_id",
    "resource_id",
    "machine_type",
    "region",
    "usage_hours",
    "on_demand_cost",
    "existing_commitment",
    "sud_eligible",
    "history_days",
]


@dataclass
class DataLoadResult:
    """Structured result returned by the data loader."""
    df: Optional[pd.DataFrame] = None
    is_valid: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def has_errors(self) -> bool:
        return len(self.errors) > 0


def load_usage_data(
    source: Union[str, Path, StringIO, BytesIO, Any],
    max_daily_hours: float = 24.0,
    allow_deduplication: bool = True,
) -> DataLoadResult:
    """
    Load and rigorously validate Google Cloud compute usage CSV data.

    Parameters
    ----------
    source : Union[str, Path, StringIO, BytesIO, Any]
        Filepath, path object, StringIO, or Streamlit UploadedFile.
    max_daily_hours : float
        Maximum physically plausible compute hours per single instance per day (default 24.0).
    allow_deduplication : bool
        If True, deduplicates exact (date, resource_id) duplicates after recording a warning.

    Returns
    -------
    DataLoadResult
        Result object containing cleaned DataFrame, validation flag, errors, and warnings.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Ingestion / Parse CSV
    try:
        if hasattr(source, "read"):
            # File-like object (StringIO, BytesIO, UploadedFile)
            source.seek(0)
            raw_df = pd.read_csv(source)
        elif isinstance(source, (str, Path)):
            path_obj = Path(source)
            if not path_obj.exists():
                return DataLoadResult(
                    is_valid=False,
                    errors=[f"Source file not found at path: {source}"],
                )
            raw_df = pd.read_csv(path_obj)
        else:
            return DataLoadResult(
                is_valid=False,
                errors=[f"Unsupported data source type: {type(source)}"],
            )
    except pd.errors.EmptyDataError:
        return DataLoadResult(
            is_valid=False,
            errors=["Uploaded CSV file is completely empty or has no data."],
        )
    except Exception as e:
        return DataLoadResult(
            is_valid=False,
            errors=[f"Failed to parse CSV file: {str(e)}"],
        )

    if raw_df.empty:
        return DataLoadResult(
            is_valid=False,
            errors=["Uploaded CSV file is completely empty."],
        )

    # 2. Schema / Column validation
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in raw_df.columns]
    if missing_cols:
        return DataLoadResult(
            is_valid=False,
            errors=[f"Missing required columns in dataset: {missing_cols}. Required schema: {REQUIRED_COLUMNS}"],
        )

    df = raw_df.copy()

    # 3. Check for missing (NaN) values in essential columns
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        null_details = [f"{col}: {cnt} null rows" for col, cnt in cols_with_nulls.items()]
        errors.append(f"Missing values found in required columns: {', '.join(null_details)}")

    # 4. Date validation
    try:
        df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce")
        invalid_dates = df[df["parsed_date"].isnull()]
        if len(invalid_dates) > 0:
            sample_bad = df.loc[df["parsed_date"].isnull(), "date"].head(3).tolist()
            errors.append(
                f"Found {len(invalid_dates)} rows with invalid or malformed dates (e.g. {sample_bad}). "
                "Dates must follow standard YYYY-MM-DD ISO format."
            )
    except Exception as e:
        errors.append(f"Error parsing date column: {str(e)}")

    # 5. Usage Hours validation (physically bounded: 0.0 <= usage_hours <= max_daily_hours)
    try:
        df["usage_hours"] = pd.to_numeric(df["usage_hours"], errors="coerce")
        nan_hours = df[df["usage_hours"].isnull()]
        if len(nan_hours) > 0:
            errors.append(f"Found {len(nan_hours)} non-numeric values in 'usage_hours' column.")

        neg_hours = df[df["usage_hours"] < 0.0]
        if len(neg_hours) > 0:
            sample_neg = neg_hours["usage_hours"].head(3).tolist()
            errors.append(
                f"Found {len(neg_hours)} rows with negative usage_hours (e.g. {sample_neg}). Usage cannot be negative."
            )

        excess_hours = df[df["usage_hours"] > max_daily_hours]
        if len(excess_hours) > 0:
            sample_excess = excess_hours["usage_hours"].head(3).tolist()
            errors.append(
                f"Found {len(excess_hours)} rows with impossible usage_hours > {max_daily_hours} hrs/day (e.g. {sample_excess}). "
                "Daily instance usage cannot exceed 24 hours per day."
            )
    except Exception as e:
        errors.append(f"Error validating usage_hours: {str(e)}")

    # 6. Cost validation (must be non-negative)
    try:
        df["on_demand_cost"] = pd.to_numeric(df["on_demand_cost"], errors="coerce")
        nan_cost = df[df["on_demand_cost"].isnull()]
        if len(nan_cost) > 0:
            errors.append(f"Found {len(nan_cost)} non-numeric values in 'on_demand_cost' column.")

        neg_cost = df[df["on_demand_cost"] < 0.0]
        if len(neg_cost) > 0:
            sample_cost = neg_cost["on_demand_cost"].head(3).tolist()
            errors.append(
                f"Found {len(neg_cost)} rows with negative on_demand_cost (e.g. {sample_cost}). Base cost cannot be negative."
            )
    except Exception as e:
        errors.append(f"Error validating on_demand_cost: {str(e)}")

    # 7. Existing commitment validation (ratio 0.0 to 1.0)
    try:
        df["existing_commitment"] = pd.to_numeric(df["existing_commitment"], errors="coerce").fillna(0.0)
        invalid_commit = df[(df["existing_commitment"] < 0.0) | (df["existing_commitment"] > 1.0)]
        if len(invalid_commit) > 0:
            sample_commit = invalid_commit["existing_commitment"].head(3).tolist()
            warnings.append(
                f"Found {len(invalid_commit)} rows with existing_commitment outside [0.0, 1.0] (e.g. {sample_commit}). "
                "Commitment coverage ratio should be between 0% (0.0) and 100% (1.0)."
            )
    except Exception as e:
        warnings.append(f"Error validating existing_commitment: {str(e)}")

    # 8. SUD eligibility parsing
    try:
        if df["sud_eligible"].dtype == object:
            # Handle string Booleans ('True', 'False', 'true', 'false', '1', '0')
            df["sud_eligible"] = df["sud_eligible"].astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"])
        else:
            df["sud_eligible"] = df["sud_eligible"].astype(bool)
    except Exception as e:
        warnings.append(f"Error parsing sud_eligible flag: {str(e)}")

    # If critical errors are already found, do not proceed with invalid conversions
    if errors:
        return DataLoadResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
        )

    # 9. Duplicate Record Detection
    # Look for multiple entries of (parsed_date, resource_id)
    dup_mask = df.duplicated(subset=["parsed_date", "resource_id"], keep=False)
    num_dups = dup_mask.sum()
    if num_dups > 0:
        dup_samples = df[dup_mask][["date", "resource_id"]].head(4).to_dict(orient="records")
        if allow_deduplication:
            warnings.append(
                f"Detected {num_dups} duplicate records for the same (date, resource_id). "
                f"Samples: {dup_samples}. Exact duplicates were consolidated by retaining the first entry."
            )
            df = df.drop_duplicates(subset=["parsed_date", "resource_id"], keep="first")
        else:
            errors.append(
                f"Detected {num_dups} duplicate records for (date, resource_id) without deduplication allowed."
            )
            return DataLoadResult(is_valid=False, errors=errors, warnings=warnings)

    # 10. Actual History Length Calculation from Dates
    # Instead of blindly trusting CSV 'history_days', compute from ground-truth date span
    df["date"] = df["parsed_date"].dt.strftime("%Y-%m-%d")

    workload_history_info = {}
    for res_id, group in df.groupby("resource_id"):
        min_d = group["parsed_date"].min()
        max_d = group["parsed_date"].max()
        actual_span_days = int((max_d - min_d).days) + 1
        active_days = int(group["parsed_date"].nunique())
        stated_history = float(group["history_days"].iloc[0]) if "history_days" in group.columns else actual_span_days

        # Check for discrepancy between stated history_days and actual calendar span
        if abs(stated_history - actual_span_days) > 2:
            warnings.append(
                f"Workload '{res_id}': CSV stated history_days={int(stated_history)}, "
                f"but actual date range span is {actual_span_days} days ({min_d.strftime('%Y-%m-%d')} to {max_d.strftime('%Y-%m-%d')}). "
                "CloudWise uses ground-truth calendar date span."
            )

        workload_history_info[res_id] = {
            "calculated_history_days": actual_span_days,
            "active_days": active_days,
            "completeness_pct": round((active_days / max(actual_span_days, 1)) * 100, 1),
            "start_date": min_d.strftime("%Y-%m-%d"),
            "end_date": max_d.strftime("%Y-%m-%d"),
        }

    # Map ground truth history days to dataframe
    df["calculated_history_days"] = df["resource_id"].map(
        lambda r: workload_history_info[r]["calculated_history_days"]
    )
    df["active_recorded_days"] = df["resource_id"].map(
        lambda r: workload_history_info[r]["active_days"]
    )
    df["data_completeness_pct"] = df["resource_id"].map(
        lambda r: workload_history_info[r]["completeness_pct"]
    )

    # Cleanup temporary columns
    df = df.drop(columns=["parsed_date"])

    # 11. Compile summary statistics
    min_overall_date = df["date"].min()
    max_overall_date = df["date"].max()
    overall_span = (pd.to_datetime(max_overall_date) - pd.to_datetime(min_overall_date)).days + 1

    summary = {
        "total_records": len(df),
        "unique_workloads": int(df["resource_id"].nunique()),
        "unique_projects": int(df["project_id"].nunique()),
        "unique_regions": int(df["region"].nunique()),
        "unique_machine_types": int(df["machine_type"].nunique()),
        "date_range": f"{min_overall_date} to {max_overall_date} ({overall_span} calendar days)",
        "total_on_demand_cost": round(float(df["on_demand_cost"].sum()), 2),
        "total_usage_hours": round(float(df["usage_hours"].sum()), 2),
        "workload_history": workload_history_info,
    }

    return DataLoadResult(
        df=df,
        is_valid=True,
        errors=[],
        warnings=warnings,
        summary=summary,
    )


def get_sample_csv_path() -> Path:
    """Return absolute path to the bundled sample dataset."""
    module_dir = Path(__file__).resolve().parent
    # Check parent/data/sample_usage.csv
    candidate_1 = module_dir.parent / "data" / "sample_usage.csv"
    if candidate_1.exists():
        return candidate_1
    # Check root data/sample_usage.csv
    candidate_2 = module_dir.parent.parent / "data" / "sample_usage.csv"
    if candidate_2.exists():
        return candidate_2
    return candidate_1
