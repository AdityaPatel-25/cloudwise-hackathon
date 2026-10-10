"""
Unit tests for src/data_loader.py.

Verifies schema enforcement, date parsing, physical usage bounds,
negative cost rejection, ground-truth history calculation, and deduplication.
"""

from io import StringIO
import pytest
import pandas as pd
from src.data_loader import load_usage_data, get_sample_csv_path, REQUIRED_COLUMNS


def create_csv_string(rows_content: str) -> StringIO:
    header = "date,project_id,resource_id,machine_type,region,usage_hours,on_demand_cost,existing_commitment,sud_eligible,history_days\n"
    return StringIO(header + rows_content)


def test_load_bundled_sample_dataset():
    """Test loading the bundled sample_usage.csv."""
    path = get_sample_csv_path()
    assert path.exists(), f"Sample dataset not found at {path}"

    res = load_usage_data(path)
    assert res.is_valid is True
    assert len(res.errors) == 0
    assert res.df is not None
    assert len(res.df) > 400
    assert res.summary["unique_workloads"] == 6
    assert "calculated_history_days" in res.df.columns


def test_valid_custom_csv_loading():
    """Test loading a clean, valid custom CSV."""
    data = (
        "2026-07-01,proj-test,vm-01,n2-standard-4,us-central1,24.0,5.00,0.0,True,3\n"
        "2026-07-02,proj-test,vm-01,n2-standard-4,us-central1,24.0,5.00,0.0,True,3\n"
        "2026-07-03,proj-test,vm-01,n2-standard-4,us-central1,24.0,5.00,0.0,True,3\n"
    )
    res = load_usage_data(create_csv_string(data))
    assert res.is_valid is True
    assert len(res.errors) == 0
    assert len(res.df) == 3
    # Check calculated history days from date range (3 days: 07-01 to 07-03)
    assert res.df["calculated_history_days"].iloc[0] == 3


def test_missing_required_column():
    """Test rejection when a required column is missing."""
    csv_missing = StringIO("date,project_id,resource_id,machine_type,region,usage_hours,on_demand_cost\n2026-07-01,p,r,m,reg,10,1.0\n")
    res = load_usage_data(csv_missing)
    assert res.is_valid is False
    assert any("Missing required columns" in err for err in res.errors)


def test_invalid_dates_rejection():
    """Test rejection when dates are malformed."""
    data = "not-a-date,proj-test,vm-01,n2-standard-4,us-central1,24.0,5.00,0.0,True,1\n"
    res = load_usage_data(create_csv_string(data))
    assert res.is_valid is False
    assert any("invalid or malformed dates" in err for err in res.errors)


def test_negative_cost_rejection():
    """Test rejection of negative on-demand costs."""
    data = "2026-07-01,proj-test,vm-01,n2-standard-4,us-central1,24.0,-10.50,0.0,True,1\n"
    res = load_usage_data(create_csv_string(data))
    assert res.is_valid is False
    assert any("negative on_demand_cost" in err for err in res.errors)


def test_impossible_usage_hours_rejection():
    """Test rejection when usage hours exceed 24.0 hours/day for an instance."""
    data = "2026-07-01,proj-test,vm-01,n2-standard-4,us-central1,28.5,5.00,0.0,True,1\n"
    res = load_usage_data(create_csv_string(data))
    assert res.is_valid is False
    assert any("impossible usage_hours > 24.0" in err for err in res.errors)


def test_negative_usage_hours_rejection():
    """Test rejection of negative usage hours."""
    data = "2026-07-01,proj-test,vm-01,n2-standard-4,us-central1,-2.0,5.00,0.0,True,1\n"
    res = load_usage_data(create_csv_string(data))
    assert res.is_valid is False
    assert any("negative usage_hours" in err for err in res.errors)


def test_empty_csv_rejection():
    """Test handling of completely empty CSV."""
    res = load_usage_data(StringIO(""))
    assert res.is_valid is False
    assert any("empty" in err.lower() for err in res.errors)


def test_history_days_calculation_from_dates():
    """Test that actual date calendar span is calculated rather than blindly trusting history_days."""
    # Data has 2 rows separated by 10 days, but claims history_days=99
    data = (
        "2026-07-01,proj-test,vm-01,n2-standard-4,us-central1,24.0,5.00,0.0,True,99\n"
        "2026-07-10,proj-test,vm-01,n2-standard-4,us-central1,24.0,5.00,0.0,True,99\n"
    )
    res = load_usage_data(create_csv_string(data))
    assert res.is_valid is True
    # 2026-07-01 to 2026-07-10 is 10 days span
    assert res.df["calculated_history_days"].iloc[0] == 10
    # There should be a warning about the discrepancy
    assert any("discrepancy" in w.lower() or "stated history_days=99" in w for w in res.warnings)


def test_deduplication_of_records():
    """Test detection and handling of duplicate (date, resource_id) rows."""
    data = (
        "2026-07-01,proj-test,vm-01,n2-standard-4,us-central1,24.0,5.00,0.0,True,1\n"
        "2026-07-01,proj-test,vm-01,n2-standard-4,us-central1,24.0,5.00,0.0,True,1\n"
    )
    res = load_usage_data(create_csv_string(data), allow_deduplication=True)
    assert res.is_valid is True
    assert len(res.df) == 1
    assert any("duplicate records" in w.lower() for w in res.warnings)
