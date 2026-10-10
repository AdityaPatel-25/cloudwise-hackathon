"""
Deterministic Synthetic Dataset Generator for CloudWise Hackathon MVP.
Generates ~90 days of synthetic daily Google Cloud compute usage across 6 workloads
illustrating different operational patterns:
1. Stable Production Workload (wl-prod-db-01)
2. Production with Existing Commitment Coverage (wl-prod-api-cluster)
3. Business-Hours Development Workload (wl-dev-microservices)
4. Bursty / Variable Analytics Workload (wl-batch-etl-spikes)
5. Declining / Migrated Workload (wl-legacy-monolith)
6. Short-History Pilot Workload (wl-pilot-inference)

NOTE: All pricing, discount rates, and usage records are illustrative and synthetic.
They do not represent actual Google Cloud pricing or customer billing data.
"""

import os
from datetime import date, timedelta
import numpy as np
import pandas as pd


def generate_synthetic_usage_data(
    start_date: date = date(2026, 7, 1),
    num_days: int = 90,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    records = []

    # Hourly rates (illustrative synthetic USD rates)
    rates = {
        "n2-standard-8": 0.388,
        "c2-standard-4": 0.208,
        "e2-standard-4": 0.134,
        "n1-standard-8": 0.380,
        "n1-standard-4": 0.190,
        "n2-standard-4": 0.194,
    }

    dates = [start_date + timedelta(days=i) for i in range(num_days)]

    for day_idx, current_date in enumerate(dates):
        iso_date = current_date.isoformat()
        is_weekend = current_date.weekday() >= 5  # 5=Sat, 6=Sun

        # 1. Stable Production DB: runs 24/7 with tiny random fluctuations
        h1 = round(float(np.clip(np.random.normal(23.9, 0.08), 23.0, 24.0)), 2)
        cost1 = round(h1 * rates["n2-standard-8"], 4)
        records.append({
            "date": iso_date,
            "project_id": "prj-production-core",
            "resource_id": "wl-prod-db-01",
            "machine_type": "n2-standard-8",
            "region": "us-central1",
            "usage_hours": h1,
            "on_demand_cost": cost1,
            "existing_commitment": 0.0,
            "sud_eligible": True,
            "history_days": num_days,
        })

        # 2. Production API Cluster: 24/7 with 50% existing commitment coverage
        h2 = round(float(np.clip(np.random.normal(23.95, 0.05), 23.5, 24.0)), 2)
        cost2 = round(h2 * rates["c2-standard-4"], 4)
        records.append({
            "date": iso_date,
            "project_id": "prj-production-core",
            "resource_id": "wl-prod-api-cluster",
            "machine_type": "c2-standard-4",
            "region": "us-central1",
            "usage_hours": h2,
            "on_demand_cost": cost2,
            "existing_commitment": 0.50,  # 50% existing CUD
            "sud_eligible": True,
            "history_days": num_days,
        })

        # 3. Dev Microservices: business hours (Mon-Fri ~8.5-10h, weekends ~0-1h), E2 not SUD-eligible
        if is_weekend:
            h3 = round(float(np.clip(np.random.exponential(0.3), 0.0, 1.5)), 2)
        else:
            h3 = round(float(np.clip(np.random.normal(9.2, 0.6), 7.5, 11.5)), 2)
        cost3 = round(h3 * rates["e2-standard-4"], 4)
        records.append({
            "date": iso_date,
            "project_id": "prj-dev-sandbox",
            "resource_id": "wl-dev-microservices",
            "machine_type": "e2-standard-4",
            "region": "us-east1",
            "usage_hours": h3,
            "on_demand_cost": cost3,
            "existing_commitment": 0.0,
            "sud_eligible": False,  # E2 machine family is ineligible for SUD in GCP
            "history_days": num_days,
        })

        # 4. Bursty Analytics Worker: low baseline (2-4 hrs) with spikes (20-24 hrs) every 8-12 days
        is_spike = (day_idx % 11 == 0) or (day_idx in [29, 30, 59, 60, 88, 89])
        if is_spike:
            h4 = round(float(np.clip(np.random.normal(22.5, 1.0), 19.0, 24.0)), 2)
        else:
            h4 = round(float(np.clip(np.random.normal(3.2, 0.8), 1.0, 6.0)), 2)
        cost4 = round(h4 * rates["n1-standard-8"], 4)
        records.append({
            "date": iso_date,
            "project_id": "prj-data-analytics",
            "resource_id": "wl-batch-etl-spikes",
            "machine_type": "n1-standard-8",
            "region": "us-west1",
            "usage_hours": h4,
            "on_demand_cost": cost4,
            "existing_commitment": 0.0,
            "sud_eligible": True,
            "history_days": num_days,
        })

        # 5. Legacy Monolith: declining workload
        # Month 1 (days 0-29): ~23h/day
        # Month 2 (days 30-59): ~13h/day
        # Month 3 (days 60-89): ~4h/day
        if day_idx < 30:
            h5 = round(float(np.clip(np.random.normal(23.2, 0.6), 21.0, 24.0)), 2)
        elif day_idx < 60:
            h5 = round(float(np.clip(np.random.normal(13.4, 1.2), 9.0, 16.5)), 2)
        else:
            h5 = round(float(np.clip(np.random.normal(4.1, 1.0), 1.0, 7.5)), 2)
        cost5 = round(h5 * rates["n1-standard-4"], 4)
        records.append({
            "date": iso_date,
            "project_id": "prj-migration-phaseout",
            "resource_id": "wl-legacy-monolith",
            "machine_type": "n1-standard-4",
            "region": "us-central1",
            "usage_hours": h5,
            "on_demand_cost": cost5,
            "existing_commitment": 0.0,
            "sud_eligible": True,
            "history_days": num_days,
        })

        # 6. Pilot Inference Service: Short-history workload (only active in the last 18 days: days 72-89)
        if day_idx >= (num_days - 18):
            h6 = round(float(np.clip(np.random.normal(20.4, 1.2), 16.0, 24.0)), 2)
            cost6 = round(h6 * rates["n2-standard-4"], 4)
            records.append({
                "date": iso_date,
                "project_id": "prj-ai-innovation",
                "resource_id": "wl-pilot-inference",
                "machine_type": "n2-standard-4",
                "region": "us-central1",
                "usage_hours": h6,
                "on_demand_cost": cost6,
                "existing_commitment": 0.0,
                "sud_eligible": True,
                "history_days": 18,
            })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, "sample_usage.csv")
    df = generate_synthetic_usage_data()
    df.to_csv(csv_path, index=False)
    print(f"Generated synthetic dataset with {len(df)} records at {csv_path}")
    print("Workloads summary:")
    print(df.groupby("resource_id").agg(
        records=("date", "count"),
        avg_hours=("usage_hours", "mean"),
        total_cost=("on_demand_cost", "sum")
    ))
