"""Synthetic campaign data generator for EconoCausal.

Generates a realistic observational dataset with confounding and heterogeneous
treatment effects. Saves CSV to the provided path.

Provides:
- generate_sample_campaign(output_path, n_customers=5000, random_seed=None)
- CLI entrypoint when executed directly
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "customer_id",
    "age",
    "income",
    "previous_purchases",
    "website_visits",
    "customer_tenure",
    "previous_spend",
    "engagement_score",
    "treatment",
    "discount_amount",
    "purchase",
    "revenue",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_sample_campaign(
    output_path: str = "data/sample_campaign_data.csv",
    n_customers: int = 5000,
    random_seed: int | None = 42,
) -> str:
    """Generate a synthetic campaign dataset and write to output_path.

    The generator intentionally creates confounding: higher-engagement
    customers are more likely to receive discounts and more likely to
    purchase. Heterogeneous treatment effects are created by sampling a
    sensitivity parameter per customer and mixing persuadable/always/never types.
    """
    if n_customers < 1:
        raise ValueError("n_customers must be at least 1")

    rng = np.random.default_rng(random_seed)
    age = np.clip(rng.normal(38, 12, n_customers).round(), 18, 90).astype(int)
    income = np.maximum(
        12_000,
        rng.lognormal(np.log(45_000), 0.35, n_customers).round(),
    ).astype(int)
    previous_purchases = rng.poisson(1.5, n_customers)
    website_visits = np.maximum(0, rng.normal(8, 6, n_customers).round()).astype(int)
    customer_tenure = np.maximum(1, rng.exponential(24, n_customers).round()).astype(int)
    previous_spend = np.maximum(
        0,
        previous_purchases * rng.uniform(20, 120, n_customers) + income * 0.02,
    ).round(2)
    engagement_score = np.clip(
        website_visits * 3
        + previous_purchases * 8
        + rng.normal(0, 10, n_customers),
        0,
        100,
    ).round(2)

    # Engagement and purchase history affect treatment assignment: this is the
    # intentional observational confounding in the campaign.
    logit_promo_10 = (
        -2.0 + 0.03 * engagement_score + 0.01 * previous_purchases - 0.00001 * income
    )
    logit_promo_20 = (
        -3.0 + 0.04 * engagement_score + 0.015 * previous_purchases - 0.000015 * income
    )

    p10 = _sigmoid(logit_promo_10)
    p20 = _sigmoid(logit_promo_20)
    assignment = rng.random(n_customers)
    treatment = np.select(
        [assignment < p20, assignment < p20 + p10],
        [20, 10],
        default=0,
    ).astype(int)
    discount_amount = treatment.astype(float)

    baseline_logit = (
        -3.0
        + 0.04 * engagement_score
        + 0.02 * previous_purchases
        + 0.001 * (income / 1000)
        + 0.01 * np.minimum(customer_tenure, 36)
    )
    baseline_prob = _sigmoid(baseline_logit)

    # Latent customer segments create heterogeneous effects. The segment is
    # intentionally not emitted; it represents unobserved response behavior.
    segment = rng.random(n_customers)
    sensitivity = np.select(
        [segment < 0.30, segment < 0.70],
        [rng.uniform(0.12, 0.35, n_customers), rng.uniform(0.02, 0.12, n_customers)],
        default=rng.uniform(-0.02, 0.05, n_customers),
    )
    purchase_probability = np.minimum(
        0.99,
        baseline_prob + sensitivity * np.select([treatment == 10, treatment == 20], [1.0, 1.6], default=0.0),
    )
    purchase = (rng.random(n_customers) < purchase_probability).astype(int)

    base_price = np.maximum(
        5,
        np.where(
            previous_purchases > 0,
            previous_spend / np.maximum(previous_purchases, 1),
            rng.uniform(20, 80, n_customers),
        )
        + rng.normal(0, 8, n_customers),
    )
    revenue = np.where(
        purchase == 1,
        np.maximum(0, base_price - discount_amount).round(2),
        0,
    )
    frame = pd.DataFrame(
        {
            "customer_id": [f"CUST{i:06d}" for i in range(1, n_customers + 1)],
            "age": age,
            "income": income,
            "previous_purchases": previous_purchases,
            "website_visits": website_visits,
            "customer_tenure": customer_tenure,
            "previous_spend": previous_spend,
            "engagement_score": engagement_score,
            "treatment": treatment,
            "discount_amount": discount_amount,
            "purchase": purchase,
            "revenue": revenue,
        },
        columns=REQUIRED_COLUMNS,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)

    return str(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic EconoCausal campaign data")
    parser.add_argument("--out", default="data/sample_campaign_data.csv")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = generate_sample_campaign(args.out, n_customers=args.n, random_seed=args.seed)
    print(f"Wrote sample campaign data to {out}")
