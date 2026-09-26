"""Default, configurable thresholds for customer response segmentation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentationConfig:
    """Business-neutral thresholds for interpreting model estimates.

    Values are probabilities/effects on the purchase-probability scale. Supply
    a new instance to ``segment_customers`` to tune them for a campaign.
    """

    high_baseline_probability: float = 0.70
    low_baseline_probability: float = 0.30
    meaningful_positive_ite: float = 0.05
    small_absolute_ite: float = 0.05
    negative_ite: float = -0.01

