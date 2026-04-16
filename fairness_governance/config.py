"""Global configuration helpers for the Fairness Governance System."""

from __future__ import annotations

from dataclasses import asdict, dataclass


FAIRNESS_METRICS = {
    "Demographic Parity": "demographic_parity",
    "Equal Opportunity": "equal_opportunity",
}


@dataclass
class FairnessCharter:
    target: str
    sensitive_attribute: str
    fairness_metric: str = "Demographic Parity"
    epsilon: float = 0.05

    def to_dict(self) -> dict:
        data = asdict(self)
        data["metric_key"] = FAIRNESS_METRICS.get(
            self.fairness_metric, "demographic_parity"
        )
        return data


GLOBAL_CONFIG: dict = {}


def set_global_config(charter: FairnessCharter) -> dict:
    """Persist the active fairness charter in process memory."""
    GLOBAL_CONFIG.clear()
    GLOBAL_CONFIG.update(charter.to_dict())
    return GLOBAL_CONFIG

