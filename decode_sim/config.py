from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "population": {"borrowers": 1000, "loans_per_borrower": [1, 1]},
    "panel": {"months": 12},
    "demographics": {
        "age_mean": 42,
        "age_std": 13,
        "age_min": 18,
        "age_max": 78,
        "gender_weights": {
            "female": 0.49,
            "male": 0.49,
            "non_binary": 0.01,
            "not_informed": 0.01,
        },
    },
    "loan": {
        "principal_min": 800.0,
        "principal_max": 60000.0,
        "initial_dpd_mean": 55,
        "initial_dpd_std": 38,
    },
    "actions": {
        "channel_weights": {
            "none": 0.20,
            "sms": 0.25,
            "whatsapp": 0.25,
            "phone": 0.20,
            "field_visit": 0.10,
        },
        "max_discount": 0.55,
        "intensity_confounds_with_friction": True,
        "channel_costs": {},
        "human_channels": ["phone", "field_visit"],
    },
    "oracle": {
        "target_monthly_recovery_rate": 0.08,
        "offer_sensitivity": 3.0,
        "offer_effect": 1.1,
        "dpd_curve": {"shape": "linear"},
        "dpd_effect": -0.018,
        "liquidacao_age_effect": -0.050,
        "loan_age_effect": -0.012,
        "hidden_confounding_strength": 0.75,
        "noise_std": 0.25,
        "chargeoff_after_liquidacao_days": 360,
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    cfg = _deep_update(DEFAULT_CONFIG, data)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    borrowers = int(cfg["population"]["borrowers"])
    months = int(cfg["panel"]["months"])
    if borrowers <= 0:
        raise ValueError("population.borrowers must be positive")
    if months <= 0:
        raise ValueError("panel.months must be positive")
    if not 0.0 < float(cfg["oracle"]["target_monthly_recovery_rate"]) < 1.0:
        raise ValueError("oracle.target_monthly_recovery_rate must be in (0, 1)")
    if float(cfg["actions"]["max_discount"]) < 0.0:
        raise ValueError("actions.max_discount must be non-negative")
