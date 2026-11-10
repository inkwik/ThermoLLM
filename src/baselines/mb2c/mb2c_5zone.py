#!/usr/bin/env python3
"""Shared observation schema and feature helpers for the five-zone model."""
from __future__ import annotations

import math

import numpy as np

TIMESTEPS_PER_HOUR = 4
ZONE = range(1, 6)
BUILDING_VARS = (
    [f"air_temperature_space{i}" for i in ZONE]
    + [f"air_humidity_space{i}" for i in ZONE]
    + [f"new_pmv_space{i}" for i in ZONE]
    + ["total_electricity_HVAC"]
)
WEATHER_VARS = [
    "outdoor_temperature", "outdoor_humidity", "wind_speed", "wind_direction",
    "diffuse_solar_radiation", "direct_solar_radiation",
]
OCC_VARS = [f"air_occ_space{i}" for i in ZONE]
ACTION_LOW = np.array([12.0, 23.25] * 5, dtype=np.float32)
ACTION_HIGH = np.array([23.25, 30.0] * 5, dtype=np.float32)


def exogenous(obs: dict[str, float]) -> np.ndarray:
    """Return weather, occupancy, and cyclical calendar context (17 values)."""
    month, day, hour = int(obs["month"]), int(obs["day_of_month"]), float(obs["hour"])
    doy = (month - 1) * 30 + day
    cyclical = [math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12),
                math.sin(2 * math.pi * doy / 365.25), math.cos(2 * math.pi * doy / 365.25),
                math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)]
    return np.asarray([obs[k] for k in WEATHER_VARS + OCC_VARS] + cyclical, dtype=np.float32)


def as_dict(obs: np.ndarray, keys: list[str]) -> dict[str, float]:
    return {key: float(value) for key, value in zip(keys, obs)}
