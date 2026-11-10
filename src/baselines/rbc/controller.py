#!/usr/bin/env python3
"""
Simple Rule-Based Control (RBC) for the Sinergym 5-zone environment.

"""

from __future__ import annotations

import re
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import pandas as pd
import sinergym  # noqa: F401  # Registers Sinergym environment IDs in Gymnasium.

from sinergym.utils.wrappers import CSVLogger, LoggerWrapper

# -----------------------------------------------------------------------------
# 1) Basic environment settings
# -----------------------------------------------------------------------------
ENV_ID = "Eplus-5zone-zonal-arizona-continuous-v1"

# User-facing experiment name requested.
EXP_NAME_RAW = "rbc_azz_summer"

# Keep folder names safe and readable: replace spaces/symbols with underscores.
EXP_NAME_SAFE = re.sub(r"[^A-Za-z0-9_-]+", "_", EXP_NAME_RAW).strip("_")
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"{EXP_NAME_SAFE}_{RUN_TS}"

# Runperiod is day-first format in this workspace:
# (start_day, start_month, start_year, end_day, end_month, end_year)
BUILDING_CONFIG = {
    "timesteps_per_hour": 4,            # 15 minutes per step
    "runperiod": (1, 6, 1991, 7, 6, 1991),  
}

# -----------------------------------------------------------------------------
# 2) Rule constants
# -----------------------------------------------------------------------------
SUMMER_MONTHS = {6, 7, 8, 9}

WINTER_HEAT = 20.0
WINTER_COOL = 23.5
SUMMER_HEAT = 23.0
SUMMER_COOL = 26.0

# Occupancy variables used to filter comfort metrics and report occupancy.
OCCUPANCY_VARS = [
    "air_occ_space1",
    "air_occ_space2",
    "air_occ_space3",
    "air_occ_space4",
    "air_occ_space5",
]

MONTH_VAR = "month"
TIMESTEP_HOURS = 0.25
ZONE_NAMES = [f"Zone {i}" for i in range(1, 6)]
TEMP_VARS = [f"air_temperature_space{i}" for i in range(1, 6)]


def print_run_metrics(observations_path: str) -> None:
    """Print whole-run HVAC energy and occupied-step PMV/comfort metrics."""
    try:
        from pythermalcomfort.models import pmv_ppd_iso
    except ImportError as exc:
        raise ImportError(
            "Custom PMV requires pythermalcomfort in the Sinergym container. "
            "Install it there with `pip install pythermalcomfort`."
        ) from exc

    data = pd.read_csv(observations_path)
    energy_col = "HVAC_electricity_demand_rate"
    required = [energy_col, MONTH_VAR, *OCCUPANCY_VARS, *TEMP_VARS]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise KeyError(f"Missing columns in {observations_path}: {missing}")

    energy_kwh = data[energy_col].sum() * TIMESTEP_HOURS / 1000.0
    print(f"\nTotal HVAC electrical energy (all steps in range): {energy_kwh:.4f} kWh")
    print("(from column 'HVAC_electricity_demand_rate', W × 0.25 h / 1000 per row)")

    summer = data[MONTH_VAR].isin(SUMMER_MONTHS)
    custom_pmv: dict[str, np.ndarray] = {}
    for i, temp_col in enumerate(TEMP_VARS, start=1):
        clo = np.where(summer, 0.5, 1.0)
        custom_pmv[f"pmv_custom_space{i}"] = np.asarray(pmv_ppd_iso(
            tdb=data[temp_col].to_numpy(), tr=data[temp_col].to_numpy(),
            vr=0.15, rh=40, met=1.1, clo=clo,
            model="7730-2005", limit_inputs=False, round_output=False,
        ).pmv, dtype=float)

    def summarize_custom_pmv() -> None:
        print("\nCustom ISO 7730 PMV (occupied steps only, per zone: air_occ_space* > 0):")
        print("  pmv_viol_%     : |PMV| outside [-0.5, 0.5]")
        print("  comfort_viol_%: air temp outside winter [20.0, 23.5] °C or summer [23.0, 26.0] °C")
        print(f"{'Zone':<8}{'n_steps':>8}{'mean_PMV':>11}{'std_PMV':>11}{'pmv_viol_%':>13}{'comfort_viol_%':>16}")
        all_values: list[np.ndarray] = []
        all_comfort: list[np.ndarray] = []
        total_n = 0
        for i, (zone, occ_col, temp_col) in enumerate(
            zip(ZONE_NAMES, OCCUPANCY_VARS, TEMP_VARS), start=1
        ):
            occupied = data[occ_col].to_numpy() > 0
            values = custom_pmv[f"pmv_custom_space{i}"][occupied]
            temps = data[temp_col].to_numpy(dtype=float)[occupied]
            months = data[MONTH_VAR].to_numpy()[occupied]
            limits = np.where(np.isin(months, list(SUMMER_MONTHS))[:, None],
                              np.array([23.0, 26.0]), np.array([20.0, 23.5]))
            comfort_bad = (temps < limits[:, 0]) | (temps > limits[:, 1])
            n = len(values)
            if n:
                print(f"{zone:<8}{n:>8d}{np.mean(values):>11.3f}{np.std(values):>11.3f}"
                      f"{100*np.mean(np.abs(values) > 0.5):>13.2f}{100*np.mean(comfort_bad):>16.2f}")
                all_values.append(values)
                all_comfort.append(comfort_bad)
                total_n += n
            else:
                print(f"{zone:<8}{0:>8d}{'nan':>11}{'nan':>11}{'nan':>13}{'nan':>16}")
        if total_n:
            values = np.concatenate(all_values)
            comfort_bad = np.concatenate(all_comfort)
            print(f"{'ALL':<8}{total_n:>8d}{np.mean(values):>11.3f}{np.std(values):>11.3f}"
                  f"{100*np.mean(np.abs(values) > 0.5):>13.2f}{100*np.mean(comfort_bad):>16.2f}")

    summarize_custom_pmv()


# -----------------------------------------------------------------------------
# 3) Helper functions
# -----------------------------------------------------------------------------
def build_obs_index_map(env: gym.Env) -> dict[str, int]:
    """Map observation variable name -> index in observation vector."""
    obs_names = list(getattr(env.unwrapped, "observation_variables", []))
    return {name: i for i, name in enumerate(obs_names)}


def setpoints_for_month(month: int) -> tuple[float, float]:
    """Return the seasonal heating and cooling setpoints."""
    if month in SUMMER_MONTHS:
        return SUMMER_HEAT, SUMMER_COOL
    return WINTER_HEAT, WINTER_COOL


def build_action(obs: np.ndarray, obs_idx: dict[str, int]) -> np.ndarray:
    """
    Build 10-D action vector:
    [H1, C1, H2, C2, H3, C3, H4, C4, H5, C5]
    """
    month = int(obs[obs_idx[MONTH_VAR]])

    action_values: list[float] = []
    for _ in ZONE_NAMES:
        heat_sp, cool_sp = setpoints_for_month(month)
        action_values.extend([heat_sp, cool_sp])

    return np.array(action_values, dtype=np.float32)


# -----------------------------------------------------------------------------
# 4) Main simulation
# -----------------------------------------------------------------------------
def main() -> None:
    print("Starting RBC run...")
    print("Environment ID:", ENV_ID)
    print("Requested experiment name:", EXP_NAME_RAW)
    print("Run name used for workspace:", RUN_NAME)
    print("Building config:", BUILDING_CONFIG)

    # Create base env with custom run name.
    env = gym.make(ENV_ID, env_name=RUN_NAME, building_config=BUILDING_CONFIG)

    # CSVLogger requires a BaseLoggerWrapper child class under it.
    # LoggerWrapper provides that requirement.
    env = LoggerWrapper(env)
    env = CSVLogger(env)

    workspace_path = None
    try:
        # Build variable index map once.
        obs_idx = build_obs_index_map(env)

        # Check required observation names before running.
        required = [MONTH_VAR, *OCCUPANCY_VARS]
        missing = [name for name in required if name not in obs_idx]
        if missing:
            raise KeyError(
                f"Missing required observation names: {missing}. "
                f"Available names: {list(obs_idx.keys())}"
            )

        # Reset starts the first episode.
        obs, info = env.reset()
        print("Reset complete. Info keys:", list(info.keys()))

        done = False
        step = 0

        while not done:
            # Build RBC action from current observation.
            action = build_action(obs, obs_idx)

            # Apply action in environment.
            obs, _, terminated, truncated, info = env.step(action)
            step += 1
            done = terminated or truncated

            # Print short progress for first few steps and final step.
            if step <= 5 or done:
                month = int(obs[obs_idx[MONTH_VAR]])
                occ_snapshot = [float(obs[obs_idx[v]]) for v in OCCUPANCY_VARS]
                print(
                    f"Step {step} | month={month} | occupancy={occ_snapshot} | "
                    f"action={action.tolist()} | term={terminated} trunc={truncated}"
                )

        print("Episode finished. Total steps:", step)
        print("Sinergym workspace path:", env.get_wrapper_attr("workspace_path"))
        workspace_path = env.get_wrapper_attr("workspace_path")
        print("CSVLogger files are saved automatically in that workspace.")

    finally:
        env.close()
        print("Environment closed.")

    if workspace_path:
        observations_path = os.path.join(
            workspace_path, "episode-1", "monitor", "observations.csv"
        )
        print_run_metrics(observations_path)


if __name__ == "__main__":
    main()
