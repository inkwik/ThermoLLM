#!/usr/bin/env python3
"""Evaluate a trained five-zone PPO policy on a deterministic 7-day window.

The default winter test is 1–7 January 1991 (inclusive). For the summer test,
pass ``--test-period summer`` to use 1–7 June 1991. Weather and policy actions
are deterministic by default. The evaluator discovers trainer-created
``*_best_model`` directories and loads their observation-normalization stats.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import site
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Prevent oversized thread pools when several policies are tested.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "common"))
LOCAL_SITE_PACKAGES = REPO_ROOT / ".container_site_packages"
if LOCAL_SITE_PACKAGES.is_dir():
    site.addsitedir(str(LOCAL_SITE_PACKAGES))

ENERGYPLUS_PYTHON_PATH = Path(
    os.environ.get("EPLUS_PATH", "/usr/local/EnergyPlus-25-1-0")
)
if (ENERGYPLUS_PYTHON_PATH / "pyenergyplus").is_dir():
    site.addsitedir(str(ENERGYPLUS_PYTHON_PATH))

# Sinergym fixes its output root during import. Set it first so temporary test
# workspaces stay on the /mnt-backed codes mount instead of Docker's overlay.
DEFAULT_TEST_WORKDIR = SCRIPT_DIR / "PPO_test_runs"
TEST_WORKDIR = Path(os.environ.get("SINERGYM_TEST_WORKDIR", DEFAULT_TEST_WORKDIR))
TEST_WORKDIR.mkdir(parents=True, exist_ok=True)
if TEST_WORKDIR.resolve() == DEFAULT_TEST_WORKDIR.resolve():
    os.chmod(TEST_WORKDIR, 0o2775)
os.chdir(TEST_WORKDIR)

DEFAULT_REPORT_DIR = SCRIPT_DIR / "PPO_test_reports"
REPORT_DIR = Path(os.environ.get("PPO_TEST_REPORT_DIR", DEFAULT_REPORT_DIR))
REPORT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_SUMMARY_DIR = SCRIPT_DIR
SUMMARY_DIR = Path(os.environ.get("PPO_TEST_SUMMARY_DIR", DEFAULT_SUMMARY_DIR))
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import sinergym  # noqa: F401  # Registers Sinergym environments.
from sinergym.utils.rewards import TruePMV
from sinergym.utils.wrappers import (
    CSVLogger,
    LoggerWrapper,
    NormalizeAction,
    NormalizeObservation,
)
from pmv_energy_metrics import (
    compute_performance_metrics,
    format_pmv_summary_report,
    runperiod_to_date_range,
    save_pmv_summary_artifacts,
)
from stable_baselines3 import PPO


# -----------------------------------------------------------------------------
# Easy settings: edit these values or override them with command-line options.
# -----------------------------------------------------------------------------
# Paste one complete ``*_best_model`` folder here, then run this script without
# model-selection arguments. Both host (/mnt/kb/codes/...) and container
# (/workspaces/proj/codes/...) paths are accepted. Leave empty to select by the
# settings below or with command-line options.
MODEL_BEST_FOLDER = ""

# Single-model selection defaults.
MODEL_ENERGY_KWH_MAX = 4.5
MODEL_PMV_MAX = 4.0

# Common scoring scale used for every tested policy. Each environment also uses
# the selected model's original training reward settings. The common score is
# calculated from raw PMV, occupancy, and HVAC energy for fair cross-model
# ranking; rewards on different training scales are not directly comparable.
COMMON_SCORE_ENERGY_KWH_MAX = 4.5
COMMON_SCORE_PMV_MAX = 4.0

DEFAULT_TEST_EPISODES = 1
DEFAULT_TEST_SEED_BASE = 20_042  # unseen relative to training eval seeds 10042+
DEFAULT_WEATHER = "deterministic"

ENVIRONMENT_IDS = {
    "chicago": {
        "deterministic": "Eplus-5zone-zonal-chicago-continuous-v1",
        "stochastic": "Eplus-5zone-zonal-chicago-continuous-stochastic-v1",
    },
    "ny": {
        "deterministic": "Eplus-5zone-zonal-mixed-continuous-v1",
        "stochastic": "Eplus-5zone-zonal-mixed-continuous-stochastic-v1",
    },
    "arizona": {
        "deterministic": "Eplus-5zone-zonal-arizona-continuous-v1",
        "stochastic": "Eplus-5zone-zonal-arizona-continuous-stochastic-v1",
    },
}
TIMESTEPS_PER_HOUR = 4
TEST_BUILDING_CONFIGS = {
    # Fixed 7-day seasonal evaluation windows in 1991, inclusive.
    # Winter is the default; pass --test-period summer for 1–7 June.
    "winter": {
        "timesteps_per_hour": TIMESTEPS_PER_HOUR,
        "runperiod": (1, 1, 1991, 7, 1, 1991),
    },
    "summer": {
        "timesteps_per_hour": TIMESTEPS_PER_HOUR,
        "runperiod": (1, 6, 1991, 7, 6, 1991),
    },
}
ZONE_COUNT = 5

BASE_REWARD_KWARGS: dict[str, Any] = {
    "pmv_variables": [f"new_pmv_space{index}" for index in range(1, ZONE_COUNT + 1)],
    "occupancy_variables": [
        f"air_occ_space{index}" for index in range(1, ZONE_COUNT + 1)
    ],
    "energy_variable": "total_electricity_HVAC",
    "pmv_min": 0.0,
    "energy_kwh_min": 0.0,
    "rho_occupied": 1.1,
    "rho_unoccupied": 0.1,
}

RUN_PATTERN = re.compile(
    r"^(?P<run_base>"
    r"ppo_(?P<location>chicago|ny|azz|arizona)_"
    r"(?P<training_period>[A-Za-z0-9]+)_"
    r"(?:training_)?TruePMV_stochastic_"
    r"(?:subproc(?P<n_envs>[0-9]+)_)?"
    r"energy(?P<energy>[0-9]+(?:\.[0-9]+)?)_"
    r"pmv(?P<pmv>[0-9]+(?:\.[0-9]+)?)_"
    r"(?P<episodes>[0-9]+)_"
    r"(?P<learning_rate>[0-9.eE+-]+)_"
    r"(?P<seed>[0-9]+)_"
    r"(?P<timestamp>.+)"
    r")_best_model$",
    re.IGNORECASE,
)
COMPACT_RUN_PATTERN = re.compile(
    r"^(?P<run_base>ppo_truepmv_"
    r"(?P<period>winter|summer)_"
    r"(?P<season>febapr|julsep)_"
    r"(?P<location>arizona|chicago|newyork)_"
    r"(?P<episodes>[0-9]+)ep_(?P<timestamp>[0-9]{8})"
    r")_best_model$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ArtifactSet:
    run_base_name: str
    location: str
    training_period: str
    energy_kwh_max: float
    pmv_max: float
    training_episodes: int
    learning_rate: float
    training_seed: int
    directory: Path
    model: Path
    mean: Path
    var: Path
    count: Path


def calculate_steps_per_episode(building_config: dict[str, Any]) -> int:
    start_day, start_month, start_year, end_day, end_month, end_year = (
        building_config["runperiod"]
    )
    start = datetime(start_year, start_month, start_day)
    end = datetime(end_year, end_month, end_day)
    return ((end - start).days + 1) * 24 * int(
        building_config["timesteps_per_hour"]
    )


TEST_STEPS_PER_EPISODE = {
    period: calculate_steps_per_episode(config)
    for period, config in TEST_BUILDING_CONFIGS.items()
}


def discover_artifacts() -> list[ArtifactSet]:
    """Find complete best-model/statistics sets written by the current trainer."""
    discovered: list[ArtifactSet] = []
    for directory in SCRIPT_DIR.glob("ppo_*_best_model"):
        try:
            discovered.append(artifact_from_directory(directory))
        except (FileNotFoundError, ValueError):
            continue
    return sorted(
        discovered,
        key=lambda item: (
            item.energy_kwh_max,
            item.pmv_max,
            item.training_episodes,
            item.directory.stat().st_mtime,
        ),
    )


def resolve_best_folder(value: str | Path) -> Path:
    """Resolve host or container spelling of a best-model directory."""
    raw_path = Path(value).expanduser()
    host_codes_root = Path("/mnt/kb/codes")
    container_codes_root = Path("/workspaces/proj/codes")
    try:
        relative = raw_path.relative_to(host_codes_root)
    except ValueError:
        pass
    else:
        raw_path = SCRIPT_DIR / relative
    try:
        relative = raw_path.relative_to(container_codes_root)
    except ValueError:
        pass
    else:
        raw_path = SCRIPT_DIR / relative
    if not raw_path.is_absolute():
        raw_path = SCRIPT_DIR / raw_path
    return raw_path.resolve()


def artifact_from_directory(value: str | Path) -> ArtifactSet:
    """Build and validate one artifact set from its *_best_model folder."""
    directory = resolve_best_folder(value)
    if not directory.is_dir():
        raise FileNotFoundError(f"Best-model folder not found: {directory}")
    match = RUN_PATTERN.fullmatch(directory.name)
    compact_match = COMPACT_RUN_PATTERN.fullmatch(directory.name)
    if match is None and compact_match is None:
        raise ValueError(
            "Best-model folder name is not recognized: "
            f"{directory.name!r}. Pass the complete trainer-created "
            "*_best_model folder without renaming it."
        )
    if match is not None:
        run_base_name = match.group("run_base")
        location = match.group("location").lower()
        training_period = match.group("training_period")
        energy_kwh_max = float(match.group("energy"))
        pmv_max = float(match.group("pmv"))
        training_episodes = int(match.group("episodes"))
        learning_rate = float(match.group("learning_rate"))
        training_seed = int(match.group("seed"))
    else:
        assert compact_match is not None
        run_base_name = compact_match.group("run_base")
        location = compact_match.group("location").lower()
        training_period = compact_match.group("period").lower()
        # The compact name is produced by the current 2026 multi-climate
        # training jobs, whose trainer settings are fixed at these values.
        energy_kwh_max = 4.5
        pmv_max = 4.0
        training_episodes = int(compact_match.group("episodes"))
        learning_rate = 3e-4
        training_seed = 42
    paths = {
        "model": directory / "best_model.zip",
        "mean": directory / "mean.txt",
        "var": directory / "var.txt",
        "count": directory / "count.txt",
    }
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete best-model folder {directory}: missing "
            + ", ".join(missing)
        )
    return ArtifactSet(
        run_base_name=run_base_name,
        location={
            "ny": "ny",
            "azz": "arizona",
            "arizona": "arizona",
            "chicago": "chicago",
            "newyork": "ny",
        }[location],
        training_period=training_period,
        energy_kwh_max=energy_kwh_max,
        pmv_max=pmv_max,
        training_episodes=training_episodes,
        learning_rate=learning_rate,
        training_seed=training_seed,
        directory=directory,
        model=paths["model"].resolve(),
        mean=paths["mean"].resolve(),
        var=paths["var"].resolve(),
        count=paths["count"].resolve(),
    )


def newest_longest_per_reward(
    artifacts: list[ArtifactSet],
) -> list[ArtifactSet]:
    """Choose the longest, then newest, complete run for each reward pair."""
    selected: dict[tuple[float, float], ArtifactSet] = {}
    for artifact in artifacts:
        key = (artifact.energy_kwh_max, artifact.pmv_max)
        previous = selected.get(key)
        if previous is None or (
            artifact.training_episodes,
            artifact.directory.stat().st_mtime,
        ) > (
            previous.training_episodes,
            previous.directory.stat().st_mtime,
        ):
            selected[key] = artifact
    return sorted(
        selected.values(),
        key=lambda item: (item.energy_kwh_max, item.pmv_max),
    )


def select_artifacts(
    artifacts: list[ArtifactSet], arguments: argparse.Namespace
) -> list[ArtifactSet]:
    if arguments.best_model_folder:
        return [artifact_from_directory(arguments.best_model_folder)]

    if arguments.run_base_name:
        selected = [
            item
            for item in artifacts
            if item.run_base_name == arguments.run_base_name
        ]
        if not selected:
            raise FileNotFoundError(
                f"No complete best-model set found for {arguments.run_base_name!r}."
            )
        return selected

    if arguments.all_models:
        return newest_longest_per_reward(artifacts)

    matches = [
        item
        for item in artifacts
        if math.isclose(
            item.energy_kwh_max, arguments.energy_max, abs_tol=1e-9
        )
        and math.isclose(item.pmv_max, arguments.pmv_max, abs_tol=1e-9)
    ]
    if not matches:
        raise FileNotFoundError(
            "No complete best-model set matches "
            f"energy_kwh_max={arguments.energy_max:g}, "
            f"pmv_max={arguments.pmv_max:g}. Use --list-models."
        )
    return [
        max(
            matches,
            key=lambda item: (
                item.training_episodes,
                item.directory.stat().st_mtime,
            ),
        )
    ]


def print_artifacts(artifacts: list[ArtifactSet]) -> None:
    if not artifacts:
        print("No complete current-format best models were found.")
        return
    print(
        "location  period       energy_max  pmv_max  episodes  learning_rate  seed  run_base_name"
    )
    for item in artifacts:
        print(
            f"{item.location:8s}  {item.training_period:11s}  "
            f"{item.energy_kwh_max:10g}  {item.pmv_max:7g}  "
            f"{item.training_episodes:8d}  {item.learning_rate:13g}  "
            f"{item.training_seed:4d}  {item.run_base_name}"
        )


def infer_test_period(
    artifact: ArtifactSet,
    requested_period: str,
) -> str:
    """Select the fixed January or June test week from the training label."""
    if requested_period != "auto":
        return requested_period
    label = artifact.training_period.lower()
    if any(token in label for token in ("jun", "jul", "summer", "sept")):
        return "summer"
    if any(token in label for token in ("jan", "winter", "april")):
        return "winter"
    raise ValueError(
        f"Cannot infer test period from {artifact.training_period!r}; "
        "pass --test-period winter or --test-period summer."
    )


def make_environment(
    *,
    environment_id: str,
    environment_name: str,
    mean_path: Path,
    var_path: Path,
    count_path: Path,
    training_energy_max: float,
    training_pmv_max: float,
    building_config: dict[str, Any],
) -> gym.Env:
    reward_kwargs = {
        **BASE_REWARD_KWARGS,
        "pmv_max": training_pmv_max,
        "energy_kwh_max": training_energy_max,
    }
    environment = gym.make(
        environment_id,
        env_name=environment_name,
        building_config=building_config,
        reward=TruePMV,
        reward_kwargs=reward_kwargs,
    )
    environment = NormalizeObservation(
        environment,
        automatic_update=False,
        mean=str(mean_path),
        var=str(var_path),
        count=str(count_path),
    )
    environment = NormalizeAction(environment)
    # Keep the same logger stack used by the RBC scripts. CSVLogger writes
    # episode-*/monitor/observations.csv, which compare_pmv_energy_runs.py uses.
    environment = LoggerWrapper(environment)
    environment = CSVLogger(environment, monitor=True)
    return environment


def calculate_truepmv_terms(
    *,
    zone_abs_pmv: list[float],
    zone_occupancy: list[float],
    hvac_energy_kwh: float,
    pmv_max: float,
    energy_kwh_max: float,
) -> tuple[float, float, float]:
    """Reproduce TruePMV terms for a requested reporting scale."""
    if pmv_max <= 0 or energy_kwh_max <= 0:
        raise ValueError("PMV and energy maximums must be greater than zero.")
    comfort_penalty = 0.0
    for abs_pmv, occupancy in zip(zone_abs_pmv, zone_occupancy):
        violation = max(abs_pmv - 0.5, 0.0)
        normalized_violation = min(max(violation / pmv_max, 0.0), 1.0)
        rho = (
            BASE_REWARD_KWARGS["rho_occupied"]
            if occupancy > 0
            else BASE_REWARD_KWARGS["rho_unoccupied"]
        )
        comfort_penalty += float(rho) * normalized_violation
    normalized_energy = min(
        max(hvac_energy_kwh / energy_kwh_max, 0.0), 1.0
    )
    comfort_term = -comfort_penalty
    energy_term = -normalized_energy
    return comfort_term + energy_term, comfort_term, energy_term


def observations_for_custom_pmv_metrics(obs_df: pd.DataFrame) -> pd.DataFrame:
    """Use the custom PMV observations in the shared PMV/energy report."""
    custom_columns = [
        f"new_pmv_space{index}" for index in range(1, ZONE_COUNT + 1)
    ]
    missing = [column for column in custom_columns if column not in obs_df]
    if missing:
        raise ValueError(
            "Cannot build the custom-PMV report; missing observation columns: "
            + ", ".join(missing)
        )

    metrics_df = obs_df.copy()
    for index, custom_column in enumerate(custom_columns, start=1):
        # The shared report expects pmv_spaceN columns. Replace those in this
        # metrics-only copy so its PMV rows use new_pmv_spaceN, while leaving
        # the original Sinergym observations CSV untouched.
        metrics_df[f"pmv_space{index}"] = pd.to_numeric(
            obs_df[custom_column], errors="coerce"
        )
    return metrics_df


def safe_remove_workspace(workspace_path: str | None) -> None:
    if not workspace_path:
        return
    root = TEST_WORKDIR.resolve()
    workspace = Path(workspace_path).resolve()
    try:
        inside_root = os.path.commonpath([root, workspace]) == str(root)
    except ValueError:
        inside_root = False
    if (
        inside_root
        and workspace != root
        and "PPO_TEST_" in workspace.name
    ):
        shutil.rmtree(workspace, ignore_errors=True)


def evaluate_worker(arguments: argparse.Namespace) -> None:
    """Evaluate exactly one model inside this isolated EnergyPlus process."""
    seeds = [int(value) for value in arguments.worker_seeds.split(",")]
    environment_id = ENVIRONMENT_IDS[arguments.worker_location][
        arguments.worker_weather
    ]
    building_config = TEST_BUILDING_CONFIGS[arguments.worker_test_period]
    environment = make_environment(
        environment_id=environment_id,
        environment_name=arguments.worker_env_name,
        mean_path=Path(arguments.worker_mean),
        var_path=Path(arguments.worker_var),
        count_path=Path(arguments.worker_count),
        training_energy_max=arguments.worker_training_energy_max,
        training_pmv_max=arguments.worker_training_pmv_max,
        building_config=building_config,
    )
    workspace_path = str(environment.get_wrapper_attr("workspace_path"))
    episode_rows: list[dict[str, float]] = []
    environment_closed = False

    try:
        model = PPO.load(arguments.worker_model, env=environment)
        for episode_number, seed in enumerate(seeds, start=1):
            observation, _ = environment.reset(seed=seed)
            terminated = False
            truncated = False
            training_episode_return = 0.0
            training_comfort_term_sum = 0.0
            training_energy_term_sum = 0.0
            common_episode_return = 0.0
            common_comfort_term_sum = 0.0
            common_energy_term_sum = 0.0
            total_energy_kwh = 0.0
            total_abs_pmv = 0.0
            step_count = 0
            any_zone_violation_steps = 0
            zone_samples = 0
            zone_violations = 0
            occupied_zone_samples = 0
            occupied_zone_violations = 0

            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = (
                    environment.step(action)
                )
                training_episode_return += float(reward)
                training_comfort_term_sum += float(info["comfort_term"])
                training_energy_term_sum += float(info["energy_term"])
                step_hvac_energy_kwh = float(info["total_hvac_energy_kwh"])
                total_energy_kwh += step_hvac_energy_kwh
                step_count += 1

                zone_abs_pmv = [
                    abs(float(info[f"zone_{index}_pmv"]))
                    for index in range(1, ZONE_COUNT + 1)
                ]
                zone_occupancy = [
                    float(info[f"zone_{index}_occupancy"])
                    for index in range(1, ZONE_COUNT + 1)
                ]
                common_reward, common_comfort, common_energy = (
                    calculate_truepmv_terms(
                        zone_abs_pmv=zone_abs_pmv,
                        zone_occupancy=zone_occupancy,
                        hvac_energy_kwh=step_hvac_energy_kwh,
                        pmv_max=arguments.worker_score_pmv_max,
                        energy_kwh_max=arguments.worker_score_energy_max,
                    )
                )
                common_episode_return += common_reward
                common_comfort_term_sum += common_comfort
                common_energy_term_sum += common_energy
                violation_flags = [value > 0.5 for value in zone_abs_pmv]
                total_abs_pmv += sum(zone_abs_pmv)
                any_zone_violation_steps += int(any(violation_flags))
                zone_samples += ZONE_COUNT
                zone_violations += sum(violation_flags)
                for occupancy, violation in zip(
                    zone_occupancy, violation_flags
                ):
                    if occupancy > 0:
                        occupied_zone_samples += 1
                        occupied_zone_violations += int(violation)

            episode_rows.append(
                {
                    "episode": float(episode_number),
                    "seed": float(seed),
                    # Unqualified reward fields use the shared comparison scale.
                    "episode_return": common_episode_return,
                    "episode_steps": float(step_count),
                    "mean_reward": common_episode_return / max(step_count, 1),
                    "mean_comfort_term": common_comfort_term_sum
                    / max(step_count, 1),
                    "mean_energy_term": common_energy_term_sum
                    / max(step_count, 1),
                    "training_scale_episode_return": training_episode_return,
                    "mean_training_scale_reward": training_episode_return
                    / max(step_count, 1),
                    "mean_training_scale_comfort_term":
                    training_comfort_term_sum / max(step_count, 1),
                    "mean_training_scale_energy_term":
                    training_energy_term_sum / max(step_count, 1),
                    "mean_hvac_kwh_per_step": total_energy_kwh
                    / max(step_count, 1),
                    "total_hvac_energy_kwh": total_energy_kwh,
                    "mean_abs_pmv": total_abs_pmv / max(zone_samples, 1),
                    "any_zone_pmv_violation_time_pct": 100.0
                    * any_zone_violation_steps
                    / max(step_count, 1),
                    "zone_pmv_violation_rate_pct": 100.0
                    * zone_violations
                    / max(zone_samples, 1),
                    "occupied_zone_pmv_violation_rate_pct": 100.0
                    * occupied_zone_violations
                    / max(occupied_zone_samples, 1),
                }
            )

        # CSVLogger finalizes observations.csv when the environment closes.
        environment.close()
        environment_closed = True
        observation_paths = sorted(
            Path(workspace_path).glob("episode-*/monitor/observations.csv")
        )
        if not observation_paths:
            fallback_path = Path(workspace_path) / "monitor" / "observations.csv"
            if fallback_path.is_file():
                observation_paths = [fallback_path]
        if not observation_paths:
            raise FileNotFoundError(
                "Sinergym did not produce monitor/observations.csv files in "
                f"{workspace_path}"
            )

        metric_frames = [
            observations_for_custom_pmv_metrics(pd.read_csv(path))
            for path in observation_paths
        ]
        test_observations = pd.concat(metric_frames, ignore_index=True)
        date_start, date_end = runperiod_to_date_range(
            tuple(building_config["runperiod"])
        )
        pmv_energy_metrics = compute_performance_metrics(
            test_observations, date_start, date_end
        )
        report_directory = REPORT_DIR
        report_stem = f"{arguments.worker_env_name}_custom_pmv_energy"
        report_csv_path = report_directory / f"{report_stem}.csv"
        report_txt_path = report_directory / f"{report_stem}.txt"
        report_label = (
            f"{arguments.worker_env_name} | custom PMV: "
            "new_pmv_space1..5"
        )
        save_pmv_summary_artifacts(
            pmv_energy_metrics,
            label=report_label,
            csv_path=str(report_csv_path),
            txt_path=str(report_txt_path),
            date_start=date_start,
            date_end=date_end,
        )
        pmv_energy_report = format_pmv_summary_report(
            pmv_energy_metrics,
            label=report_label,
            date_start=date_start,
            date_end=date_end,
        )

        numeric_keys = [
            key
            for key in episode_rows[0]
            if key not in {"episode", "seed"}
        ]
        summary: dict[str, Any] = {
            key: float(np.mean([row[key] for row in episode_rows]))
            for key in numeric_keys
        }
        summary["std_episode_return"] = float(
            np.std([row["episode_return"] for row in episode_rows])
        )
        summary["test_episodes"] = len(episode_rows)
        summary["test_seeds"] = ",".join(str(seed) for seed in seeds)
        summary["episode_details"] = episode_rows
        summary["result_folder"] = workspace_path
        summary["observations_csv"] = str(
            Path(workspace_path) / "episode-1" / "monitor" / "observations.csv"
        )
        summary["pmv_energy_summary_table"] = pmv_energy_metrics["summary_table"]
        summary["pmv_energy_summary_report"] = pmv_energy_report
        summary["pmv_energy_summary_csv"] = str(report_csv_path)
        summary["pmv_energy_summary_txt"] = str(report_txt_path)
        with open(arguments.worker_result, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, sort_keys=True)
    finally:
        try:
            if not environment_closed:
                environment.close()
        finally:
            if not arguments.worker_keep_workspace:
                safe_remove_workspace(workspace_path)


def run_isolated_test(
    artifact: ArtifactSet,
    arguments: argparse.Namespace,
    index: int,
) -> dict[str, Any]:
    test_period = infer_test_period(artifact, arguments.test_period)
    seeds = [
        arguments.seed_base + episode
        for episode in range(arguments.episodes)
    ]
    with tempfile.TemporaryDirectory(prefix="ppo_model_test_") as directory:
        result_path = Path(directory) / "result.json"
        log_path = Path(directory) / "worker.log"
        environment_name = (
            f"PPO_TEST_{artifact.location.upper()}_{test_period.upper()}_"
            f"E{artifact.energy_kwh_max:g}_"
            f"P{artifact.pmv_max:g}_{index}_{os.getpid()}"
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--worker-model",
            str(artifact.model),
            "--worker-mean",
            str(artifact.mean),
            "--worker-var",
            str(artifact.var),
            "--worker-count",
            str(artifact.count),
            "--worker-result",
            str(result_path),
            "--worker-seeds",
            ",".join(str(seed) for seed in seeds),
            "--worker-weather",
            arguments.weather,
            "--worker-location",
            artifact.location,
            "--worker-test-period",
            test_period,
            "--worker-training-energy-max",
            str(artifact.energy_kwh_max),
            "--worker-training-pmv-max",
            str(artifact.pmv_max),
            "--worker-score-energy-max",
            str(arguments.score_energy_max),
            "--worker-score-pmv-max",
            str(arguments.score_pmv_max),
            "--worker-env-name",
            environment_name,
        ]
        if arguments.keep_workspaces:
            command.append("--worker-keep-workspace")
        with open(log_path, "w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0 or not result_path.is_file():
            log_tail = log_path.read_text(
                encoding="utf-8", errors="replace"
            )[-4000:]
            raise RuntimeError(
                f"Testing failed for {artifact.run_base_name} "
                f"(exit {completed.returncode}).\n{log_tail}"
            )
        with open(result_path, "r", encoding="utf-8") as file:
            result = json.load(file)

    result.update(
        {
            "run_base_name": artifact.run_base_name,
            "location": artifact.location,
            "training_period": artifact.training_period,
            "test_period": test_period,
            "test_runperiod": str(
                TEST_BUILDING_CONFIGS[test_period]["runperiod"]
            ),
            "training_energy_kwh_max": artifact.energy_kwh_max,
            "training_pmv_max": artifact.pmv_max,
            "training_episodes": artifact.training_episodes,
            "training_learning_rate": artifact.learning_rate,
            "training_seed": artifact.training_seed,
            "model_path": str(artifact.model),
            "norm_mean_path": str(artifact.mean),
            "norm_var_path": str(artifact.var),
            "norm_count_path": str(artifact.count),
            "test_weather": arguments.weather,
            "deterministic_actions": True,
            "score_energy_kwh_max": arguments.score_energy_max,
            "score_pmv_max": arguments.score_pmv_max,
        }
    )
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test one or all best PPO models saved by the current trainer."
    )
    parser.add_argument(
        "--best-model-folder",
        default=MODEL_BEST_FOLDER or None,
        help=(
            "Complete trainer-created *_best_model folder. Overrides model "
            "selection by run name or reward values."
        ),
    )
    parser.add_argument(
        "--energy-max",
        type=float,
        default=MODEL_ENERGY_KWH_MAX,
        help="Training energy_kwh_max used to select one model.",
    )
    parser.add_argument(
        "--pmv-max",
        type=float,
        default=MODEL_PMV_MAX,
        help="Training pmv_max used to select one model.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Test the longest/newest complete run for every energy/PMV pair.",
    )
    parser.add_argument(
        "--run-base-name",
        default=None,
        help="Test one exact training run instead of selecting by reward values.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List discovered complete best-model sets and exit.",
    )
    parser.add_argument(
        "--score-energy-max",
        type=float,
        default=COMMON_SCORE_ENERGY_KWH_MAX,
        help="Common test reward energy scale used for every selected model.",
    )
    parser.add_argument(
        "--score-pmv-max",
        type=float,
        default=COMMON_SCORE_PMV_MAX,
        help="Common test reward PMV scale used for every selected model.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_TEST_EPISODES,
        help="Number of fixed-seed episodes per model.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=DEFAULT_TEST_SEED_BASE,
        help="First test seed; all models receive the same seed sequence.",
    )
    parser.add_argument(
        "--weather",
        choices=("deterministic", "stochastic"),
        default=DEFAULT_WEATHER,
        help="Fixed weather for comparison or stochastic weather for robustness.",
    )
    parser.add_argument(
        "--test-period",
        choices=("auto", "winter", "summer"),
        default="winter",
        help=(
            "Seasonal 7-day test window: winter (Jan 1-7, default), summer "
            "(Jun 1-7), or auto (infer from model label)."
        ),
    )
    retention_group = parser.add_mutually_exclusive_group()
    retention_group.add_argument(
        "--keep-workspaces",
        dest="keep_workspaces",
        action="store_true",
        default=True,
        help="Keep compatible Sinergym -res folders (default).",
    )
    retention_group.add_argument(
        "--delete-workspaces",
        dest="keep_workspaces",
        action="store_false",
        help="Delete detailed Sinergym/EnergyPlus result folders after testing.",
    )

    # Internal isolated-worker arguments.
    hidden = argparse.SUPPRESS
    parser.add_argument("--worker", action="store_true", help=hidden)
    parser.add_argument("--worker-model", help=hidden)
    parser.add_argument("--worker-mean", help=hidden)
    parser.add_argument("--worker-var", help=hidden)
    parser.add_argument("--worker-count", help=hidden)
    parser.add_argument("--worker-result", help=hidden)
    parser.add_argument("--worker-seeds", help=hidden)
    parser.add_argument(
        "--worker-weather",
        choices=("deterministic", "stochastic"),
        help=hidden,
    )
    parser.add_argument(
        "--worker-location",
        choices=tuple(ENVIRONMENT_IDS),
        help=hidden,
    )
    parser.add_argument(
        "--worker-test-period",
        choices=tuple(TEST_BUILDING_CONFIGS),
        help=hidden,
    )
    parser.add_argument("--worker-training-energy-max", type=float, help=hidden)
    parser.add_argument("--worker-training-pmv-max", type=float, help=hidden)
    parser.add_argument("--worker-score-energy-max", type=float, help=hidden)
    parser.add_argument("--worker-score-pmv-max", type=float, help=hidden)
    parser.add_argument("--worker-env-name", help=hidden)
    parser.add_argument(
        "--worker-keep-workspace", action="store_true", help=hidden
    )
    arguments = parser.parse_args()

    if arguments.worker:
        required = (
            "worker_model",
            "worker_mean",
            "worker_var",
            "worker_count",
            "worker_result",
            "worker_seeds",
            "worker_weather",
            "worker_location",
            "worker_test_period",
            "worker_training_energy_max",
            "worker_training_pmv_max",
            "worker_score_energy_max",
            "worker_score_pmv_max",
            "worker_env_name",
        )
        missing = [name for name in required if getattr(arguments, name) is None]
        if missing:
            parser.error("Worker mode missing: " + ", ".join(missing))
    elif arguments.episodes < 1:
        parser.error("--episodes must be at least 1.")
    elif arguments.score_energy_max <= 0 or arguments.score_pmv_max <= 0:
        parser.error("--score-energy-max and --score-pmv-max must be positive.")
    return arguments


def main(arguments: argparse.Namespace) -> None:
    artifacts = discover_artifacts()
    if arguments.list_models:
        print_artifacts(artifacts)
        return
    if not artifacts and not arguments.best_model_folder:
        raise FileNotFoundError(
            f"No complete current-format best models found in {SCRIPT_DIR}."
        )

    selected = select_artifacts(artifacts, arguments)
    print("Selected best models:")
    print_artifacts(selected)
    print("\nTest setup:")
    print("  weather:", arguments.weather)
    print("  locations:", sorted({item.location for item in selected}))
    print("  deterministic actions: True")
    print("  episodes per model:", arguments.episodes)
    print(
        "  fixed seeds:",
        [arguments.seed_base + index for index in range(arguments.episodes)],
    )
    print("  common score energy_kwh_max:", arguments.score_energy_max)
    print("  common score pmv_max:", arguments.score_pmv_max)
    print("  environment reward: each model's original training settings")
    print("  requested test period:", arguments.test_period)
    for artifact in selected:
        test_period = infer_test_period(artifact, arguments.test_period)
        print(
            f"  {artifact.location}/{artifact.training_period}: "
            f"{TEST_BUILDING_CONFIGS[test_period]['runperiod']} "
            f"({TEST_STEPS_PER_EPISODE[test_period]} steps)"
        )

    results: list[dict[str, Any]] = []
    for index, artifact in enumerate(selected, start=1):
        print(
            f"\n[{index}/{len(selected)}] Testing "
            f"energy={artifact.energy_kwh_max:g}, "
            f"pmv={artifact.pmv_max:g}, "
            f"episodes={artifact.training_episodes}...",
            flush=True,
        )
        result = run_isolated_test(artifact, arguments, index)
        results.append(result)
        print(
            f"  mean_reward={result['mean_reward']:.6f}, "
            f"comfort={result['mean_comfort_term']:.6f}, "
            f"energy={result['mean_energy_term']:.6f}, "
            f"training-scale reward="
            f"{result['mean_training_scale_reward']:.6f}, "
            f"HVAC={result['total_hvac_energy_kwh']:.2f} kWh, "
            f"occupied PMV violation="
            f"{result['occupied_zone_pmv_violation_rate_pct']:.2f}%"
        )
        print(f"  Sinergym result folder: {result['result_folder']}")
        print(f"  observations.csv: {result['observations_csv']}")
        print("\n" + result["pmv_energy_summary_report"])
        print(
            "[PMV + energy table CSV] "
            f"{result['pmv_energy_summary_csv']}"
        )
        print(
            "[PMV + energy report TXT] "
            f"{result['pmv_energy_summary_txt']}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    summary_path = SUMMARY_DIR / f"ppo_best_models_test_{timestamp}.csv"
    summary_rows = [
        {
            key: value
            for key, value in result.items()
            if key
            not in {
                "episode_details",
                "pmv_energy_summary_table",
                "pmv_energy_summary_report",
            }
        }
        for result in results
    ]
    comparison = pd.DataFrame(summary_rows).sort_values(
        "mean_reward", ascending=False
    )
    comparison.to_csv(summary_path, index=False)

    print("\nCommon-scale ranking (higher mean reward is better):")
    print(
        comparison[
            [
                "training_energy_kwh_max",
                "training_pmv_max",
                "training_episodes",
                "mean_reward",
                "mean_comfort_term",
                "mean_energy_term",
                "mean_training_scale_reward",
                "total_hvac_energy_kwh",
                "occupied_zone_pmv_violation_rate_pct",
            ]
        ].to_string(index=False)
    )
    print("\nComparison CSV saved:", summary_path)
    print("\nDetailed Sinergym result folders:")
    for result in results:
        print(" ", result["result_folder"])
    print(
        "Use physical HVAC energy and PMV violation metrics together with "
        "the common-scale reward when selecting the final policy."
    )


if __name__ == "__main__":
    args = parse_arguments()
    if args.worker:
        evaluate_worker(args)
    else:
        main(args)
