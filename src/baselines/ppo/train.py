#!/usr/bin/env python3
"""Train the five-zone PPO controller with stochastic Sinergym weather.

The winter run covers 1 February–30 April 1991 (inclusive). Training
uses a stochastic environment; periodic checkpoint evaluation uses the
deterministic environment over the same run period. For a summer run, set
``RUNPERIOD=1,7,1991,30,9,1991`` and ``RUN_LABEL=SummerJulSep``. Select
Arizona, Chicago, or New York with ``ENV_ID`` and ``LOCATION_LABEL``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import shutil
import signal
import site
import subprocess
import sys
import tempfile
from datetime import datetime
from typing import Any, Dict, List

# Prevent CPU oversubscription in the PPO parent and EnergyPlus workers.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
LOCAL_SITE_PACKAGES = os.path.join(REPO_ROOT, ".container_site_packages")
if os.path.isdir(LOCAL_SITE_PACKAGES):
    site.addsitedir(LOCAL_SITE_PACKAGES)

ENERGYPLUS_PATH = os.environ.get("EPLUS_PATH", "/usr/local/EnergyPlus-25-1-0")
if os.path.isdir(os.path.join(ENERGYPLUS_PATH, "pyenergyplus")):
    site.addsitedir(ENERGYPLUS_PATH)

# Sinergym fixes its output root at import time.
DEFAULT_SINERGYM_WORKDIR = os.path.join(SCRIPT_DIR, "sinergym_runs")
SINERGYM_WORKDIR = os.environ.get(
    "SINERGYM_WORKDIR",
    DEFAULT_SINERGYM_WORKDIR,
)
os.makedirs(SINERGYM_WORKDIR, exist_ok=True)
os.chdir(SINERGYM_WORKDIR)

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import sinergym  # noqa: F401
from sinergym.utils.rewards import TruePMV
from sinergym.utils.wrappers import (
    LoggerWrapper,
    NormalizeAction,
    NormalizeObservation,
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize


# =============================================================================
# 1) EDITABLE EXPERIMENT CONFIGURATION
# =============================================================================

# Weather/environment:
#   Chicago: Eplus-5zone-zonal-chicago-continuous-stochastic-v1
#   Arizona: Eplus-5zone-zonal-arizona-continuous-stochastic-v1
#   New York: Eplus-5zone-zonal-mixed-continuous-stochastic-v1
#   Port Angeles: Eplus-5zone-zonal-cool-continuous-stochastic-v1
ENV_ID = os.environ.get(
    "ENV_ID", "Eplus-5zone-zonal-arizona-continuous-stochastic-v1"
)
EVAL_ENV_ID = os.environ.get(
    "EVAL_ENV_ID", "Eplus-5zone-zonal-arizona-continuous-v1"
)
LOCATION_LABEL = os.environ.get("LOCATION_LABEL", "AZZ")
RUN_LABEL = os.environ.get("RUN_LABEL", "WinterFebApr")

# Parallel collection. Six workers is the practical default when three training
# jobs run concurrently. Override with N_ENVS=8 at launch if desired.
N_ENVS = int(os.environ.get("N_ENVS", "6"))
SUBPROC_START_METHOD = os.environ.get("SUBPROC_START_METHOD", "spawn")

# Simulation clock and seasonal training period (start/end day, month, year).
# Default: winter, 1 February–30 April 1991 inclusive (three months).
# For summer, override with RUNPERIOD=1,7,1991,30,9,1991 and
# RUN_LABEL=SummerJulSep. Each seasonal experiment uses a stochastic ENV_ID.
TIMESTEPS_PER_HOUR = 4
BUILDING_CONFIG = {
    "timesteps_per_hour": TIMESTEPS_PER_HOUR,
    "runperiod": tuple(
        int(value)
        for value in os.environ.get(
            "RUNPERIOD", "1,2,1991,30,4,1991"
        ).split(",")
    ),
}
if len(BUILDING_CONFIG["runperiod"]) != 6:
    raise ValueError("RUNPERIOD must contain six comma-separated integers")
EVAL_BUILDING_CONFIG = dict(BUILDING_CONFIG)

# TruePMV reward using custom PMV observations.
REWARD_KWARGS = {
    "pmv_variables": [
        "new_pmv_space1",
        "new_pmv_space2",
        "new_pmv_space3",
        "new_pmv_space4",
        "new_pmv_space5",
    ],
    "occupancy_variables": [
        "air_occ_space1",
        "air_occ_space2",
        "air_occ_space3",
        "air_occ_space4",
        "air_occ_space5",
    ],
    "energy_variable": "total_electricity_HVAC",
    "pmv_min": 0.0,
    "pmv_max": 4.0,
    "energy_kwh_min": 0.0,
    "energy_kwh_max": 4.5,
    "rho_occupied": 1.1,
    "rho_unoccupied": 0.1,
}

TOTAL_EPISODES = int(os.environ.get("TOTAL_EPISODES", "1500"))
LEARNING_RATE = 3e-4
SEED = 42

# Preserve the original combined PPO rollout size. With six workers, each
# worker contributes 1024 steps: 6 * 1024 = 6144.
COMBINED_ROLLOUT_STEPS = 6144
BATCH_SIZE = 512

# Evaluation remains isolated from the EnergyPlus training processes.
EVAL_EVERY_EPISODES = 200
# Fixed weather + deterministic actions produce the same trajectory for every
# seed, so one episode is sufficient. Set EVAL_EPISODES=3 to restore the old
# duplicate evaluation behavior.
EVAL_EPISODES = int(os.environ.get("EVAL_EPISODES", "1"))
ALL_EVAL_SEEDS = tuple(SEED + 10_000 + index for index in range(3))

# W&B receives streaming episode summaries (reward, comfort, energy, demand and
# violation statistics), complete evaluation summaries, and best-model scores.
# Streaming aggregation avoids retaining full 8736-step episodes in memory.
ENABLE_WANDB = os.environ.get("ENABLE_WANDB", "0").strip().lower() in ("1", "true", "yes")
WANDB_AVAILABLE = importlib.util.find_spec("wandb") is not None
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "thermollm-reproduction")
WANDB_GROUP = os.environ.get("WANDB_GROUP", "")

ROLLING_WINDOW_EPISODES = min(100, TOTAL_EPISODES)


def calculate_steps_per_episode(building_config: dict) -> int:
    start_day, start_month, start_year, end_day, end_month, end_year = (
        building_config["runperiod"]
    )
    start_dt = datetime(start_year, start_month, start_day)
    end_dt = datetime(end_year, end_month, end_day)
    days = (end_dt - start_dt).days + 1
    return days * 24 * int(building_config["timesteps_per_hour"])


STEPS_PER_EPISODE = calculate_steps_per_episode(BUILDING_CONFIG)
TOTAL_TIMESTEPS = TOTAL_EPISODES * STEPS_PER_EPISODE

if N_ENVS < 2:
    raise ValueError("N_ENVS must be at least 2 for SubprocVecEnv training.")
if COMBINED_ROLLOUT_STEPS % N_ENVS != 0:
    raise ValueError(
        f"N_ENVS={N_ENVS} must divide COMBINED_ROLLOUT_STEPS="
        f"{COMBINED_ROLLOUT_STEPS}. Use 2, 3, 4, 6, 8, or 12."
    )
if EVAL_EPISODES < 1 or EVAL_EPISODES > len(ALL_EVAL_SEEDS):
    raise ValueError(
        f"EVAL_EPISODES must be between 1 and {len(ALL_EVAL_SEEDS)}."
    )

N_STEPS_PER_ENV = COMBINED_ROLLOUT_STEPS // N_ENVS
EVAL_SEEDS = ALL_EVAL_SEEDS[:EVAL_EPISODES]

PPO_PARAMS: Dict[str, object] = {
    "policy": "MlpPolicy",
    "learning_rate": LEARNING_RATE,
    "n_steps": N_STEPS_PER_ENV,
    "batch_size": BATCH_SIZE,
    "n_epochs": 5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.1,
    "target_kl": 0.015,
    "ent_coef": 1e-3,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "net_arch": [200, 200],
        "activation_fn": nn.ReLU,
    },
    "verbose": 1,
    "seed": SEED,
}


# =============================================================================
# 2) OUTPUT PATHS
# =============================================================================

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
RUN_BASE_NAME = os.environ.get(
    "RUN_BASE_NAME",
    f"ppo_{LOCATION_LABEL}_{RUN_LABEL}_TruePMV_stochastic_subproc{N_ENVS}_"
    f"energy{REWARD_KWARGS['energy_kwh_max']:g}_"
    f"pmv{REWARD_KWARGS['pmv_max']:g}_"
    f"{TOTAL_EPISODES}_{LEARNING_RATE:g}_{SEED}_{RUN_TS}",
)
ENV_NAME = RUN_BASE_NAME
OUTPUT_DIR = os.path.abspath(os.environ.get("PPO_OUTPUT_DIR", SCRIPT_DIR))
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, f"{RUN_BASE_NAME}.zip")
REWARD_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    f"{RUN_BASE_NAME}_episode_rewards.csv",
)
PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    f"{RUN_BASE_NAME}_reward_convergence.png",
)
NORM_MEAN_PATH = os.path.join(OUTPUT_DIR, f"{RUN_BASE_NAME}_norm_mean.txt")
NORM_VAR_PATH = os.path.join(OUTPUT_DIR, f"{RUN_BASE_NAME}_norm_var.txt")
NORM_COUNT_PATH = os.path.join(OUTPUT_DIR, f"{RUN_BASE_NAME}_norm_count.txt")

BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, f"{RUN_BASE_NAME}_best_model")
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "best_model.zip")
BEST_NORM_MEAN_PATH = os.path.join(BEST_MODEL_DIR, "mean.txt")
BEST_NORM_VAR_PATH = os.path.join(BEST_MODEL_DIR, "var.txt")
BEST_NORM_COUNT_PATH = os.path.join(BEST_MODEL_DIR, "count.txt")
EVALUATION_METRICS_PATH = os.path.join(
    BEST_MODEL_DIR,
    "evaluation_metrics.csv",
)
EVALUATION_LOG_PATH = os.path.join(
    BEST_MODEL_DIR,
    "evaluation_subprocess.log",
)


def sanitize_env_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


def _atomic_copy(source: str, destination: str) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".best_artifact_",
        dir=os.path.dirname(destination),
    )
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


# =============================================================================
# 3) TRAINING ENVIRONMENTS AND CALLBACKS
# =============================================================================

def make_worker(worker_rank: int, run_name: str):
    """Return a factory for one isolated Sinergym/EnergyPlus worker."""

    worker_name = sanitize_env_name(f"{run_name}_worker_{worker_rank:02d}")

    def _init():
        env = gym.make(
            ENV_ID,
            env_name=worker_name,
            building_config=BUILDING_CONFIG,
            reward=TruePMV,
            reward_kwargs=REWARD_KWARGS,
            seed=SEED + worker_rank,
        )
        env = NormalizeAction(env)
        # Monitor supplies episode return/length without retaining complete
        # episodes as Sinergym LoggerWrapper does.
        env = Monitor(env)
        return env

    return _init


def remove_opencv_typing_path() -> None:
    """Keep OpenCV's nested ``typing`` package from shadowing stdlib typing.

    The host user-site OpenCV install adds its ``cv2`` directory to sys.path.
    Spawned workers then resolve ``import typing`` to ``cv2/typing`` and can
    fail when the container lacks OpenGL. OpenCV is not needed by this trainer;
    keep the parent user-site path (which provides SB3) and remove only cv2.
    """
    opencv_paths = [
        entry
        for entry in sys.path
        if os.path.basename(os.path.normpath(entry)) == "cv2"
        and os.path.isfile(os.path.join(entry, "__init__.py"))
        and os.path.isdir(os.path.join(entry, "typing"))
    ]
    for entry in opencv_paths:
        sys.path.remove(entry)
        print(
            f"Removed OpenCV package path from spawn import path: {entry}",
            flush=True,
        )


class EpisodeRewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.timesteps: List[int] = []
        self.episode_rewards: List[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.timesteps.append(self.num_timesteps)
                self.episode_rewards.append(float(info["episode"]["r"]))
        return True


class ParentWandBCallback(BaseCallback):
    """Build Sinergym-style episode summaries with constant-memory streaming."""

    SUMMARY_FIELDS = {
        "reward": "reward",
        "comfort_term": "reward_comfort_term",
        "energy_term": "reward_energy_term",
        "comfort_penalty": "comfort_penalty",
        "energy_penalty": "energy_penalty",
        "total_temperature_violation": "temperature_violation",
        "total_power_demand": "power_demand",
    }

    def __init__(self, wandb_module: Any):
        super().__init__()
        self.wandb = wandb_module
        self.completed_episodes = 0
        self.worker_stats: List[Dict[str, Any]] = []

    @staticmethod
    def _new_worker_stats() -> Dict[str, Any]:
        return {
            "count": 0,
            "sums": {
                source: 0.0
                for source in ParentWandBCallback.SUMMARY_FIELDS
            },
            "sum_squares": {
                source: 0.0
                for source in ParentWandBCallback.SUMMARY_FIELDS
            },
            "comfort_violation_steps": 0,
        }

    def _on_training_start(self) -> None:
        self.worker_stats = [
            self._new_worker_stats()
            for _ in range(self.training_env.num_envs)
        ]

    @staticmethod
    def _mean_and_std(
        total: float,
        total_squares: float,
        count: int,
    ) -> tuple[float, float]:
        if count <= 0:
            return 0.0, 0.0
        mean = total / count
        variance = max(total_squares / count - mean * mean, 0.0)
        return mean, float(np.sqrt(variance))

    def _log_completed_episode(
        self,
        worker_rank: int,
        info: Dict[str, Any],
    ) -> None:
        stats = self.worker_stats[worker_rank]
        count = int(stats["count"])
        self.completed_episodes += 1

        summary: Dict[str, Any] = {
            "episode_summaries/episode_num": self.completed_episodes,
            "episode_summaries/training_timesteps": self.num_timesteps,
            "episode_summaries/worker_rank": worker_rank,
            "episode_summaries/episode_return": float(
                info["episode"]["r"]
            ),
            "episode_summaries/length(timesteps)": int(
                info["episode"]["l"]
            ),
            "episode_summaries/time_elapsed(hours)": float(
                info.get("time_elapsed(hours)", 0.0)
            ),
            "episode_summaries/comfort_violation_time(%)": (
                100.0
                * stats["comfort_violation_steps"]
                / max(count, 1)
            ),
            "episode_summaries/cumulative_power_demand": float(
                stats["sums"]["total_power_demand"]
            ),
        }

        for source, output_name in self.SUMMARY_FIELDS.items():
            mean, standard_deviation = self._mean_and_std(
                float(stats["sums"][source]),
                float(stats["sum_squares"][source]),
                count,
            )
            summary[f"episode_summaries/mean_{output_name}"] = mean
            summary[f"episode_summaries/std_{output_name}"] = (
                standard_deviation
            )

        # Do not pass W&B's global `step` here. Six workers often complete at
        # exactly the same aggregate PPO timestep, and explicit duplicate steps
        # can cause later records to be dropped. The metric's custom step axis is
        # episode_summaries/episode_num, defined when the W&B run is created.
        self.wandb.log(summary)
        self.worker_stats[worker_rank] = self._new_worker_stats()

    def _on_step(self) -> bool:
        for worker_rank, info in enumerate(self.locals.get("infos", [])):
            stats = self.worker_stats[worker_rank]
            stats["count"] += 1

            for source in self.SUMMARY_FIELDS:
                value = float(info.get(source, 0.0))
                stats["sums"][source] += value
                stats["sum_squares"][source] += value * value

            if float(info.get("total_temperature_violation", 0.0)) > 0:
                stats["comfort_violation_steps"] += 1

            if "episode" in info:
                self._log_completed_episode(worker_rank, info)
        return True


class SubprocessEvalCallback(BaseCallback):
    """Evaluate a frozen policy in a dedicated EnergyPlus process."""

    def __init__(
        self,
        train_env: VecNormalize,
        verbose: int = 1,
        wandb_module: Any = None,
    ):
        super().__init__(verbose=verbose)
        self.train_env_raw = train_env
        self.eval_freq = EVAL_EVERY_EPISODES * STEPS_PER_EPISODE
        self.next_evaluation_timesteps = self.eval_freq
        self.evaluation_num = 0
        self.best_mean_reward = -np.inf
        self.wandb = wandb_module
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _evaluation_is_due(self) -> bool:
        return (
            self.eval_freq > 0
            and self.num_timesteps >= self.next_evaluation_timesteps
        )

    def _on_rollout_start(self) -> None:
        if self._evaluation_is_due():
            self._run_evaluation()

    def _on_training_end(self) -> None:
        if self._evaluation_is_due():
            self._run_evaluation()

    def _run_evaluation(self) -> None:
        while self.next_evaluation_timesteps <= self.num_timesteps:
            self.next_evaluation_timesteps += self.eval_freq

        self.evaluation_num += 1
        print(
            f"\nStarting isolated deterministic evaluation "
            f"{self.evaluation_num} at {self.num_timesteps} timesteps...",
            flush=True,
        )

        with tempfile.TemporaryDirectory(
            prefix=f"{RUN_BASE_NAME}_evaluation_"
        ) as temporary_dir:
            candidate_model_path = os.path.join(
                temporary_dir,
                "candidate_model.zip",
            )
            candidate_mean_path = os.path.join(temporary_dir, "mean.txt")
            candidate_var_path = os.path.join(temporary_dir, "var.txt")
            candidate_count_path = os.path.join(temporary_dir, "count.txt")
            evaluation_config_path = os.path.join(
                temporary_dir,
                "evaluation_config.json",
            )
            result_path = os.path.join(temporary_dir, "result.json")

            self.model.save(candidate_model_path)
            np.savetxt(candidate_mean_path, self.train_env_raw.obs_rms.mean)
            np.savetxt(candidate_var_path, self.train_env_raw.obs_rms.var)
            np.savetxt(
                candidate_count_path,
                [self.train_env_raw.obs_rms.count],
            )

            with open(
                evaluation_config_path,
                "w",
                encoding="utf-8",
            ) as config_file:
                json.dump(
                    {
                        "eval_env_id": EVAL_ENV_ID,
                        "building_config": EVAL_BUILDING_CONFIG,
                        "reward_kwargs": REWARD_KWARGS,
                    },
                    config_file,
                    indent=2,
                    sort_keys=True,
                )

            command = [
                sys.executable,
                os.path.abspath(__file__),
                "--evaluate-candidate",
                "--model-path",
                candidate_model_path,
                "--norm-mean-path",
                candidate_mean_path,
                "--norm-var-path",
                candidate_var_path,
                "--norm-count-path",
                candidate_count_path,
                "--evaluation-config-path",
                evaluation_config_path,
                "--result-path",
                result_path,
                "--evaluation-num",
                str(self.evaluation_num),
                "--evaluation-timesteps",
                str(self.num_timesteps),
                "--evaluation-seeds",
                ",".join(str(seed) for seed in EVAL_SEEDS),
                "--eval-env-name",
                sanitize_env_name(
                    f"{RUN_BASE_NAME}_EVALUATION_{self.evaluation_num}"
                ),
            ]

            with open(EVALUATION_LOG_PATH, "a", encoding="utf-8") as log_file:
                completed = subprocess.run(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

            if completed.returncode != 0 or not os.path.exists(result_path):
                print(
                    f"[WARN] Evaluation failed with exit code "
                    f"{completed.returncode}. See {EVALUATION_LOG_PATH}",
                    flush=True,
                )
                return

            with open(result_path, "r", encoding="utf-8") as result_file:
                summary = json.load(result_file)

            pd.DataFrame([summary]).to_csv(
                EVALUATION_METRICS_PATH,
                mode="a",
                header=not os.path.exists(EVALUATION_METRICS_PATH),
                index=False,
            )

            score = float(summary["mean_reward"])
            print(
                f"Evaluation {self.evaluation_num}: "
                f"mean_reward={score:.6f}, "
                f"mean_episode_return="
                f"{summary['mean_episode_return']:.6f}",
                flush=True,
            )

            if self.wandb is not None:
                evaluation_log: Dict[str, Any] = {
                    "evaluation/evaluation_num": self.evaluation_num,
                    "evaluation/training_timesteps": self.num_timesteps,
                }
                for key, value in summary.items():
                    if key in (
                        "evaluation_num",
                        "training_timesteps",
                        "evaluation_seeds",
                    ):
                        continue
                    if isinstance(
                        value,
                        (bool, int, float, np.integer, np.floating),
                    ):
                        evaluation_log[f"evaluation/{key}"] = value
                # evaluation/evaluation_num is the custom W&B step axis.
                self.wandb.log(evaluation_log)

            if score > self.best_mean_reward:
                _atomic_copy(candidate_model_path, BEST_MODEL_PATH)
                _atomic_copy(candidate_mean_path, BEST_NORM_MEAN_PATH)
                _atomic_copy(candidate_var_path, BEST_NORM_VAR_PATH)
                _atomic_copy(candidate_count_path, BEST_NORM_COUNT_PATH)
                self.best_mean_reward = score
                print(f"New best model saved: {BEST_MODEL_PATH}", flush=True)
                if self.wandb is not None:
                    self.wandb.log(
                        {
                            "best_model/training_timesteps": (
                                self.num_timesteps
                            ),
                            "best_model/mean_reward": score,
                            "best_model/mean_episode_return": float(
                                summary["mean_episode_return"]
                            ),
                        }
                    )


# =============================================================================
# 4) STANDALONE EVALUATOR
# =============================================================================

def make_evaluation_env(
    *,
    env_id: str,
    env_name: str,
    building_config: Dict[str, object],
    reward_kwargs: Dict[str, object],
    mean_path: str,
    var_path: str,
    count_path: str,
) -> gym.Env:
    env = gym.make(
        env_id,
        env_name=env_name,
        building_config=building_config,
        reward=TruePMV,
        reward_kwargs=reward_kwargs,
    )
    env = NormalizeObservation(
        env,
        automatic_update=False,
        mean=mean_path,
        var=var_path,
        count=count_path,
    )
    env = NormalizeAction(env)
    env = LoggerWrapper(env)
    env = Monitor(env)
    return env


def evaluate_candidate_in_subprocess(arguments: argparse.Namespace) -> None:
    evaluation_seeds = [
        int(seed) for seed in arguments.evaluation_seeds.split(",")
    ]
    if not evaluation_seeds:
        raise ValueError("At least one evaluation seed is required.")

    random.seed(evaluation_seeds[0])
    np.random.seed(evaluation_seeds[0])
    torch.manual_seed(evaluation_seeds[0])

    with open(
        arguments.evaluation_config_path,
        "r",
        encoding="utf-8",
    ) as config_file:
        config = json.load(config_file)

    building_config = dict(config["building_config"])
    building_config["runperiod"] = tuple(building_config["runperiod"])
    reward_kwargs = dict(config["reward_kwargs"])

    eval_env = make_evaluation_env(
        env_id=str(config["eval_env_id"]),
        env_name=arguments.eval_env_name,
        building_config=building_config,
        reward_kwargs=reward_kwargs,
        mean_path=arguments.norm_mean_path,
        var_path=arguments.norm_var_path,
        count_path=arguments.norm_count_path,
    )
    workspace_path = str(eval_env.get_wrapper_attr("workspace_path"))

    try:
        model = PPO.load(arguments.model_path, env=eval_env)
        episode_summaries: List[Dict[str, float]] = []

        for fixed_seed in evaluation_seeds:
            observation, _ = eval_env.reset(seed=fixed_seed)
            terminated = truncated = False
            episode_return = 0.0
            episode_steps = 0
            any_zone_violation_steps = 0
            zone_samples = 0
            zone_violations = 0
            occupied_zone_samples = 0
            occupied_zone_violations = 0

            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = (
                    eval_env.step(action)
                )
                episode_return += float(reward)
                episode_steps += 1

                zone_abs_pmv = [
                    abs(float(info[f"zone_{index}_pmv"]))
                    for index in range(1, 6)
                ]
                zone_occupancy = [
                    float(info[f"zone_{index}_occupancy"])
                    for index in range(1, 6)
                ]
                violation_flags = [pmv > 0.5 for pmv in zone_abs_pmv]
                any_zone_violation_steps += int(any(violation_flags))
                zone_samples += len(violation_flags)
                zone_violations += sum(violation_flags)

                for occupancy, violation in zip(
                    zone_occupancy,
                    violation_flags,
                ):
                    if occupancy > 0:
                        occupied_zone_samples += 1
                        occupied_zone_violations += int(violation)

            raw_summary = eval_env.get_wrapper_attr("get_episode_summary")()
            numeric_summary = {
                key: float(value)
                for key, value in raw_summary.items()
                if isinstance(
                    value,
                    (int, float, np.integer, np.floating),
                )
            }
            numeric_summary["episode_return"] = episode_return
            numeric_summary["logger_nonzero_abs_pmv_time(%)"] = (
                numeric_summary["comfort_violation_time(%)"]
            )
            numeric_summary["comfort_violation_time(%)"] = (
                100.0
                * any_zone_violation_steps
                / max(episode_steps, 1)
            )
            numeric_summary["zone_pmv_violation_rate(%)"] = (
                100.0 * zone_violations / max(zone_samples, 1)
            )
            numeric_summary[
                "occupied_zone_pmv_violation_rate(%)"
            ] = (
                100.0
                * occupied_zone_violations
                / max(occupied_zone_samples, 1)
            )
            episode_summaries.append(numeric_summary)

        common_keys = set.intersection(
            *(set(summary) for summary in episode_summaries)
        )
        evaluation_summary: Dict[str, Any] = {
            "evaluation_num": arguments.evaluation_num,
            "training_timesteps": arguments.evaluation_timesteps,
            "evaluation_episodes": len(evaluation_seeds),
            "evaluation_seeds": ",".join(
                str(seed) for seed in evaluation_seeds
            ),
            "deterministic_actions": True,
            "deterministic_weather": True,
        }
        for key in sorted(common_keys):
            evaluation_summary[key] = float(
                np.mean([summary[key] for summary in episode_summaries])
            )

        evaluation_summary["mean_episode_return"] = (
            evaluation_summary.pop("episode_return")
        )
        evaluation_summary["std_episode_return"] = float(
            np.std(
                [
                    summary["episode_return"]
                    for summary in episode_summaries
                ]
            )
        )

        with open(
            arguments.result_path,
            "w",
            encoding="utf-8",
        ) as result_file:
            json.dump(
                evaluation_summary,
                result_file,
                indent=2,
                sort_keys=True,
            )
    finally:
        eval_env.close()
        workdir_real = os.path.realpath(SINERGYM_WORKDIR)
        workspace_real = os.path.realpath(workspace_path)
        try:
            inside_workdir = (
                os.path.commonpath([workdir_real, workspace_real])
                == workdir_real
            )
        except ValueError:
            inside_workdir = False
        if (
            inside_workdir
            and workspace_real != workdir_real
            and "_EVALUATION_" in os.path.basename(workspace_real)
        ):
            shutil.rmtree(workspace_real, ignore_errors=True)


# =============================================================================
# 5) ARTIFACTS AND TRAINING
# =============================================================================

def save_training_artifacts(
    model: PPO,
    train_env: VecNormalize,
    reward_callback: EpisodeRewardCallback,
    *,
    interrupted: bool,
) -> None:
    suffix = " (interrupted — partial run)" if interrupted else ""

    model.save(MODEL_PATH)
    np.savetxt(NORM_MEAN_PATH, train_env.obs_rms.mean)
    np.savetxt(NORM_VAR_PATH, train_env.obs_rms.var)
    np.savetxt(NORM_COUNT_PATH, [train_env.obs_rms.count])

    reward_frame = pd.DataFrame(
        {
            "timesteps": reward_callback.timesteps,
            "episode_reward": reward_callback.episode_rewards,
        }
    )
    rolling_column = f"reward_rolling_avg_{ROLLING_WINDOW_EPISODES}"
    if reward_frame.empty:
        reward_frame[rolling_column] = []
    else:
        reward_frame[rolling_column] = (
            reward_frame["episode_reward"]
            .rolling(
                window=ROLLING_WINDOW_EPISODES,
                min_periods=1,
            )
            .mean()
        )
    reward_frame.to_csv(REWARD_CSV_PATH, index=False)

    plt.figure(figsize=(10, 5))
    if not reward_frame.empty:
        plt.plot(
            reward_frame["timesteps"],
            reward_frame["episode_reward"],
            alpha=0.35,
            label="Episode reward",
        )
        plt.plot(
            reward_frame["timesteps"],
            reward_frame[rolling_column],
            linewidth=2.0,
            label=f"Rolling avg ({ROLLING_WINDOW_EPISODES} episodes)",
        )
        plt.legend()
    plt.title(
        f"PPO Reward Convergence ({LOCATION_LABEL}, TruePMV, "
        f"{N_ENVS} parallel environments){suffix}"
    )
    plt.xlabel("Aggregate training timesteps")
    plt.ylabel("Episode reward")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()

    print("Model saved:", MODEL_PATH, flush=True)
    print("Reward CSV saved:", REWARD_CSV_PATH, flush=True)
    print("Reward plot saved:", PLOT_PATH, flush=True)
    print("Normalization statistics saved:", NORM_MEAN_PATH, flush=True)


def initialize_wandb() -> Any:
    if not (
        ENABLE_WANDB
        and WANDB_AVAILABLE
        and WANDB_ENTITY
        and WANDB_PROJECT
    ):
        return None

    import wandb

    wandb_run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=RUN_BASE_NAME,
        group=WANDB_GROUP,
        job_type="train",
        tags=[
            "ppo",
            "5zone",
            "zonal",
            LOCATION_LABEL,
            "stochastic-training",
            "subproc-vectorized",
            f"{N_ENVS}-envs",
            "TruePMV",
        ],
        save_code=False,
        reinit=False,
        config={
            "env_id": ENV_ID,
            "eval_env_id": EVAL_ENV_ID,
            "n_envs": N_ENVS,
            "subproc_start_method": SUBPROC_START_METHOD,
            "total_timesteps": TOTAL_TIMESTEPS,
            "combined_rollout_steps": COMBINED_ROLLOUT_STEPS,
            "n_steps_per_env": N_STEPS_PER_ENV,
            "building_config": BUILDING_CONFIG,
            "reward_kwargs": REWARD_KWARGS,
        },
    )
    wandb_run.define_metric(
        "episode_summaries/*",
        step_metric="episode_summaries/episode_num",
    )
    wandb_run.define_metric(
        "evaluation/*",
        step_metric="evaluation/evaluation_num",
    )
    wandb_run.define_metric(
        "best_model/*",
        step_metric="best_model/training_timesteps",
    )
    return wandb_run


def _sigterm_to_keyboard_interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def run_training() -> float:
    # Spawn workers inherit sys.path. Remove the host OpenCV subpackage that
    # otherwise shadows Python's standard-library `typing` during startup.
    remove_opencv_typing_path()
    worker_factories = [
        make_worker(rank, ENV_NAME) for rank in range(N_ENVS)
    ]
    subprocess_env = SubprocVecEnv(
        worker_factories,
        start_method=SUBPROC_START_METHOD,
    )
    train_env = VecNormalize(
        subprocess_env,
        training=True,
        norm_obs=True,
        norm_reward=False,
        clip_obs=float("inf"),
        epsilon=1e-8,
    )

    model = None
    reward_callback = EpisodeRewardCallback()
    artifacts_saved = False
    interrupted = False
    wandb_run = None
    old_sigterm: Any = None

    if hasattr(signal, "SIGTERM"):
        old_sigterm = signal.signal(
            signal.SIGTERM,
            _sigterm_to_keyboard_interrupt,
        )

    try:
        wandb_run = initialize_wandb()
        model = PPO(env=train_env, **PPO_PARAMS)

        callbacks: List[BaseCallback] = [reward_callback]
        wandb_module = None
        if wandb_run is not None:
            import wandb

            wandb_module = wandb
            callbacks.append(ParentWandBCallback(wandb))

        callbacks.append(
            SubprocessEvalCallback(
                train_env,
                verbose=1,
                wandb_module=wandb_module,
            )
        )

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=CallbackList(callbacks),
        )
        save_training_artifacts(
            model,
            train_env,
            reward_callback,
            interrupted=False,
        )
        artifacts_saved = True
    except (KeyboardInterrupt, SystemExit):
        interrupted = True
        print(
            "Training interrupted; saving partial artifacts...",
            flush=True,
        )
        raise
    finally:
        if not artifacts_saved and model is not None:
            try:
                save_training_artifacts(
                    model,
                    train_env,
                    reward_callback,
                    interrupted=True,
                )
            except Exception as exc:
                print(
                    f"[WARN] Partial artifact save failed: {exc}",
                    flush=True,
                )

        if old_sigterm is not None and hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, old_sigterm)

        train_env.close()
        if wandb_run is not None:
            wandb_run.finish(exit_code=130 if interrupted else 0)

    if reward_callback.episode_rewards:
        return float(np.mean(reward_callback.episode_rewards))
    return -1e12


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Starting standalone vectorized PPO training...")
    print("ENV_ID:", ENV_ID)
    print("EVAL_ENV_ID:", EVAL_ENV_ID)
    print("LOCATION_LABEL:", LOCATION_LABEL)
    print("N_ENVS:", N_ENVS)
    print("SUBPROC_START_METHOD:", SUBPROC_START_METHOD)
    print("BUILDING_CONFIG:", BUILDING_CONFIG)
    print("STEPS_PER_EPISODE:", STEPS_PER_EPISODE)
    print("TOTAL_EPISODES (aggregate):", TOTAL_EPISODES)
    print("TOTAL_TIMESTEPS (aggregate):", TOTAL_TIMESTEPS)
    print("N_STEPS_PER_ENV:", N_STEPS_PER_ENV)
    print("COMBINED_ROLLOUT_STEPS:", N_ENVS * N_STEPS_PER_ENV)
    print("PPO_PARAMS:", PPO_PARAMS)
    print("EVAL_EVERY_EPISODES:", EVAL_EVERY_EPISODES)
    print("EVAL_EPISODES:", EVAL_EPISODES)
    print("EVAL_SEEDS:", EVAL_SEEDS)
    print("ENABLE_WANDB:", ENABLE_WANDB)
    print("MODEL output:", MODEL_PATH)
    print("BEST_MODEL output:", BEST_MODEL_PATH)

    mean_reward = run_training()
    print("Training mean episode reward:", mean_reward)
    print("Parallel training complete.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-candidate", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--norm-mean-path")
    parser.add_argument("--norm-var-path")
    parser.add_argument("--norm-count-path")
    parser.add_argument("--evaluation-config-path")
    parser.add_argument("--result-path")
    parser.add_argument("--evaluation-num", type=int)
    parser.add_argument("--evaluation-timesteps", type=int)
    parser.add_argument("--evaluation-seeds")
    parser.add_argument("--eval-env-name")
    arguments = parser.parse_args()

    if arguments.evaluate_candidate:
        required = (
            "model_path",
            "norm_mean_path",
            "norm_var_path",
            "norm_count_path",
            "evaluation_config_path",
            "result_path",
            "evaluation_num",
            "evaluation_timesteps",
            "evaluation_seeds",
            "eval_env_name",
        )
        missing = [
            name for name in required if getattr(arguments, name) is None
        ]
        if missing:
            parser.error(
                "Evaluator mode is missing: "
                + ", ".join(
                    f"--{name.replace('_', '-')}" for name in missing
                )
            )
    return arguments


if __name__ == "__main__":
    args = parse_arguments()
    try:
        if args.evaluate_candidate:
            evaluate_candidate_in_subprocess(args)
        else:
            main()
    except KeyboardInterrupt:
        print(
            "\nKeyboardInterrupt: training stopped; partial artifacts were "
            "saved best-effort.",
            flush=True,
        )
        raise SystemExit(130) from None
