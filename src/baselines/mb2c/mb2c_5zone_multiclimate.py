#!/usr/bin/env python3
"""Shared collector for the final three-climate, seeded-TMY MB2C dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from mb2c_5zone import (
    ACTION_HIGH, ACTION_LOW, BUILDING_VARS, OCC_VARS, TIMESTEPS_PER_HOUR,
    WEATHER_VARS, as_dict, exogenous,
)

CLIMATES = (("arizona", "Arizona"), ("chicago", "Chicago"), ("mixed", "New York"))
SUMMER_MONTHS = {6, 7, 8, 9}


def collect_one(spec: tuple) -> str:
    index, climate_key, climate_name, year, seed, data_dir, work_root, max_steps = spec
    workdir = Path(work_root) / f"worker_{index:02d}"
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ["SINERGYM_WORKDIR"] = str(workdir)
    os.chdir(workdir)

    import gymnasium as gym
    import sinergym  # noqa: F401; registers local environments

    env_id = f"Eplus-5zone-zonal-{climate_key}-continuous-stochastic-v1"
    env_name = f"mb2c5z_{climate_key}_{year}_{seed}"
    env = gym.make(
        env_id,
        env_name=env_name,
        building_config={
            "timesteps_per_hour": TIMESTEPS_PER_HOUR,
            "runperiod": (1, 1, year, 31, 12, year),
        },
        seed=seed,
    )
    names = list(env.get_wrapper_attr("observation_variables"))
    required = set(BUILDING_VARS + WEATHER_VARS + OCC_VARS + ["month", "day_of_month", "hour"])
    missing = required - set(names)
    if missing:
        env.close()
        raise RuntimeError(f"{env_id} missing observations: {sorted(missing)}")

    obs, _ = env.reset(seed=seed)
    current = as_dict(obs, names)
    xs, deltas, currents, nexts = [], [], [], []
    for _ in range(max_steps or 370 * 24 * TIMESTEPS_PER_HOUR):
        month = int(current["month"])
        action_values = []
        for occ_var in OCC_VARS:
            occupied = current[occ_var] > 0
            if not occupied:
                heat, cool = 12.0, 30.0
            elif month in SUMMER_MONTHS:
                heat, cool = 23.0, 24.5
            else:
                heat, cool = 20.0, 23.5
            action_values.extend((heat, cool))
        action = np.asarray(action_values, dtype=np.float32)
        if np.any(action < ACTION_LOW) or np.any(action > ACTION_HIGH):
            raise RuntimeError(f"RBC action outside legal bounds: {action}")

        nxt_obs, _, terminated, truncated, _ = env.step(action)
        nxt = as_dict(nxt_obs, names)
        state = np.asarray([current[k] for k in BUILDING_VARS], dtype=np.float32)
        next_state = np.asarray([nxt[k] for k in BUILDING_VARS], dtype=np.float32)
        xs.append(np.concatenate((state, action, exogenous(nxt))))
        deltas.append(next_state - state)
        currents.append(state)
        nexts.append(next_state)
        current = nxt
        if terminated or truncated:
            break
    env.close()

    weather_files = list(workdir.rglob("*_OU_Noise.epw"))
    if len(weather_files) != 1:
        raise RuntimeError(f"Expected one frozen OU-noised EPW in {workdir}, found {len(weather_files)}")
    weather_sha256 = hashlib.sha256(weather_files[0].read_bytes()).hexdigest()

    # The source EPWs are TMY3 files with 8760 hourly records (365 days).
    # EnergyPlus therefore produces 35040 steps even when the simulation
    # calendar year is a leap year; it does not add a Feb 29 TMY weather day.
    expected_steps = 365 * 24 * TIMESTEPS_PER_HOUR
    if not max_steps and len(xs) < expected_steps - 4:
        raise RuntimeError(f"{climate_name} {year}: short episode ({len(xs)} vs {expected_steps} steps)")
    out = Path(data_dir) / f"episode_{index:02d}.npz"
    np.savez_compressed(
        out, x=np.asarray(xs), delta=np.asarray(deltas), current=np.asarray(currents),
        next=np.asarray(nexts), seed=np.asarray(seed), steps=np.asarray(len(xs)),
        climate=np.asarray(climate_name), weather_key=np.asarray(climate_key),
        weather_year=np.asarray(year), weather_sha256=np.asarray(weather_sha256),
    )
    print(f"Collected {climate_name} {year}: {len(xs)} steps -> {out}", flush=True)
    return str(out)


def collection_specs(root: Path, max_steps: int = 0) -> tuple[list[tuple], dict]:
    """Return the recorded 12 train, 3 validation, and 3 test trajectories."""
    split_specs = (("train", 0, 4, root / "development"),
                   ("validation", 12, 1, root / "development"),
                   ("test", 0, 1, root / "test"))
    specs, splits = [], {"train": [], "validation": [], "test": []}
    seed_offset = {"train": 0, "validation": 100, "test": 200}
    for split, start, repeats, data_dir in split_specs:
        for cycle in range(repeats):
            for climate_index, (key, name) in enumerate(CLIMATES):
                episode_index = start + cycle * len(CLIMATES) + climate_index
                seed = 74000 + seed_offset[split] + cycle * len(CLIMATES) + climate_index
                work_root = root / "sinergym_collection" / split
                specs.append((episode_index, key, name, 1991, seed, str(data_dir),
                              str(work_root), max_steps))
                splits[split].append({
                    "climate": name,
                    "episode": f"{data_dir.relative_to(root)}/episode_{episode_index:02d}.npz",
                    "seed": seed,
                })
    manifest = {
        "method": "Five-zone stochastic Sinergym trajectories under the fixed RBC",
        "climates": {name: key for key, name in CLIMATES},
        "weather": "One TMY3 EPW per climate with seeded stochastic perturbations; seeds are replicates, not distinct historical years.",
        "runperiod": "1991-01-01 to 1991-12-31; 15-minute steps",
        "step_minutes": 15,
        "rbc_setpoints_c": {"occupied_summer_months_6_to_9": [23.0, 24.5],
                             "occupied_other_months": [20.0, 23.5],
                             "unoccupied": [12.0, 30.0]},
        "splits": splits,
    }
    return specs, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Limit steps for a short smoke run; 0 collects full trajectories")
    parser.add_argument("--only-index", type=int, choices=range(18), default=None,
                        help="Collect one trajectory from the canonical 18-trajectory split")
    args = parser.parse_args()
    root = args.output_root.resolve()
    for folder in (root / "development", root / "test", root / "sinergym_collection"):
        folder.mkdir(parents=True, exist_ok=True)
    specs, manifest = collection_specs(root, args.max_steps)
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.only_index is not None:
        specs = [specs[args.only_index]]
    with mp.get_context("spawn").Pool(processes=min(args.workers, len(specs))) as pool:
        for result in pool.imap_unordered(collect_one, specs):
            print(f"Collected: {result}", flush=True)


if __name__ == "__main__":
    main()
