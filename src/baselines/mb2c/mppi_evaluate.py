#!/usr/bin/env python3
"""Evaluate the trained MB2C model on a fixed seven-day weather trace."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import deque
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

ZONE_IDS = range(1, 6)
BUILDING_VARS = (
    [f"air_temperature_space{i}" for i in ZONE_IDS]
    + [f"air_humidity_space{i}" for i in ZONE_IDS]
    + [f"new_pmv_space{i}" for i in ZONE_IDS]
    + ["total_electricity_HVAC"]
)
WEATHER_VARS = ["outdoor_temperature", "outdoor_humidity", "wind_speed", "wind_direction",
                "diffuse_solar_radiation", "direct_solar_radiation"]
OCC_VARS = [f"air_occ_space{i}" for i in ZONE_IDS]
PMV_VARS = [f"new_pmv_space{i}" for i in ZONE_IDS]
ACTION_LOW = np.array([12.0, 23.25] * 5, dtype=np.float32)
ACTION_HIGH = np.array([23.25, 30.0] * 5, dtype=np.float32)


def enforce_heat_cool_gap(actions, gap=0.5):
    """Project each zone's HSP/CSP pair to bounds with CSP - HSP >= gap."""
    actions = np.clip(actions, ACTION_LOW, ACTION_HIGH).copy()
    heat, cool = actions[..., 0::2], actions[..., 1::2]
    bad = (cool - heat) < gap
    projected_heat = (heat + cool - gap) / 2.0
    projected_cool = projected_heat + gap
    low_heat = ACTION_LOW[0::2]
    high_cool = ACTION_HIGH[1::2]
    too_low = projected_heat < low_heat
    too_high = projected_cool > high_cool
    projected_heat = np.where(too_low, low_heat, projected_heat)
    projected_cool = np.where(too_low, low_heat + gap, projected_cool)
    projected_cool = np.where(too_high, high_cool, projected_cool)
    projected_heat = np.where(too_high, high_cool - gap, projected_heat)
    heat[:] = np.where(bad, projected_heat, heat)
    cool[:] = np.where(bad, projected_cool, cool)
    return actions


class Member(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 200), nn.ReLU(),
                                 nn.Linear(200, 200), nn.ReLU(),
                                 nn.Linear(200, output_dim))

    def forward(self, x):
        return self.net(x)


def exog_from_obs(obs: dict[str, float]) -> np.ndarray:
    month, day, hour = int(obs["month"]), int(obs["day_of_month"]), float(obs["hour"])
    doy = (month - 1) * 30 + day
    cyc = [math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12),
           math.sin(2 * math.pi * doy / 365.25), math.cos(2 * math.pi * doy / 365.25),
           math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)]
    return np.asarray([obs[k] for k in WEATHER_VARS + OCC_VARS] + cyc, dtype=np.float32)


def state_from_obs(obs: dict[str, float]) -> np.ndarray:
    return np.asarray([obs[k] for k in BUILDING_VARS], dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truepmv_reward(states: np.ndarray, exog: np.ndarray) -> np.ndarray:
    pmv = states[:, 10:15]
    occupancy = exog[:, 6:11]
    rho = np.where(occupancy > 0.0, 1.1, 0.1)
    pmv_penalty = np.clip(np.maximum(np.abs(pmv) - 0.5, 0.0) / 4.0, 0.0, 1.0)
    energy_kwh = states[:, 15] / 3.6e6
    energy_penalty = np.clip(energy_kwh / 4.5, 0.0, 1.0)
    return -1.5 * (rho * pmv_penalty).sum(axis=1) - 1.0 * energy_penalty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--environment-data", type=Path, required=True,
                        help="Held-out trajectory NPZ for this climate")
    parser.add_argument("--weather-file", type=Path, required=True,
                        help="Frozen OU-noised EPW used to reproduce the held-out trace")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="MPPI cost temperature lambda; paper does not publish its numeric value")
    parser.add_argument("--noise-sigma", type=float, default=2.0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=0, help="Short login-node smoke only; 0 runs full period")
    parser.add_argument("--run-start-month", type=int, required=True)
    parser.add_argument("--run-start-day", type=int, required=True)
    parser.add_argument("--run-start-year", type=int, required=True)
    parser.add_argument("--run-end-month", type=int, required=True)
    parser.add_argument("--run-end-day", type=int, required=True)
    parser.add_argument("--run-end-year", type=int, required=True)
    parser.add_argument("--report-month", type=int, required=True)
    parser.add_argument("--report-start-day", type=int, required=True)
    parser.add_argument("--report-end-day", type=int, required=True)
    parser.add_argument("--report-year", type=int, required=True)
    parser.add_argument("--report-prefix", required=True)
    parser.add_argument("--online-window-days", type=int, default=61,
                        help="Recent MPPI transitions used for seasonal model adaptation")
    parser.add_argument("--online-update-epochs", type=int, default=40)
    parser.add_argument("--online-update-every-days", type=int, default=7)
    args = parser.parse_args()

    if args.horizon % 2:
        raise ValueError("Two-step action blocks require an even horizon")
    if args.iterations < 1 or args.samples < 12 or args.temperature <= 0:
        raise ValueError("Use at least one iteration, 12 candidates, and a positive lambda")
    workdir = Path(os.environ.get("SINERGYM_WORKDIR", args.output_dir / "sinergym")).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    reference = np.load(args.environment_data)
    ref_x = reference["x"].astype(np.float32)
    trace_seed = int(reference["seed"])
    trace_climate = str(reference["weather_key"])
    expected_climate = {"arizona": "arizona", "chicago": "chicago", "mixed": "mixed"}
    climate_key = next((key for key in expected_climate
                        if f"-{key}-" in args.env_id.lower()), None)
    if climate_key is None or trace_climate != climate_key:
        raise ValueError(f"Environment {args.env_id} and held-out trace climate {trace_climate} differ")
    if trace_seed != args.seed:
        raise ValueError(f"Environment seed {args.seed} does not match held-out trace seed {trace_seed}")
    weather_sha256 = sha256_file(args.weather_file)
    if "weather_sha256" in reference and str(reference["weather_sha256"]) != weather_sha256:
        raise ValueError("Frozen EPW does not match the EPW recorded in the held-out trajectory")
    reference_year = 1991
    run_start = date(args.run_start_year, args.run_start_month, args.run_start_day)
    run_end = date(args.run_end_year, args.run_end_month, args.run_end_day)
    report_start = date(args.report_year, args.report_month, args.report_start_day)
    report_end = date(args.report_year, args.report_month, args.report_end_day)
    jan1 = date(reference_year, 1, 1)
    start_index = (date(reference_year, args.run_start_month, args.run_start_day) - jan1).days * 96
    duration_steps = ((run_end - run_start).days + 1) * 96
    report_start_index = (report_start - run_start).days * 96
    report_end_index = (report_end - run_start).days * 96 + 96
    if report_start < run_start or report_end > run_end or report_end < report_start:
        raise ValueError("Report dates must fall within the simulation run period")
    if start_index >= len(ref_x) or duration_steps > len(ref_x):
        raise ValueError("Run period must fit within one circular pass of the reference-year trace")
    x_mean, x_std = np.asarray(ckpt["x_mean"]), np.asarray(ckpt["x_std"])
    y_mean, y_std = np.asarray(ckpt["y_mean"]), np.asarray(ckpt["y_std"])
    members = []
    for sd in ckpt["models"]:
        model = Member(ref_x.shape[1], len(BUILDING_VARS)).to(device)
        model.load_state_dict(sd)
        model.eval()
        members.append(model)
    ensemble_weights = np.full(len(members), 1.0 / len(members), dtype=np.float64)
    error_window: deque[np.ndarray] = deque(maxlen=4)
    online_x: deque[np.ndarray] = deque(maxlen=args.online_window_days * 96)
    online_y: deque[np.ndarray] = deque(maxlen=args.online_window_days * 96)
    adaptation_history = []

    import gymnasium as gym
    import sinergym  # noqa: F401
    from sinergym.utils.rewards import TruePMV

    reward_kwargs = {
        "pmv_variables": PMV_VARS,
        "occupancy_variables": OCC_VARS,
        "energy_variable": "total_electricity_HVAC",
        "pmv_min": 0.0, "pmv_max": 4.0,
        "energy_kwh_min": 0.0, "energy_kwh_max": 4.5,
        "rho_occupied": 1.1, "rho_unoccupied": 0.1,
    }
    if not args.weather_file.is_file():
        raise FileNotFoundError(args.weather_file)
    env_kwargs = {"weather_files": [str(args.weather_file.resolve())]}
    env = gym.make(
        args.env_id,
        env_name=re.sub(r"[^A-Za-z0-9_-]+", "_", f"mb2c_mppi_{args.report_prefix}_{args.seed}"),
        building_config={"timesteps_per_hour": 4,
                         "runperiod": (args.run_start_day, args.run_start_month, args.run_start_year,
                                       args.run_end_day, args.run_end_month, args.run_end_year)},
        reward=TruePMV,
        reward_kwargs=reward_kwargs,
        seed=args.seed,
        **env_kwargs,
    )
    obs_names = list(env.get_wrapper_attr("observation_variables"))
    obs, _ = env.reset(seed=args.seed, options={"weather_variability": None})
    obs_dict = dict(zip(obs_names, map(float, obs)))
    state = state_from_obs(obs_dict)
    names = obs_names
    n_steps = min(duration_steps, args.max_steps or duration_steps)
    nominal = np.tile(np.array([20.0, 26.0] * 5, dtype=np.float32), (args.horizon, 1))
    rng = np.random.default_rng(args.seed + 918)
    rngs = np.full(10, args.noise_sigma, dtype=np.float32)
    rows = []
    per_zone_pmv = [[] for _ in range(5)]
    per_zone_occ = [0] * 5
    monthly = {}
    rewards, energy_j = [], []
    plan_seconds = []
    context_abs_errors = []
    context_feature_errors = [[] for _ in WEATHER_VARS + OCC_VARS + ["month_sin", "month_cos",
                                                                   "day_sin", "day_cos",
                                                                   "hour_sin", "hour_cos"]]
    initial_state_error = float(np.mean(np.abs(state - reference["current"][start_index])))

    def adapt_ensemble(step_number: int) -> None:
        """Fine-tune the pretrained ensemble on recent, actually applied MPPI transitions."""
        if not online_x:
            return
        x_train = (np.asarray(online_x, dtype=np.float32) - x_mean) / x_std
        y_train = (np.asarray(online_y, dtype=np.float32) - y_mean) / y_std
        xt = torch.as_tensor(x_train, dtype=torch.float32, device=device)
        yt = torch.as_tensor(y_train, dtype=torch.float32, device=device)
        for member_id, model in enumerate(members):
            torch.manual_seed(args.seed + step_number + member_id)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(args.online_update_epochs):
                permutation = torch.randperm(len(xt), device=device)
                for batch_start in range(0, len(permutation), 512):
                    batch = permutation[batch_start:batch_start + 512]
                    loss = ((model(xt[batch]) - yt[batch]) ** 2).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            model.eval()
        adaptation_history.append({
            "step": step_number, "transition_count": len(online_x),
            "window_days": args.online_window_days,
            "epochs_per_member": args.online_update_epochs,
            "members": len(members),
        })
        print(f"Seasonal model update at step {step_number}: fitted 5 members on "
              f"{len(online_x)} recent MPPI transitions", flush=True)

    def predict_delta(states, actions, exogs, weights):
        x = np.concatenate([states, actions, exogs], axis=1)
        xt = torch.as_tensor((x - x_mean) / x_std, dtype=torch.float32, device=device)
        with torch.no_grad():
            preds = np.stack([m(xt).cpu().numpy() * y_std + y_mean for m in members])
        return np.einsum("m,mkd->kd", weights, preds), preds

    def update_weights(per_member_delta, actual_delta):
        errs = np.mean(((per_member_delta - actual_delta[None, :]) / y_std[None, :]) ** 2, axis=1)
        error_window.append(errs)
        # Weight members by inverse error over four recent transitions; recent
        # errors decay by 0.9 per step.
        stack = np.asarray(error_window)
        age_weights = np.asarray([0.9 ** (len(stack) - 1 - i) for i in range(len(stack))])
        mse = np.average(stack, axis=0, weights=age_weights)
        spread = float(mse.max() - mse.min())
        if spread < 1e-12:
            return np.full(len(members), 1.0 / len(members))
        normalized = (mse - mse.min()) / spread
        inverse = 1.0 - normalized
        return inverse / inverse.sum()

    csv_path = args.output_dir / "mppi_energyplus_trace.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "month", "day", "hour", "reward", "energy_kwh", "ensemble_weights"]
                                + WEATHER_VARS + PMV_VARS + OCC_VARS
                                + [f"air_temperature_space{i}" for i in ZONE_IDS]
                                + [f"action_{i}" for i in range(10)])
        writer.writeheader()
        for step in range(n_steps):
            update_interval = args.online_update_every_days * 96
            window_steps = args.online_window_days * 96
            if (step >= window_steps and (step - window_steps) % update_interval == 0
                    and (args.max_steps == 0 or step >= report_start_index)):
                adapt_ensemble(step)
            t0 = time.perf_counter()
            # Rollouts use the held-out weather/occupancy trace as known context.
            # Two 15-minute steps share one candidate action. Replan every step.
            nominal = np.repeat(nominal[::2], 2, axis=0)[:args.horizon].copy()
            noise_sequence = np.broadcast_to(rngs, nominal.shape).copy()
            noise_floor = np.array([0.25, 0.15] * 5, dtype=np.float32)
            for _iteration in range(args.iterations):
                block_std = noise_sequence[::2]
                block_noise = rng.normal(0.0, block_std, size=(args.samples, len(block_std), 10)).astype(np.float32)
                noise = np.repeat(block_noise, 2, axis=1)[:, :args.horizon, :]
                candidates = enforce_heat_cool_gap(nominal[None, :, :] + noise)
                candidates[0] = nominal
                sim_state = np.repeat(state[None, :], args.samples, axis=0)
                discounted_return = np.zeros(args.samples, dtype=np.float64)
                for h in range(args.horizon):
                    ref_index = (start_index + step + h) % len(ref_x)
                    future_exog = np.repeat(ref_x[ref_index, 26:][None, :], args.samples, axis=0)
                    delta, _ = predict_delta(sim_state, candidates[:, h, :], future_exog,
                                             ensemble_weights)
                    sim_state = sim_state + delta
                    discounted_return += (0.99 ** h) * truepmv_reward(sim_state, future_exog)
                costs = -discounted_return
                elite_count = min(args.samples, max(12, int(math.ceil(0.10 * args.samples))))
                elite_ids = np.argsort(costs)[:elite_count]
                elite_costs = costs[elite_ids]
                logits = -(elite_costs - elite_costs.min()) / max(args.temperature, 1e-8)
                weights = np.exp(logits - logits.max())
                weights /= weights.sum()
                elite_actions = candidates[elite_ids]
                elite_mean = np.einsum("k,kha->ha", weights, elite_actions)
                elite_var = np.einsum("k,kha->ha", weights, (elite_actions - elite_mean) ** 2)
                nominal = enforce_heat_cool_gap(0.30 * nominal + 0.70 * elite_mean).astype(np.float32)
                noise_sequence = np.clip(0.30 * noise_sequence + 0.70 * np.sqrt(np.maximum(elite_var, 0.0)), noise_floor, rngs).astype(np.float32)
            action = nominal[0].copy()
            plan_seconds.append(time.perf_counter() - t0)

            next_obs, env_reward, terminated, truncated, _ = env.step(action)
            next_dict = dict(zip(names, map(float, next_obs)))
            next_state = state_from_obs(next_dict)
            next_exog = exog_from_obs(next_dict)
            ref_exog = ref_x[(start_index + step) % len(ref_x), 26:]
            context_abs_errors.append(float(np.mean(np.abs(next_exog - ref_exog))))
            for feature_index, err in enumerate(np.abs(next_exog - ref_exog)):
                context_feature_errors[feature_index].append(float(err))
            # Score each member's executed one-step prediction, then update the
            # paper-style dynamic ensemble weights for the next control cycle.
            _, member_deltas_normed = predict_delta(
                state[None, :], action[None, :], next_exog[None, :],
                np.full(len(members), 1.0 / len(members)),
            )
            member_deltas = member_deltas_normed[:, 0, :]
            ensemble_weights = update_weights(member_deltas, next_state - state)
            transition_x = np.concatenate([state, action, next_exog]).astype(np.float32)
            online_x.append(transition_x)
            online_y.append((next_state - state).astype(np.float32))

            energy = next_dict["total_electricity_HVAC"]
            energy_j.append(energy)
            rewards.append(float(env_reward))
            row = {"step": step + 1, "reward": float(env_reward),
                   "energy_kwh": energy / 3.6e6,
                   "month": int(next_dict["month"]),
                   "day": int(next_dict["day_of_month"]),
                   "hour": float(next_dict["hour"]),
                   "ensemble_weights": ";".join(f"{v:.5f}" for v in ensemble_weights)}
            row.update({name: next_dict[name] for name in WEATHER_VARS})
            step_date = run_start + timedelta(minutes=15 * step)
            month_key = step_date.strftime("%Y-%m")
            month_record = monthly.setdefault(month_key, {
                "steps": 0, "reward_sum": 0.0, "energy_joules": 0.0,
                "pmv_by_zone": [[] for _ in range(5)], "occupied_steps_by_zone": [0] * 5,
            })
            month_record["steps"] += 1
            month_record["reward_sum"] += float(env_reward)
            month_record["energy_joules"] += energy
            for i in range(5):
                pmv, occ = next_dict[PMV_VARS[i]], next_dict[OCC_VARS[i]]
                row[PMV_VARS[i]], row[OCC_VARS[i]] = pmv, occ
                row[f"air_temperature_space{i + 1}"] = next_dict[f"air_temperature_space{i + 1}"]
                if occ > 0:
                    per_zone_pmv[i].append(pmv)
                    per_zone_occ[i] += 1
                    month_record["pmv_by_zone"][i].append(pmv)
                    month_record["occupied_steps_by_zone"][i] += 1
            for i, value in enumerate(action):
                row[f"action_{i}"] = float(value)
            writer.writerow(row)
            rows.append(row)
            f.flush()
            state, obs_dict = next_state, next_dict
            nominal = np.concatenate([nominal[1:], nominal[-1:]], axis=0)
            if (step + 1) % 100 == 0:
                print(f"MPPI EnergyPlus step {step+1}/{n_steps}; mean planning time="
                      f"{np.mean(plan_seconds[-100:]):.3f}s", flush=True)
            if terminated or truncated:
                break

    env.close()
    completed = len(rewards)
    monthly_report = {}
    for month_key, rec in monthly.items():
        monthly_report[month_key] = {
            "steps": rec["steps"],
            "reward_sum_truepmv": float(rec["reward_sum"]),
            "hvac_energy_kwh": float(rec["energy_joules"] / 3.6e6),
            "occupied_pmv_by_zone": {
                PMV_VARS[i]: {
                    "occupied_steps": rec["occupied_steps_by_zone"][i],
                    "mean_abs_pmv": float(np.mean(np.abs(rec["pmv_by_zone"][i])))
                    if rec["pmv_by_zone"][i] else None,
                    "fraction_abs_pmv_over_0_5": float(np.mean(np.abs(rec["pmv_by_zone"][i]) > 0.5))
                    if rec["pmv_by_zone"][i] else None,
                } for i in range(5)
            },
        }
    # Report only the selected evaluation dates; preceding rows warm up and
    # adapt the controller using its own applied transitions.
    report_rows = rows[report_start_index:report_end_index]
    evaluation_rewards = [float(r["reward"]) for r in report_rows]
    evaluation_energy_kwh = [float(r["energy_kwh"]) for r in report_rows]
    evaluation_occupied_pmv = [[] for _ in range(5)]
    evaluation_occupied_count = [0] * 5
    for row in report_rows:
        for i in range(5):
            if float(row[OCC_VARS[i]]) > 0:
                evaluation_occupied_pmv[i].append(float(row[PMV_VARS[i]]))
                evaluation_occupied_count[i] += 1
    report_month_key = f"{report_start.year}-{args.report_month:02d}"
    evaluation_monthly_report = (
        {report_month_key: monthly_report[report_month_key]}
        if report_month_key in monthly_report else {}
    )
    report = {
        "controller": "MB2C-style MPPI over the trained 5-member ENN",
        "planning_objective_note": "MPPI scales TruePMV comfort penalty by 1.5; environment reward remains at its original TruePMV scale",
        "mppi_setup": {"samples_K": args.samples, "horizon_H": args.horizon, "iterations_per_step": args.iterations, "action_block_steps": 2, "elite_fraction": 0.10, "elite_minimum": 12, "update_rate": 0.70, "hsp_noise_floor_c": 0.25, "csp_noise_floor_c": 0.15, "planning_comfort_weight": 1.5, "planning_energy_weight": 1.0,
                             "minimum_heat_cool_gap_c": 0.5,
                             "timestep_minutes": 15, "discount_gamma": 0.99,
                             "warm_start": "shift previous optimized sequence by one step",
                             "ensemble_error_window_C": 4, "ensemble_recent_error_decay": 0.9},
        "mppi_adaptation": {"temperature_lambda": args.temperature,
                             "action_noise_sigma_c": args.noise_sigma,
                             "note": "Paper does not report numeric lambda/noise values; these are explicit experiment settings."},
        "environment": args.env_id,
        "heldout_trace": {"climate": trace_climate, "seed": trace_seed,
                          "weather_file": str(args.weather_file.resolve()),
                          "weather_sha256": weather_sha256,
                          "trajectory_sha256": sha256_file(args.environment_data),
                          "checkpoint_sha256": sha256_file(args.checkpoint),
                          "context_note": "Future weather and occupancy are read from the held-out trace during MPPI rollouts."},
        "runperiod": f"{run_start.isoformat()} to {run_end.isoformat()}",
        "reported_evaluation_period": f"{report_start.isoformat()} to {report_end.isoformat()}",
        "steps_completed": completed,
        "reported_steps": len(report_rows),
        "initial_state_mae_vs_reference_trace": initial_state_error,
        "mean_next_context_mae_vs_reference_trace": float(np.mean(context_abs_errors)) if context_abs_errors else None,
        "next_context_mae_by_variable": {
            name: float(np.mean(values)) if values else None
            for name, values in zip(WEATHER_VARS + OCC_VARS + ["month_sin", "month_cos",
                                                                "day_sin", "day_cos",
                                                                "hour_sin", "hour_cos"],
                                   context_feature_errors)
        },
        "mean_planning_seconds_per_step": float(np.mean(plan_seconds)) if plan_seconds else None,
        "total_return_truepmv": float(np.sum(evaluation_rewards)),
        "mean_step_reward_truepmv": float(np.mean(evaluation_rewards)) if evaluation_rewards else None,
        "total_hvac_energy_kwh": float(np.sum(evaluation_energy_kwh)),
        "mean_hvac_energy_kwh_per_step": float(np.mean(evaluation_energy_kwh)) if evaluation_energy_kwh else None,
        "ppo_truepmv": {"pmv_variables": PMV_VARS, "occupancy_variables": OCC_VARS,
                        "rho_occupied": 1.1, "rho_unoccupied": 0.1,
                        "pmv_deadband": 0.5, "pmv_max": 4.0, "energy_kwh_max": 4.5},
        "occupied_pmv_by_zone": {
            PMV_VARS[i]: {
                "occupied_steps": int(evaluation_occupied_count[i]),
                "mean_pmv": float(np.mean(evaluation_occupied_pmv[i])) if evaluation_occupied_pmv[i] else None,
                "mean_abs_pmv": float(np.mean(np.abs(evaluation_occupied_pmv[i]))) if evaluation_occupied_pmv[i] else None,
                "fraction_abs_pmv_over_0_5": float(np.mean(np.abs(evaluation_occupied_pmv[i]) > 0.5)) if evaluation_occupied_pmv[i] else None,
            } for i in range(5)
        },
        "monthly_metrics": evaluation_monthly_report,
        "seasonal_adaptation": {
            "method": "fine-tune each pretrained ensemble member on recent MPPI-applied transitions",
            "window_days": args.online_window_days,
            "update_every_days": args.online_update_every_days,
            "epochs_per_member": args.online_update_epochs,
            "updates": adaptation_history,
        },
        "final_ensemble_weights": [float(x) for x in ensemble_weights],
        "trace_csv": csv_path.name,
    }
    report_path = args.output_dir / "mppi_energyplus_report.json"
    import pandas as pd
    from pmv_energy_metrics import compute_performance_metrics, save_pmv_summary_artifacts
    metric_rows = []
    for row in report_rows:
        metric_row = {"month": row["month"], "day": row["day"],
                      "HVAC_electricity_demand_rate": row["energy_kwh"] * 1000.0 / 0.25}
        metric_row.update({f"pmv_space{i}": row[f"new_pmv_space{i}"] for i in ZONE_IDS})
        metric_row.update({f"air_occ_space{i}": row[f"air_occ_space{i}"] for i in ZONE_IDS})
        metric_row.update({f"air_temperature_space{i}": row.get(f"air_temperature_space{i}", np.nan)
                           for i in ZONE_IDS})
        metric_rows.append(metric_row)
    evaluation_metrics = compute_performance_metrics(
        pd.DataFrame(metric_rows),
        [args.report_start_day, args.report_month, report_start.year],
        [args.report_end_day, args.report_month, report_start.year])
    report["evaluation_metrics"] = evaluation_metrics
    pmv_csv = args.output_dir / f"{args.report_prefix}_pmv_summary.csv"
    pmv_txt = args.output_dir / f"{args.report_prefix}_pmv_summary.txt"
    save_pmv_summary_artifacts(
        evaluation_metrics, label=f"{args.env_id} | custom PMV | MB2C MPPI",
        csv_path=str(pmv_csv), txt_path=str(pmv_txt),
        date_start=[args.report_start_day, args.report_month, report_start.year],
        date_end=[args.report_end_day, args.report_month, report_start.year])
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(pmv_txt.read_text(), flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
