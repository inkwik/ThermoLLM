#!/usr/bin/env python3
"""Train the five-member delta ensemble on the three-climate 12/3/3 split."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

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
PMV_VARS = [f"new_pmv_space{i}" for i in ZONE]
ACTION_LOW = np.array([12.0, 23.25] * 5, dtype=np.float32)
ACTION_HIGH = np.array([23.25, 30.0] * 5, dtype=np.float32)
PMV_MAX, ENERGY_KWH_MAX = 4.0, 4.5
RHO_OCC, RHO_UNOCC = 1.1, 0.1


def reward_from_state(state: np.ndarray, context: np.ndarray) -> np.ndarray:
    """Exact TruePMV reward formula used by PPO, applied row-wise."""
    # State: 5 temps, 5 RH, 5 custom PMVs, HVAC joules. Context has weather(6),
    # occupancy(5), then six time encodings.
    pmv = state[:, 10:15]
    occ = context[:, 6:11]
    norm_pmv = np.clip(np.maximum(np.abs(pmv) - 0.5, 0.0) / PMV_MAX, 0.0, 1.0)
    rho = np.where(occ > 0, RHO_OCC, RHO_UNOCC)
    energy_kwh = state[:, 15] / 3.6e6
    norm_energy = np.clip(energy_kwh / ENERGY_KWH_MAX, 0.0, 1.0)
    return -(rho * norm_pmv).sum(axis=1) - norm_energy


def train_and_validate(dataset_root: Path, out_dir: Path, epochs: int, batch_size: int,
                       requested_device: str) -> None:
    import torch
    from torch import nn

    torch.set_num_threads(1)
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested_device)
    manifest_path = dataset_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    metadata = json.loads(manifest_path.read_text())
    split = metadata.get("splits", {})
    train_files = [dataset_root / row["episode"] for row in split.get("train", [])]
    val_files = [dataset_root / row["episode"] for row in split.get("validation", [])]
    test_files = [dataset_root / row["episode"] for row in split.get("test", [])]
    if len(train_files) != 12 or len(val_files) != 3 or len(test_files) != 3:
        raise RuntimeError("Expected 12 train, 3 validation, and 3 untouched test trajectories")
    missing = [str(path) for path in train_files + val_files + test_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset files are missing: {missing[:5]}")
    train_arrays = [np.load(path) for path in train_files]
    val_arrays = [np.load(path) for path in val_files]
    X = np.concatenate([a["x"] for a in train_arrays]).astype(np.float32)
    Y = np.concatenate([a["delta"] for a in train_arrays]).astype(np.float32)
    VX = np.concatenate([a["x"] for a in val_arrays]).astype(np.float32)
    VY = np.concatenate([a["delta"] for a in val_arrays]).astype(np.float32)
    val_current = np.concatenate([a["current"] for a in val_arrays]).astype(np.float32)
    val_next = np.concatenate([a["next"] for a in val_arrays]).astype(np.float32)
    if X.shape[1] != 43 or Y.shape[1] != 16 or VX.shape[1] != 43 or VY.shape[1] != 16:
        raise RuntimeError(f"Unexpected feature/target dimensions: X={X.shape}, Y={Y.shape}")
    x_mean, x_std = X.mean(0), X.std(0).clip(1e-6)
    y_mean, y_std = Y.mean(0), Y.std(0).clip(1e-6)
    Xn, Yn, XVn = (X - x_mean) / x_std, (Y - y_mean) / y_std, (VX - x_mean) / x_std
    xt, yt, xvt = [torch.as_tensor(a, dtype=torch.float32, device=device) for a in (Xn, Yn, XVn)]
    class Member(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(X.shape[1], 200), nn.ReLU(),
                                     nn.Linear(200, 200), nn.ReLU(), nn.Linear(200, Y.shape[1]))
        def forward(self, x): return self.net(x)

    ensemble = []
    history = []
    rng = np.random.default_rng(7301)
    for m in range(5):
        torch.manual_seed(7301 + m)
        model = Member().to(device)
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        bootstrap = rng.integers(0, len(xt), size=len(xt))
        indices = torch.as_tensor(bootstrap, dtype=torch.long, device=device)
        for epoch in range(epochs):
            perm = indices[torch.randperm(len(indices), device=device)]
            model.train()
            for start in range(0, len(perm), batch_size):
                ix = perm[start:start + batch_size]
                loss = ((model(xt[ix]) - yt[ix]) ** 2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pred = model(xvt).cpu().numpy() * y_std + y_mean
                val_mse = float(np.mean((pred - VY) ** 2))
            history.append({"member": m, "epoch": epoch + 1, "validation_delta_mse": val_mse})
        ensemble.append(model)
        print(f"Trained ensemble member {m + 1}/5 on {device}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    member_predictions = []
    for model in ensemble:
        model.eval()
        with torch.no_grad():
            member_predictions.append(model(xvt).cpu().numpy() * y_std + y_mean)
    member_predictions = np.asarray(member_predictions)
    # MB2C-inspired online ensemble reweighting: recent C=4 standardized
    # prediction errors set the target weights; phi=.9 smooths updates. Each
    # prediction is made before observing that transition's target, so the
    # held-out evaluation remains prequential and does not leak future targets.
    weights = np.full(len(ensemble), 1.0 / len(ensemble), dtype=np.float64)
    recent_errors: list[np.ndarray] = []
    weighted_deltas = []
    weight_trace = []
    for row in range(len(VX)):
        preds = member_predictions[:, row, :]
        weighted_deltas.append(np.einsum("m,md->d", weights, preds))
        weight_trace.append(weights.copy())
        normalized_error = np.mean(((preds - VY[row]) / y_std) ** 2, axis=1)
        recent_errors.append(normalized_error)
        recent_errors = recent_errors[-4:]
        recent_mean = np.mean(recent_errors, axis=0)
        scores = -recent_mean
        scores -= scores.max()
        target_weights = np.exp(scores)
        target_weights /= target_weights.sum()
        weights = 0.9 * weights + 0.1 * target_weights
        weights /= weights.sum()
    delta_hat = np.asarray(weighted_deltas)
    next_hat = val_current + delta_hat
    next_actual = val_next
    # Evaluate recursively one-step using actual observed prior states (teacher-forced
    # one-step MB2C validation) and the exact PPO reward on predicted state/context.
    feature_names = BUILDING_VARS
    mae = np.mean(np.abs(next_hat - next_actual), axis=0)
    rmse = np.sqrt(np.mean((next_hat - next_actual) ** 2, axis=0))
    ctx_val = VX[:, len(BUILDING_VARS) + 10:]
    r_actual = reward_from_state(next_actual, ctx_val)
    r_pred = reward_from_state(next_hat, ctx_val)
    report = {
        "method": "MB2C-style five-member environment-conditioned delta ensemble with online error reweighting",
        "environment": metadata.get("climates"), "runperiod": metadata.get("runperiod"),
        "timesteps_per_hour": TIMESTEPS_PER_HOUR,
        "train_episodes": [path.relative_to(dataset_root).as_posix() for path in train_files],
        "device": str(device), "n_train_transitions": int(len(X)), "n_validation_transitions": int(len(VX)),
        "architecture": "5 bootstrap ensemble members; each 200-ReLU-200-ReLU output MLP",
        "initialization": "Xavier uniform weights; zero biases",
        "ensemble_update": "recent window C=4, exponential weight smoothing phi=0.9, prequential validation updates",
        "final_ensemble_weights": [float(v) for v in weights],
        "epochs_per_member": epochs, "optimizer": "Adam lr=0.001", "batch_size": batch_size,
        "validation_one_step": {name: {"mae": float(a), "rmse": float(r)}
                                 for name, a, r in zip(feature_names, mae, rmse)},
        "custom_pmv_mae_by_zone": {f"new_pmv_space{i+1}": float(mae[10+i]) for i in range(5)},
        "reward_formula": "-sum_z(rho_z*clip(max(abs(new_pmv_z)-0.5,0)/4,0,1))-clip((HVAC_J/3.6e6)/4.5,0,1); rho=1.1 occupied else 0.1",
        "reward_mae": float(np.mean(np.abs(r_pred - r_actual))),
        "reward_rmse": float(np.sqrt(np.mean((r_pred - r_actual) ** 2))),
        "actual_mean_total_hvac_kwh_per_step": float(np.mean(next_actual[:, 15]) / 3.6e6),
        "predicted_mean_total_hvac_kwh_per_step": float(np.mean(next_hat[:, 15]) / 3.6e6),
        "ppo_normalization": {"pmv_max": PMV_MAX, "energy_kwh_max": ENERGY_KWH_MAX,
                              "rho_occupied": RHO_OCC, "rho_unoccupied": RHO_UNOCC,
                              "energy_variable": "total_electricity_HVAC", "pmv_variables": PMV_VARS},
    }
    torch.save({"models": [m.cpu().state_dict() for m in ensemble], "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std, "building_vars": BUILDING_VARS,
                "weather_vars": WEATHER_VARS, "occupancy_vars": OCC_VARS,
                "action_low": ACTION_LOW, "action_high": ACTION_HIGH}, out_dir / "mb2c_5zone_ensemble.pt")
    np.savez_compressed(out_dir / "heldout_predictions.npz", actual=next_actual, predicted=next_hat,
                        reward_actual=r_actual, reward_predicted=r_pred,
                        ensemble_weights=np.asarray(weight_trace))
    report["validation_episodes"] = [path.relative_to(dataset_root).as_posix() for path in val_files]
    report["test_episodes_reserved"] = [path.relative_to(dataset_root).as_posix() for path in test_files]
    report["experiment_metadata"] = metadata
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    train_and_validate(args.dataset_root.resolve(), args.output_dir.resolve(),
                       args.epochs, args.batch_size, args.device)


if __name__ == "__main__":
    main()
