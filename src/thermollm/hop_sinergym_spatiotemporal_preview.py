# Build Brick spatiotemporal context from 5zone building config for HVAC control.

from __future__ import annotations

import os
import re
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from rdflib import BNode, Graph, URIRef

import hop
import sinergym  # noqa: F401
from sinergym.utils.wrappers import CSVLogger, LoggerWrapper

TTL_CANDIDATES = [
    Path("/mnt/kb/sinergym/data/buildings/5ZoneAutoDXVAVnw_brick.ttl"),
]

ENV_ID = os.environ.get(
    "BRICK_PREVIEW_ENV_ID",
    "Eplus-5zone-zonal-arizona-continuous-v1",
)
PMV_SOURCE = os.environ.get("PMV_SOURCE", "custom").strip().lower()
if PMV_SOURCE not in {"custom", "energyplus"}:
    raise ValueError("PMV_SOURCE must be 'custom' or 'energyplus'")
BUILDING_CONFIG = {
    "timesteps_per_hour": 4,
    "runperiod": (1, 6, 1991, 7, 6, 1991),
}
MAX_PREVIEW_STEPS = int(os.environ.get("MAX_PREVIEW_STEPS", "5"))
MAX_HISTORY = 10
STEP_HOURS = 1.0 / BUILDING_CONFIG["timesteps_per_hour"]
SUMMER_MONTHS = {6, 7, 8, 9}
ZONE_NAMES = ["SPACE1-1", "SPACE2-1", "SPACE3-1", "SPACE4-1", "SPACE5-1"]

WINTER_OCC_HEAT = 20.0
WINTER_OCC_COOL = 23.5
SUMMER_OCC_HEAT = 23.0
SUMMER_OCC_COOL = 26.0
UNOCC_HEAT = 12.0
UNOCC_COOL = 30.0

GLOBAL_REF_TO_OBS = {
    "sinergym:outdoor_temperature": "outdoor_temperature",
    "sinergym:outdoor_humidity": "outdoor_humidity",
    "sinergym:direct_solar_radiation": "direct_solar_radiation",
    "sinergym:wind_speed": "wind_speed",
}


def resolve_ttl_path() -> Path:
    ttl_path = next((path for path in TTL_CANDIDATES if path.exists()), None)
    if ttl_path is None:
        raise FileNotFoundError(
            "Brick TTL file not found. Checked:\n"
            + "\n".join(str(path) for path in TTL_CANDIDATES)
        )
    return ttl_path


def zone_index(zone_name: str) -> int:
    match = re.match(r"SPACE(\d+)-\d+", zone_name)
    if not match:
        raise ValueError(f"Unexpected zone name format: {zone_name}")
    return int(match.group(1))


def build_obs_index_map(env: gym.Env) -> dict[str, int]:
    obs_names = list(getattr(env.unwrapped, "observation_variables", []))
    return {name: i for i, name in enumerate(obs_names)}


def obs_to_dict(obs: np.ndarray, obs_idx: dict[str, int]) -> dict[str, float]:
    return {name: float(obs[idx]) for name, idx in obs_idx.items()}


def setpoints_for_zone(month: int, occupancy_value: float) -> tuple[float, float]:
    if occupancy_value > 0:
        if month in SUMMER_MONTHS:
            return SUMMER_OCC_HEAT, SUMMER_OCC_COOL
        return WINTER_OCC_HEAT, WINTER_OCC_COOL
    return UNOCC_HEAT, UNOCC_COOL


def pmv_observation_name(zi: int) -> str:
    """Return the same PMV observation selected by controller.py."""
    prefix = "new_pmv_space" if PMV_SOURCE == "custom" else "pmv_space"
    return f"{prefix}{zi}"


def build_action(obs: np.ndarray, obs_idx: dict[str, int]) -> np.ndarray:
    month = int(obs[obs_idx["month"]])
    action_values: list[float] = []
    for zone_name in ZONE_NAMES:
        zi = zone_index(zone_name)
        occ = float(obs[obs_idx[f"air_occ_space{zi}"]])
        heat_sp, cool_sp = setpoints_for_zone(month, occ)
        action_values.extend([heat_sp, cool_sp])
    return np.array(action_values, dtype=np.float32)


def extract_hvac_energy_kwh(obs_dict: dict[str, float]) -> float | None:
    if "HVAC_electricity_demand_rate" in obs_dict:
        watts = obs_dict.get("HVAC_electricity_demand_rate")
        if watts is not None:
            return float(watts) * STEP_HOURS / 1000.0
    if "total_electricity_HVAC" in obs_dict:
        joules = obs_dict.get("total_electricity_HVAC")
        if joules is not None:
            return float(joules) / 3_600_000.0
    return None


def current_timeseries_values(obs_dict: dict[str, float]) -> dict[str, float | None]:
    current: dict[str, float | None] = {}

    for ref_name, obs_name in GLOBAL_REF_TO_OBS.items():
        current[ref_name] = obs_dict.get(obs_name)

    current["sinergym:total_hvac_energy_kwh"] = extract_hvac_energy_kwh(obs_dict)

    for zone_name in ZONE_NAMES:
        zi = zone_index(zone_name)
        current[f"sinergym:air_temperature_space{zi}"] = obs_dict.get(
            f"air_temperature_space{zi}"
        )
        current[f"sinergym:air_humidity_space{zi}"] = obs_dict.get(
            f"air_humidity_space{zi}"
        )
        occ_value = obs_dict.get(f"air_occ_space{zi}")
        current[f"sinergym:air_occ_space{zi}"] = 1.0 if (occ_value or 0.0) > 0.0 else 0.0
        # Keep the Brick reference key from the TTL, but source its value from
        # the controller-selected PMV observation (custom by default).
        current[f"sinergym:pmv_space{zi}"] = obs_dict.get(pmv_observation_name(zi))
        current[f"sinergym:damper_position_space{zi}"] = obs_dict.get(
            f"damper_position_space{zi}"
        )
        current[f"sinergym:htg_setpoint_space{zi}"] = obs_dict.get(
            f"htg_setpoint_space{zi}"
        )
        current[f"sinergym:clg_setpoint_space{zi}"] = obs_dict.get(
            f"clg_setpoint_space{zi}"
        )

    return current


def season_from_month(month: int) -> str:
    return "summer" if month in SUMMER_MONTHS else "winter"


def timestamp_parts(obs_dict: dict[str, float], step_number: int) -> tuple[int, int, int, int]:
    month = int(obs_dict.get("month", 1) or 1)
    day = int(obs_dict.get("day_of_month", 1) or 1)
    hour = int(obs_dict.get("hour", 0) or 0)
    slot = step_number % BUILDING_CONFIG["timesteps_per_hour"]
    minute = int(slot * 60 / BUILDING_CONFIG["timesteps_per_hour"])
    return month, day, hour, minute


def current_time_label(obs_dict: dict[str, float], step_number: int) -> str:
    month, day, hour, minute = timestamp_parts(obs_dict, step_number)
    season = season_from_month(month)
    return f"{month:02d}/{day:02d}/{hour:02d}/{minute:02d}_{season}"


def history_time_label(obs_dict: dict[str, float], step_number: int) -> str:
    month, day, hour, minute = timestamp_parts(obs_dict, step_number)
    return f"{month:02d}/{day:02d}/{hour:02d}/{minute:02d}"


def scalar_text(value: Any) -> str:
    if value is None:
        return "not supplied"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def history_text(values: list[Any]) -> str:
    return "[" + ", ".join(scalar_text(value) for value in values) + "]"


def current_time_without_season(current_label: str | None) -> str:
    if current_label is None:
        return "unknown"
    return current_label.split("_", 1)[0]


def point_value_key(
    point_label: str,
    point_types: list[str],
    reference: str | None,
    current_step: int | None,
    current_label: str | None,
    history_meta: list[tuple[int, str]],
) -> str:
    step_text = f"step{current_step}" if current_step is not None else "step?"
    time_text = current_label or "unknown"
    type_text = " ".join(point_types).lower()
    label_text = point_label.lower()

    if reference == "sinergym:total_hvac_energy_kwh" or "electrical energy usage sensor" in type_text:
        if current_step is None or current_step <= 0 or not history_meta:
            return f"hvac_energy_before_first_agent_action_time:{time_text}"
        prev_step, prev_time = history_meta[-1]
        return (
            "hvac_energy_consumed_for_action_applied_at_previous_"
            f"step{prev_step}_time:{prev_time}-{current_time_without_season(current_label)}"
        )

    if "setpoint" in type_text or "setpoint" in label_text:
        return f"active_setpoint_before_{step_text}_action_time:{time_text}"
    if "zone air temperature sensor" in type_text:
        return f"zone_temperature_before_{step_text}_action_time:{time_text}"
    if "zone air humidity sensor" in type_text:
        return f"zone_humidity_before_{step_text}_action_time:{time_text}"
    if "occupancy sensor" in type_text:
        return f"occupancy_at_{step_text}_time:{time_text}"
    if "damper position sensor" in type_text:
        return f"damper_position_before_{step_text}_action_time:{time_text}"
    if "outside air temperature sensor" in type_text:
        return f"outdoor_temperature_at_{step_text}_time:{time_text}"
    if "outside air humidity sensor" in type_text:
        return f"outdoor_humidity_at_{step_text}_time:{time_text}"
    if "solar irradiance sensor" in type_text:
        return f"solar_radiation_at_{step_text}_time:{time_text}"
    if "wind speed sensor" in type_text:
        return f"wind_speed_at_{step_text}_time:{time_text}"
    if reference and reference.startswith("sinergym:pmv_space"):
        zi = int(reference.rsplit("space", 1)[-1])
        return f"{pmv_observation_name(zi)}_before_{step_text}_action_time:{time_text}"

    return f"observed_before_{step_text}_action_time:{time_text}"


def point_history_key(
    point_label: str,
    point_types: list[str],
    reference: str | None,
    history_meta: list[tuple[int, str]],
) -> str:
    if not history_meta:
        return "history_stepnone_timeempty"

    start_step, start_time = history_meta[0]
    end_step, end_time = history_meta[-1]
    type_text = " ".join(point_types).lower()
    label_text = point_label.lower()
    time_range = f"step{start_step}-{end_step}_time{start_time}-{end_time}"

    if reference == "sinergym:total_hvac_energy_kwh" or "electrical energy usage sensor" in type_text:
        return f"hvac_energy_history_for_previous_actions_{time_range}"
    if "setpoint" in type_text or "setpoint" in label_text:
        return f"active_setpoint_history_from_actions_{time_range}"
    if reference and reference.startswith("sinergym:pmv_space"):
        zi = int(reference.rsplit("space", 1)[-1])
        return f"{pmv_observation_name(zi)}_history_{time_range}"
    return f"observed_history_{time_range}"


def runtime_payload_for_step(
    current_by_ref: dict[str, float | None],
    history_by_ref: dict[str, deque[float | None]],
    step_number: int,
    current_label: str,
    history_meta: list[tuple[int, str]],
) -> dict[str, dict[str, Any]]:
    runtime_data: dict[str, dict[str, Any]] = {}
    for ref_name, current_value in current_by_ref.items():
        runtime_data[ref_name] = {
            "current": current_value,
            "current_step": step_number,
            "current_time_label": current_label,
            "history": list(history_by_ref.get(ref_name, []))[-MAX_HISTORY:],
            "history_meta": history_meta[-MAX_HISTORY:],
        }
    return runtime_data


def point_details_with_step_context(
    graph: Graph,
    point: URIRef | BNode,
    runtime_data: dict[str, dict[str, Any]],
    indent: int = 4,
    is_setpoint: bool = False,
) -> list[str]:
    pad = " " * indent
    child_pad = " " * (indent + 2)

    types = hop.rdf_types(graph, point)
    point_type = ", ".join(types) if types else "not available"
    reference = hop.timeseries_reference(graph, point)
    runtime = hop.runtime_for(reference, runtime_data)
    current_value = runtime.get("current", "not supplied")
    current_step = runtime.get("current_step")
    current_label = runtime.get("current_time_label")
    history_values = list(runtime.get("history", []))
    history_meta = list(runtime.get("history_meta", []))
    point_label = hop.label(graph, point)

    lines = [f"{pad}Point: {point_label}"]
    lines.append(f"{child_pad}type: {point_type}")

    value_key = point_value_key(
        point_label=point_label,
        point_types=types,
        reference=reference,
        current_step=current_step,
        current_label=current_label,
        history_meta=history_meta,
    )
    lines.append(f"{child_pad}{value_key}: {scalar_text(current_value)}")

    history_key = point_history_key(
        point_label=point_label,
        point_types=types,
        reference=reference,
        history_meta=history_meta,
    )
    lines.append(f"{child_pad}{history_key}: {history_text(history_values)}")

    return lines


def build_clean_context(
    graph: Graph,
    runtime_data: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "=" * 78,
        "BUILDING SPATIO-TEMPORAL CONTROL CONTEXT",
        "=" * 78,
        "",
    ]
    lines.extend(clean_weather_section(graph, runtime_data))
    lines.extend(clean_central_hvac_section(graph, runtime_data))
    lines.extend(hop.zone_branches_section(graph, runtime_data))
    lines.extend(hop.return_air_section(graph))
    return "\n".join(lines).rstrip() + "\n"


def clean_weather_section(
    graph: Graph,
    runtime_data: dict[str, dict[str, Any]],
) -> list[str]:
    lines = ["[WEATHER AND OUTDOOR CONDITIONS]", ""]
    for station in hop.sorted_nodes(
        graph,
        graph.subjects(hop.RDF.type, hop.BRICK.Weather_Station),
    ):
        lines.append(f"Weather Station: {hop.label(graph, station)}")
        points = [
            point
            for point in graph.objects(station, hop.BRICK.hasPoint)
            if isinstance(point, (URIRef, BNode))
        ]
        lines.extend(hop.grouped_points(graph, points, runtime_data))
        lines.append("")
    return lines


def clean_central_hvac_section(
    graph: Graph,
    runtime_data: dict[str, dict[str, Any]],
) -> list[str]:
    lines = hop.central_hvac_section(graph, runtime_data)
    return [
        line
        for line in lines
        if not line.strip().startswith("combinedElectricalEnergyOf ->")
    ]


def main() -> None:
    ttl_path = resolve_ttl_path()
    hop.TTL_FILE = ttl_path
    hop.point_details = point_details_with_step_context

    graph = Graph()
    graph.parse(ttl_path, format="turtle")

    run_name = f"hop_spatiotemporal_preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    env = gym.make(ENV_ID, env_name=run_name, building_config=BUILDING_CONFIG)
    env = LoggerWrapper(env)
    env = CSVLogger(env)

    history_by_ref: dict[str, deque[float | None]] = defaultdict(
        lambda: deque(maxlen=MAX_HISTORY)
    )
    history_meta: deque[tuple[int, str]] = deque(maxlen=MAX_HISTORY)

    try:
        obs, _info = env.reset(seed=0)
        obs_idx = build_obs_index_map(env)

        print(f"TTL file: {ttl_path}")
        print(f"Environment: {ENV_ID}")
        print(f"Workspace: {env.get_wrapper_attr('workspace_path')}")
        print(f"Observation variables: {len(obs_idx)}")
        print("")

        step = 0
        done = False
        while not done and step < MAX_PREVIEW_STEPS:
            obs_dict = obs_to_dict(obs, obs_idx)
            current_by_ref = current_timeseries_values(obs_dict)
            current_label = current_time_label(obs_dict, step)
            history_meta_snapshot = list(history_meta)
            runtime_data = runtime_payload_for_step(
                current_by_ref=current_by_ref,
                history_by_ref=history_by_ref,
                step_number=step,
                current_label=current_label,
                history_meta=history_meta_snapshot,
            )

            print("=" * 120)
            print(f"STEP {step} | current_step{step}_time:{current_label}")
            print("")
            print(build_clean_context(graph, runtime_data))

            for ref_name, value in current_by_ref.items():
                history_by_ref[ref_name].append(value)
            history_meta.append((step, history_time_label(obs_dict, step)))

            action = build_action(obs, obs_idx)
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            step += 1
    finally:
        env.close()


if __name__ == "__main__":
    main()
