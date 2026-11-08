#!/usr/bin/env python3
"""
Single-agent LLM HVAC controller using Brick spatio-temporal context.

This script uses the same retrieval style as `hop_sinergym_spatiotemporal_preview.py`.
At every Sinergym decision step it builds one LLM-ready context containing:
  - Static Brick topology and zone spatial context from the TTL.
  - Current Sinergym values attached to Brick points by timeseries reference.
  - Rolling prior-step history, capped at 10 steps.

The LLM chooses five zone heating/cooling setpoint pairs. CSVs, prompt I/O,
Sinergym output folders, metrics, and MLflow artifacts are flushed in `finally`
so completed data is preserved on interruption.

Run the full configured run period without the LLM API:
    USE_LLM_API=0 python "1 ag kg gpt 5z custom pmv.py"
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from rdflib import Graph as RDFGraph

import hop_sinergym_spatiotemporal_preview as brick_context
import sinergym  # noqa: F401
from pmv_energy_metrics import (
    PMV_LOWER,
    PMV_UPPER,
    compute_performance_metrics,
    format_metrics_report,
    runperiod_to_date_range,
    save_pmv_summary_artifacts,
)
from sinergym.utils.wrappers import CSVLogger, LoggerWrapper

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    MPL_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore[assignment]
    MPL_AVAILABLE = False


# =============================================================================
# CONFIG: simulation, action bounds, outputs, API, and tracking.
# =============================================================================

ENV_ID = "Eplus-5zone-zonal-arizona-continuous-v1"
PMV_SOURCE = os.environ.get("PMV_SOURCE", "custom").strip().lower()  # "custom" or "energyplus"
if PMV_SOURCE not in {"custom", "energyplus"}:
    raise ValueError("PMV_SOURCE must be 'custom' or 'energyplus'")
BUILDING_CONFIG = {
    "timesteps_per_hour": 4,
    "runperiod": (1, 6, 1991, 7, 6, 1991),
}


def control_steps_from_runperiod(building_config: Dict[str, Any]) -> int:
    """Return the inclusive number of 15-minute control intervals in runperiod."""
    runperiod = building_config["runperiod"]
    if len(runperiod) != 6:
        raise ValueError("runperiod must be (start_day, start_month, start_year, end_day, end_month, end_year)")
    start_day, start_month, start_year, end_day, end_month, end_year = map(int, runperiod)
    start_date = datetime(start_year, start_month, start_day)
    end_date = datetime(end_year, end_month, end_day)
    if end_date < start_date:
        raise ValueError(f"runperiod end precedes start: {runperiod}")
    timesteps_per_hour = int(building_config.get("timesteps_per_hour", 4))
    if timesteps_per_hour <= 0:
        raise ValueError(f"timesteps_per_hour must be positive, got {timesteps_per_hour}")
    inclusive_days = (end_date - start_date).days + 1
    return inclusive_days * 24 * timesteps_per_hour


TOTAL_CONTROL_STEPS = control_steps_from_runperiod(BUILDING_CONFIG)
STEP_DURATION_MIN = 15
STEP_HOURS = STEP_DURATION_MIN / 60.0
MAX_HISTORY = 10
PLAN_HORIZON = 5  # Current step plus the next four 15-minute steps.
SUMMER_MONTHS = {6, 7, 8, 9}
ZONE_NAMES = ["SPACE1-1", "SPACE2-1", "SPACE3-1", "SPACE4-1", "SPACE5-1"]

HSP_MIN, HSP_MAX = 12.0, 23.25
CSP_MIN, CSP_MAX = 23.25, 30.0
RELAXED_HSP, RELAXED_CSP = 12.0, 30.0
WINTER_COMFORT = (20.0, 23.5)
SUMMER_COMFORT = (23.0, 26.0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("AGENT_OPENAI_MODEL", "gpt-6-luna")
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_TAG = re.sub(r"[^A-Za-z0-9_-]+", "_", MODEL)
ENV_NAME = f"{MODEL_TAG}_Arizona_sumWEEK_JUNE7days_BRICK_5zone_VAV_PMV_5stepPLAN_{RUN_TS}"
SAFE_ENV_NAME = re.sub(r"[^A-Za-z0-9_-]+", "_", ENV_NAME).strip("_")
OUT_DIR = os.path.join(SCRIPT_DIR, f"llm_custom_CSV_{SAFE_ENV_NAME}")
METRICS_DIR = os.path.join(OUT_DIR, "metrics")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
PROMPTS_DIR = os.path.join(OUT_DIR, "prompts")
for _dir in (OUT_DIR, METRICS_DIR, PLOTS_DIR, PROMPTS_DIR):
    os.makedirs(_dir, exist_ok=True)

STEP_LOG_CSV = os.path.join(OUT_DIR, "step_log.csv")
GRAPH_RETRIEVAL_CSV = os.path.join(OUT_DIR, "graph_retrieval_log.csv")
PROMPT_IO_JSONL = os.path.join(PROMPTS_DIR, "llm_prompt_io.jsonl")
PROMPT_RESPONSE_CSV = os.path.join(PROMPTS_DIR, "llm_prompt_response.csv")
ACTION_SOURCE_CSV = os.path.join(OUT_DIR, "action_source_log.csv")
OPERATIONAL_STEPS_CSV = os.path.join(METRICS_DIR, "llm_operational_steps.csv")
OPERATIONAL_SUMMARY_CSV = os.path.join(METRICS_DIR, "llm_operational_summary.csv")
PERFORMANCE_JSON = os.path.join(METRICS_DIR, "performance_metrics.json")
PMV_SUMMARY_CSV = os.path.join(METRICS_DIR, "pmv_summary_table.csv")
PMV_SUMMARY_TXT = os.path.join(METRICS_DIR, "pmv_summary_report.txt")

# API credentials are read from the environment and must never be committed.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# Model-independent settings. Set AGENT_REASONING_EFFORT='' to omit this
# parameter for APIs/models that do not support reasoning effort.
TEMPERATURE = float(os.environ.get("AGENT_TEMPERATURE", "0.0"))
_reasoning_effort = os.environ.get("AGENT_REASONING_EFFORT", "none").strip().lower()
REASONING_EFFORT = _reasoning_effort or None
MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "9000"))
USE_LLM_API = os.environ.get("USE_LLM_API", "1").strip().lower() in ("1", "true", "yes")
DEBUG_PRINT = os.environ.get("DEBUG_PRINT", "1").strip().lower() in ("1", "true", "yes")

MLFLOW_ENABLED = os.environ.get("MLFLOW_ENABLED", "1").strip().lower() in ("1", "true", "yes")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "hvac-llm-brick-5zone-pmv")
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{os.path.join(SCRIPT_DIR, 'mlflow_5zone_pmv.db')}"
)
MLFLOW_ARTIFACT_DIR = os.path.join(SCRIPT_DIR, "mlflow_5zone_pmv_artifacts")
PROMPT_VERSION_TAG = os.environ.get("PROMPT_VERSION_TAG", "brick_spatiotemporal_pmv_v3_strict_output_tracking")

OPENAI_INPUT_PRICE_PER_1M = float(os.environ.get("OPENAI_INPUT_PRICE_PER_1M", "0.10"))
OPENAI_OUTPUT_PRICE_PER_1M = float(os.environ.get("OPENAI_OUTPUT_PRICE_PER_1M", "0.50"))


# =============================================================================
# PROMPTS: system role, units, context meaning, and strict output contract.
# =============================================================================

SYSTEM_PROMPT = f"""
## 1. ROLE
You are an expert thermodynamic analyst and constraint-aware HVAC control engineer specializing in energy-efficient thermal comfort control of a five-zone VAV office building.

## 2. CONTROL GOAL
Maintain PMV thermal comfort in occupied zones within the PMV range [{PMV_LOWER}, {PMV_UPPER}] while saving maximum HVAC energy.

## 3. BUILDING AND OBSERVATION CONTEXT
The user prompt provides Brick spatial/structural context, current observations, and up to 10 previous 15-minute steps for zones SPACE1-1 through SPACE5-1.

Physical and comfort principles:
- PMV confort range: [{PMV_LOWER}, {PMV_UPPER}]. PMV outside this range indicates discomfort.
- PMV is a thermal comfort index: negative values indicate cooler sensation, positive values indicate warmer sensation. PMV outside the specified range indicates discomfort.
- The formula for pmv is based on ISO 7730-2005 Fanger and here PMV calculation is function of zone air temperature, mean radiant temperature (tr), relative humidity (RH), air velocity (vr), metabolic rate (met), and clothing insulation (clo). 
- In PMV caculation formula assumed values: mean radiant temperature = zone air temperature, RH=40%, vr=0.15 m/s, met=1.1, clo=0.5 in June–September(summer months) and 1.0 otherwise (winter months).
- PMV is evaluated and matters only for occupied zones.
- Zones share central HVAC equipment through the VAV system. Consider zone-specific thermal behavior and its implications for total HVAC energy.
- Use relevant Brick geometry, adjacency, orientation, weather, and history to assess heat transfer, solar effects, and thermal inertia. Azimuth is measured clockwise from north (0° north, 90° east, 180° south, 270° west).

Observation timing:
- Observations at step X are available before choosing action X.
- Active setpoints, when provided, represent the controls in effect before action X, normally from the previous step.
- HVAC energy observations describe the preceding interval, not the proposed action. Account for delayed thermal response and other disturbances when interpreting past energy consumption.

## 4. ACTION CONSTRAINTS
For EACH zone:
- Heating setpoint (HSP): [{HSP_MIN}, {HSP_MAX}]°C.
- Cooling setpoint (CSP): [{CSP_MIN}, {CSP_MAX}]°C.
- Require HSP < CSP.
- Each action pair must be [HSP, CSP].

Control principles:
- Raising HSP generally increases heating demand.
- Lowering CSP generally increases cooling demand.
- Setpoints are control triggers, not guaranteed zone-temperature limits.
- Relaxed setpoints [{RELAXED_HSP}, {RELAXED_CSP}] means hvac inactive, allowing free thermal drift.

## 5. CONTROL DECISION RULES
1. THERMODYNAMIC ANALYSIS:
   Analyze each zone using current observations, historical trends, previous actions, occupancy, PMV, weather, and relevant Brick spatial context. Identify thermal behavior, comfort risks, and energy implications.
   Use spatio-temporal context to reason about heat transfer, thermal inertia, and zone behavior based on applied setpoints, indoor and outdoor conditions.

2. PREDICTIVE PLANNING:
   After doing proper thermodynamics and indoor outdoor conditions reasoning and analysis, think of a action setpoints and plan that aligns with control goals and constraints.
   Develop a rolling five-step HSP/CSP plan for each zone, considering expected temperature evolution, thermal inertia, HVAC response, occupancy, comfort, and energy consumption over the next 5 steps. Use physically plausible estimates without inventing measurements or unsupported numerical predictions.
   This rolling five-step HSP/CSP plan actually helps you to make future impact aware control decisions.

3. COMFORT-ENERGY OPTIMIZATION:
   Among sufficiently safe options, prefer the actions with the lowest expected HVAC energy use while maintaining PMV within -0.5 to 0.5 range for occupied zones.
   Avoid unnecessary heating, cooling, and excessive conditioning.

4. CONTROL STABILITY:
   Avoid frequent or unnecessary setpoint changes. Maintain stable actions when appropriate, but make timely adjustments when required. 

5. ROLLING-HORIZON EXECUTION:
   Only the first action pair of each five-step plan is applied. The remaining four actions are provisional and will be reconsidered at the next timestep.

6. ACTION JUSTIFICATION:
   Briefly explain the thermodynamic evidence, expected temperature response, safety considerations, and energy rationale for the selected actions and five-step plans of each zone.

## 6. OUTPUT FORMAT
Return ONLY a valid JSON object with the exact schema below. Write the five-step action plan FIRST and the zone-wise justification SECOND.
All setpoints must be numeric and satisfy the action constraints. Provide exactly five [HSP, CSP] pairs for each zone.

{{
  "five_step_setpoint_plan": {{
    "SPACE1-1": [
      [heat_X, cool_X],
      [heat_Xplus1, cool_Xplus1],
      [heat_Xplus2, cool_Xplus2],
      [heat_Xplus3, cool_Xplus3],
      [heat_Xplus4, cool_Xplus4]
    ],
    "SPACE2-1": [
      [heat_X, cool_X],
      [heat_Xplus1, cool_Xplus1],
      [heat_Xplus2, cool_Xplus2],
      [heat_Xplus3, cool_Xplus3],
      [heat_Xplus4, cool_Xplus4]
    ],
    "SPACE3-1": [
      [heat_X, cool_X],
      [heat_Xplus1, cool_Xplus1],
      [heat_Xplus2, cool_Xplus2],
      [heat_Xplus3, cool_Xplus3],
      [heat_Xplus4, cool_Xplus4]
    ],
    "SPACE4-1": [
      [heat_X, cool_X],
      [heat_Xplus1, cool_Xplus1],
      [heat_Xplus2, cool_Xplus2],
      [heat_Xplus3, cool_Xplus3],
      [heat_Xplus4, cool_Xplus4]
    ],
    "SPACE5-1": [
      [heat_X, cool_X],
      [heat_Xplus1, cool_Xplus1],
      [heat_Xplus2, cool_Xplus2],
      [heat_Xplus3, cool_Xplus3],
      [heat_Xplus4, cool_Xplus4]
    ]
  }},
  "zonewise_thermoanalysis_and_goal_aligned_action_justification": {{
    "SPACE1-1": "Concise thermodynamic analysis and action justification.",
    "SPACE2-1": "Concise thermodynamic analysis and action justification.",
    "SPACE3-1": "Concise thermodynamic analysis and action justification.",
    "SPACE4-1": "Concise thermodynamic analysis and action justification.",
    "SPACE5-1": "Concise thermodynamic analysis and action justification."
  }}
}}
""".strip()

def _zone_plan_schema() -> Dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "array",
            "description": (
                f"Exactly [HSP, CSP] in degC: {HSP_MIN} <= HSP <= {HSP_MAX}, "
                f"{CSP_MIN} <= CSP <= {CSP_MAX}, and HSP < CSP. "
                f"The pair [{HSP_MAX}, {CSP_MIN}] is invalid."
            ),
            "prefixItems": [
                {
                    "type": "number",
                    "minimum": HSP_MIN,
                    "maximum": HSP_MAX,
                    "description": f"Heating setpoint HSP in degC, from {HSP_MIN} to {HSP_MAX} inclusive.",
                },
                {
                    "type": "number",
                    "minimum": CSP_MIN,
                    "maximum": CSP_MAX,
                    "description": f"Cooling setpoint CSP in degC, from {CSP_MIN} to {CSP_MAX} inclusive.",
                },
            ],
            "minItems": 2,
            "maxItems": 2,
        },
        "minItems": PLAN_HORIZON,
        "maxItems": PLAN_HORIZON,
    }


_zone_keys = [f"SPACE{i}-1" for i in range(1, 6)]
OUTPUT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "five_step_setpoint_plan": {
            "type": "object",
            "properties": {zone: _zone_plan_schema() for zone in _zone_keys},
            "required": _zone_keys,
            "additionalProperties": False,
        },
        "zonewise_thermoanalysis_and_goal_aligned_action_justification": {
            "type": "object",
            "properties": {
                zone: {
                    "type": "string",
                    "description": "Concise zone-wise thermodynamic analysis and action justification.",
                }
                for zone in _zone_keys
            },
            "required": _zone_keys,
            "additionalProperties": False,
        },
    },
    "required": [
        "five_step_setpoint_plan",
        "zonewise_thermoanalysis_and_goal_aligned_action_justification",
    ],
    "additionalProperties": False,
}


def build_user_prompt(step: int, current_label: str, context_text: str) -> str:
    """Build the per-step user prompt that the LLM actually sees."""
    return f"""
Decision step {step}.

Use this complete spatio-temporal Brick context for the current control decision:

========================================================================================================================
STEP {step} | current_step{step}_time:{current_label}

{context_text}

Task:
- Generate a five-step rolling HSP/CSP control plan for all five zones using the provided spatio-temporal Brick context. Consider thermal behavior, future temperature trends, occupied-zone PMV comfort, and HVAC energy consumption.
- Return the complete action plan and a concise, thermodynamically grounded, goal-aligned justification for each zone in the required JSON format.
""".strip()


# =============================================================================
# SMALL HELPERS: observations, timestamps, runtime context, actions, and parsing.
# =============================================================================

def obs_to_dict(obs: np.ndarray, obs_vars: List[str]) -> Dict[str, float]:
    """Convert Sinergym observation vector into a variable-name dictionary."""
    return {name: float(obs[i]) if i < len(obs) else float("nan") for i, name in enumerate(obs_vars)}


def pmv_observation_name(zi: int) -> str:
    return f"new_pmv_space{zi}" if PMV_SOURCE == "custom" else f"pmv_space{zi}"


def observations_for_pmv_metrics(obs_df: pd.DataFrame) -> pd.DataFrame:
    """Map the selected PMV into the shared metrics schema without changing raw data."""
    selected = [pmv_observation_name(zi) for zi in range(1, 6)]
    missing = [name for name in selected if name not in obs_df.columns]
    if missing:
        raise ValueError(f"Missing {PMV_SOURCE} PMV observations: {missing}")
    metrics_df = obs_df.copy()
    for zi, name in enumerate(selected, start=1):
        metrics_df[f"pmv_space{zi}"] = obs_df[name]
    return metrics_df


def timestamp_from_obs(obs_dict: Dict[str, float], timestep: int) -> datetime:
    """Build a real timestamp from Sinergym time variables and timestep slot."""
    month = int(obs_dict.get("month", 1) or 1)
    day = int(obs_dict.get("day_of_month", 1) or 1)
    hour = int(obs_dict.get("hour", 0) or 0)
    minute = int((timestep % 4) * STEP_DURATION_MIN)
    return datetime(1991, month, day, hour, minute)


def action_array(action_dict: Dict[str, float]) -> np.ndarray:
    """Flatten zone setpoints into Sinergym action order: h1,c1,...,h5,c5."""
    return np.array(
        [value for zi in range(1, 6) for value in (action_dict[f"hsp_space{zi}"], action_dict[f"csp_space{zi}"])],
        dtype=np.float32,
    )


def relaxed_action() -> Dict[str, float]:
    """Energy-saving default used for smoke tests and invalid LLM responses."""
    return {
        **{f"hsp_space{i}": RELAXED_HSP for i in range(1, 6)},
        **{f"csp_space{i}": RELAXED_CSP for i in range(1, 6)},
    }


def sanitize_action(action_dict: Dict[str, float]) -> Dict[str, float]:
    """Clip setpoints to valid ranges and repair HSP >= CSP conflicts."""
    clean: Dict[str, float] = {}
    for zi in range(1, 6):
        hsp = float(np.clip(action_dict.get(f"hsp_space{zi}", RELAXED_HSP), HSP_MIN, HSP_MAX))
        csp = float(np.clip(action_dict.get(f"csp_space{zi}", RELAXED_CSP), CSP_MIN, CSP_MAX))
        if hsp >= csp:
            hsp, csp = RELAXED_HSP, RELAXED_CSP
        clean[f"hsp_space{zi}"] = hsp
        clean[f"csp_space{zi}"] = csp
    return clean


def action_plan_from_action(action_dict: Dict[str, float]) -> Dict[str, List[List[float]]]:
    """Repeat one valid action across the rolling planning horizon."""
    clean = sanitize_action(action_dict)
    return {
        zone: [
            [clean[f"hsp_space{zi}"], clean[f"csp_space{zi}"]]
            for _ in range(PLAN_HORIZON)
        ]
        for zi, zone in enumerate(ZONE_NAMES, start=1)
    }


def sanitize_action_plan(plan: Dict[str, Any]) -> Dict[str, List[List[float]]]:
    """Validate a complete plan without silently clipping the model's proposal."""
    clean_plan: Dict[str, List[List[float]]] = {}
    if not isinstance(plan, dict) or set(plan) != set(ZONE_NAMES):
        raise ValueError("plan must contain exactly the five configured zone names")
    for zi, zone in enumerate(ZONE_NAMES, start=1):
        zone_plan = plan.get(zone)
        if not isinstance(zone_plan, list) or len(zone_plan) != PLAN_HORIZON:
            raise ValueError(f"{zone} must contain exactly {PLAN_HORIZON} setpoint pairs")

        clean_plan[zone] = []
        for horizon_step, pair in enumerate(zone_plan):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"{zone} plan step {horizon_step} must be exactly [HSP, CSP]")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in pair):
                raise ValueError(f"{zone} plan step {horizon_step} setpoints must be numeric")
            hsp, csp = map(float, pair)
            if not all(math.isfinite(value) for value in (hsp, csp)):
                raise ValueError(f"{zone} plan step {horizon_step} setpoints must be finite")
            if not (HSP_MIN <= hsp <= HSP_MAX and CSP_MIN <= csp <= CSP_MAX and hsp < csp):
                raise ValueError(f"{zone} plan step {horizon_step} violates setpoint bounds or HSP < CSP")
            clean_plan[zone].append([hsp, csp])
    return clean_plan


def first_action_from_plan(plan: Dict[str, List[List[float]]]) -> Dict[str, float]:
    """Extract only horizon index 0, the action applied to the environment."""
    action: Dict[str, float] = {}
    for zi, zone in enumerate(ZONE_NAMES, start=1):
        action[f"hsp_space{zi}"] = plan[zone][0][0]
        action[f"csp_space{zi}"] = plan[zone][0][1]
    return sanitize_action(action)


def fallback_action(obs_dict: Dict[str, float]) -> Dict[str, float]:
    """Deterministic PMV-aware fallback for smoke tests or failed LLM output."""
    month = int(obs_dict.get("month", 1) or 1)
    low, high = SUMMER_COMFORT if month in SUMMER_MONTHS else WINTER_COMFORT
    action: Dict[str, float] = {}

    for zi in range(1, 6):
        occ = float(obs_dict.get(f"air_occ_space{zi}", 0.0) or 0.0) > 0.0
        temp = float(obs_dict.get(f"air_temperature_space{zi}", float("nan")))

        if not occ:
            hsp, csp = RELAXED_HSP, RELAXED_CSP
        else:
            pmv_value = obs_dict.get(pmv_observation_name(zi))
            pmv = float(pmv_value) if pmv_value is not None and np.isfinite(float(pmv_value)) else None
            hsp, csp = low, high
            if pmv is not None and pmv < PMV_LOWER:
                hsp = max(low, temp + 0.5 if not np.isnan(temp) else low)
            elif pmv is not None and pmv > PMV_UPPER:
                hsp = RELAXED_HSP
                csp = min(high, temp - 0.5 if not np.isnan(temp) else high)

        action[f"hsp_space{zi}"] = hsp
        action[f"csp_space{zi}"] = csp

    return sanitize_action(action)


def parse_llm_response(
    raw: str,
) -> tuple[Dict[str, str], Dict[str, List[List[float]]], Optional[str]]:
    """Require the complete five-zone JSON contract or return a failure reason."""

    try:
        data = json.loads(raw or "")
        if not isinstance(data, dict):
            raise ValueError("response must be a JSON object")
        required_fields = {
            "five_step_setpoint_plan",
            "zonewise_thermoanalysis_and_goal_aligned_action_justification",
        }
        if set(data) != required_fields:
            raise ValueError("response must contain exactly the required top-level fields")
        reasoning = data["zonewise_thermoanalysis_and_goal_aligned_action_justification"]
        if not isinstance(reasoning, dict) or set(reasoning) != set(ZONE_NAMES):
            raise ValueError("zone justifications must contain exactly the five configured zones")
        cleaned_reasoning: Dict[str, str] = {}
        for zone in ZONE_NAMES:
            value = reasoning[zone]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"missing or empty justification for {zone}")
            cleaned_reasoning[zone] = value.strip()
        clean_plan = sanitize_action_plan(data["five_step_setpoint_plan"])
        return cleaned_reasoning, clean_plan, None
    except Exception as exc:
        print(f"[LLM parse] failed: {exc}")
        return {}, {}, str(exc)


def current_values_with_occupied_pmv(obs_dict: Dict[str, float]) -> Dict[str, Any]:
    """Feed selected PMV into existing Brick references, masking unoccupied zones."""
    current = brick_context.current_timeseries_values(obs_dict)
    for zi in range(1, 6):
        occ = float(current.get(f"sinergym:air_occ_space{zi}") or 0.0) > 0.0
        current[f"sinergym:pmv_space{zi}"] = (
            obs_dict.get(pmv_observation_name(zi)) if occ else "unoccupied"
        )
    return current


def runtime_payload(
    current_by_ref: Dict[str, Any],
    history_by_ref: Dict[str, deque[Any]],
    step: int,
    current_label: str,
    history_meta: deque[tuple[int, str]],
) -> Dict[str, Dict[str, Any]]:
    """Attach current value, exact step-time label, and prior history to each Brick reference."""
    return brick_context.runtime_payload_for_step(
        current_by_ref=current_by_ref,
        history_by_ref=history_by_ref,
        step_number=step,
        current_label=current_label,
        history_meta=list(history_meta),
    )


def write_prompt_jsonl(records: List[Dict[str, Any]]) -> None:
    """Persist prompt/response records after every step and again in finalization."""
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    with open(PROMPT_IO_JSONL, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")


def operational_summary(rows: List[Dict[str, Any]], completed_steps: int) -> Dict[str, Any]:
    """Summarize API reliability, response quality, latency, cost, and action source."""
    calls = [row for row in rows if row["api_attempted"]]
    successful = [row for row in calls if row["api_success"] is True]
    valid = [row for row in successful if row["output_valid"] is True]
    fallbacks = [row for row in rows if row["fallback_activated"]]
    latencies = [float(row["api_latency_sec"]) for row in calls if row["api_latency_sec"] is not None]
    costs_known = all(row["api_cost_usd"] is not None for row in calls)
    total_cost = sum(float(row["api_cost_usd"]) for row in calls if row["api_cost_usd"] is not None)
    sim_hours = completed_steps * STEP_HOURS

    def rate(count: int, denominator: int) -> Optional[float]:
        return 100.0 * count / denominator if denominator else None

    exactly_applied = sum(row["applied_exactly_as_proposed"] is True for row in valid)
    return {
        "Requested Model": MODEL,
        "Response Model(s)": ", ".join(sorted({str(row["response_model"]) for row in successful if row["response_model"]})) or None,
        "Temperature": TEMPERATURE,
        "Maximum Output Tokens": MAX_TOKENS,
        "Control Intervals": completed_steps,
        "Simulated Hours": sim_hours,
        "Total LLM Calls": len(calls),
        "Successful API Calls": len(successful),
        "API Failure Rate (%)": rate(sum(row["api_success"] is not True for row in calls), len(calls)),
        "Mean Latency (s)": float(np.mean(latencies)) if latencies else None,
        "P95 Latency (s)": float(np.percentile(latencies, 95)) if latencies else None,
        "Average Input Tokens/Request": sum(row["input_tokens"] for row in calls) / len(calls) if calls else None,
        "Average Output Tokens/Request": sum(row["output_tokens"] for row in calls) / len(calls) if calls else None,
        "Invalid Output Rate (%)": rate(sum(row["output_valid"] is False for row in successful), len(successful)),
        "Fallback Activations": len(fallbacks),
        "Fallback Activation Rate (%)": rate(len(fallbacks), completed_steps),
        "Valid LLM Actions": len(valid),
        "Exactly Applied LLM Actions": exactly_applied,
        "LLM Action Application Rate (%)": rate(exactly_applied, completed_steps),
        "Total API Cost (USD)": total_cost if costs_known else None,
        "API Cost/24 h (USD)": total_cost * 24.0 / sim_hours if calls and sim_hours and costs_known else None,
    }


def write_operational_artifacts(
    rows: List[Dict[str, Any]], prompt_records: List[Dict[str, Any]], completed_steps: int
) -> Dict[str, Any]:
    """Refresh per-request, prompt/response, and aggregate operational CSVs."""
    pd.DataFrame(rows).to_csv(OPERATIONAL_STEPS_CSV, index=False)
    prompt_columns = [
        "step", "model", "response_model", "generation_id", "temperature", "reasoning_effort", "max_tokens",
        "api_success", "api_error", "finish_reason", "input_tokens", "output_tokens", "total_tokens",
        "api_cost_usd", "cost_source", "output_valid", "fallback_activated", "fallback_reason",
        "action_source", "system_prompt", "user_prompt", "raw_response", "parsed_zone_reasoning",
        "parsed_five_step_setpoint_plan",
    ]
    pd.DataFrame(prompt_records).reindex(columns=prompt_columns).to_csv(PROMPT_RESPONSE_CSV, index=False)
    summary = operational_summary(rows, completed_steps)
    pd.DataFrame([summary]).to_csv(OPERATIONAL_SUMMARY_CSV, index=False)
    return summary


def prepare_mlflow_tracking() -> None:
    if not MLFLOW_ENABLED:
        return
    if not MLFLOW_AVAILABLE or mlflow is None:
        raise RuntimeError("MLflow is enabled but the mlflow package is unavailable; install it or set MLFLOW_ENABLED=0.")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if MLFLOW_TRACKING_URI.startswith("sqlite:") and mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        os.makedirs(MLFLOW_ARTIFACT_DIR, exist_ok=True)
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT,
            artifact_location=Path(MLFLOW_ARTIFACT_DIR).resolve().as_uri(),
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def log_mlflow(
    metrics: Dict[str, Any], artifacts: List[str], step_rows: List[Dict[str, Any]],
    operational_result: Dict[str, Any],
) -> None:
    if not MLFLOW_ENABLED:
        print("[MLFLOW] disabled")
        return
    prepare_mlflow_tracking()
    with mlflow.start_run(run_name=SAFE_ENV_NAME) as active_run:
        mlflow.log_params({
            "model": MODEL,
            "env_id": ENV_ID,
            "pmv_source": PMV_SOURCE,
            "runperiod": str(BUILDING_CONFIG["runperiod"]),
            "expected_control_steps": TOTAL_CONTROL_STEPS,
            "history_max": MAX_HISTORY,
            "plan_horizon": PLAN_HORIZON,
            "use_llm_api": USE_LLM_API,
            "brick_ttl": str(ttl_path),
            "temperature": TEMPERATURE,
            "reasoning_effort": REASONING_EFFORT or "provider_default",
            "max_tokens": MAX_TOKENS,
            "prompt_version": PROMPT_VERSION_TAG,
        })
        aggregate_metrics = {
            "sim_steps_completed": len(step_rows),
            "total_hvac_energy_kwh": metrics.get("total_hvac_energy_kwh", 0.0),
            "pmv_violation_pct_all": metrics.get("pmv_violation_pct_all"),
            "comfort_violation_pct_all": metrics.get("comfort_violation_pct_all"),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_api_cost_usd": operational_result.get("Total API Cost (USD)"),
            "total_runtime_sec": round(sum(step_runtimes_sec), 3),
        }
        for key, summary_key in {
            "llm_mean_latency_sec": "Mean Latency (s)",
            "llm_p95_latency_sec": "P95 Latency (s)",
            "llm_api_failure_rate_pct": "API Failure Rate (%)",
            "llm_invalid_output_rate_pct": "Invalid Output Rate (%)",
            "llm_fallback_activation_rate_pct": "Fallback Activation Rate (%)",
            "llm_action_application_rate_pct": "LLM Action Application Rate (%)",
            "llm_cost_per_24h_usd": "API Cost/24 h (USD)",
        }.items():
            aggregate_metrics[key] = operational_result.get(summary_key)
        mlflow.log_metrics({
            key: float(value) for key, value in aggregate_metrics.items()
            if value is not None and math.isfinite(float(value))
        })
        for row in step_rows:
            step_metrics: Dict[str, float] = {
                "hvac_energy_kwh_by_step": float(row["energy_kWh_after_step"]),
                "fallback_applied_by_step": float(bool(row["fallback_activated"])),
                "input_tokens_by_step": float(row["input_tokens"]),
                "output_tokens_by_step": float(row["output_tokens"]),
            }
            if row.get("llm_latency_sec") is not None:
                step_metrics["llm_latency_sec_by_step"] = float(row["llm_latency_sec"])
            if row.get("api_success") is not None:
                step_metrics["api_success_by_step"] = float(bool(row["api_success"]))
            if row.get("output_valid") is not None:
                step_metrics["output_valid_by_step"] = float(bool(row["output_valid"]))
            if row.get("api_cost_usd") is not None:
                step_metrics["api_cost_usd_by_step"] = float(row["api_cost_usd"])
            occupied_zone_violations: List[int] = []
            for zi in range(1, 6):
                for metric, column in (
                    (f"zone{zi}_temperature_c_by_step", f"air_temperature_space{zi}_after"),
                    (f"zone{zi}_hsp_c_by_step", f"HSP_applied_Z{zi}"),
                    (f"zone{zi}_csp_c_by_step", f"CSP_applied_Z{zi}"),
                ):
                    value = row.get(column)
                    if value is not None and math.isfinite(float(value)):
                        step_metrics[metric] = float(value)
                pmv_value = row.get(f"pmv_space{zi}_after")
                if pmv_value is not None and math.isfinite(float(pmv_value)):
                    step_metrics[f"zone{zi}_pmv_by_step"] = float(pmv_value)
                violation = row.get(f"pmv_violation_space{zi}_after")
                if violation is not None:
                    step_metrics[f"zone{zi}_pmv_violation_by_step"] = float(violation)
                    occupied_zone_violations.append(int(violation))
            if occupied_zone_violations:
                step_metrics["any_occupied_zone_pmv_violation_by_step"] = float(max(occupied_zone_violations))
            mlflow.log_metrics(step_metrics, step=int(row["step_number"]))
        for artifact in artifacts:
            if artifact and os.path.isfile(artifact):
                mlflow.log_artifact(artifact)
    print(f"[MLFLOW] logged run_id={active_run.info.run_id} experiment={MLFLOW_EXPERIMENT}")


# =============================================================================
# SETUP: load Brick TTL, patch formatter, make OpenAI client and Sinergym env.
# =============================================================================

if USE_LLM_API and OpenAI is None:
    raise RuntimeError("Install openai or set USE_LLM_API=0 for smoke testing.")

brick_context.hop.point_details = brick_context.point_details_with_step_context
ttl_path = brick_context.resolve_ttl_path()
brick_graph = RDFGraph()
brick_graph.parse(ttl_path, format="turtle")
print(f"[BRICK] ttl={ttl_path} | triples={len(brick_graph)}")

client = OpenAI(api_key=OPENAI_API_KEY) if USE_LLM_API else None
if MLFLOW_ENABLED:
    prepare_mlflow_tracking()
print(
    f"[SETUP] env={ENV_ID} | run={SAFE_ENV_NAME} | out_dir={OUT_DIR} | "
    f"runperiod={BUILDING_CONFIG['runperiod']} | expected_steps={TOTAL_CONTROL_STEPS} | "
    f"llm_api={'on' if USE_LLM_API else 'off'} | mlflow={'on' if MLFLOW_ENABLED and MLFLOW_AVAILABLE else 'off'} | pmv_source={PMV_SOURCE}"
)

env = gym.make(ENV_ID, env_name=SAFE_ENV_NAME, building_config=BUILDING_CONFIG)
env = LoggerWrapper(env)
env = CSVLogger(env)
signal.signal(signal.SIGTERM, lambda _signum, _frame: (_ for _ in ()).throw(KeyboardInterrupt()))

obs_vars = list(env.get_wrapper_attr("observation_variables"))
missing_pmv = [pmv_observation_name(zi) for zi in range(1, 6) if pmv_observation_name(zi) not in obs_vars]
if missing_pmv:
    env.close()
    raise ValueError(f"Missing {PMV_SOURCE} PMV observations: {missing_pmv}")
obs, info = env.reset(seed=0)


# =============================================================================
# RUN STATE: everything below is flushed repeatedly and in `finally`.
# =============================================================================

history_by_ref: Dict[str, deque[Any]] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
history_meta: deque[tuple[int, str]] = deque(maxlen=MAX_HISTORY)
step_log_rows: List[Dict[str, Any]] = []
graph_retrieval_rows: List[Dict[str, Any]] = []
prompt_io_records: List[Dict[str, Any]] = []
operational_rows: List[Dict[str, Any]] = []
action_source_rows: List[Dict[str, Any]] = []
step_runtimes_sec: List[float] = []
active_op_row: Optional[Dict[str, Any]] = None
active_prompt_record: Optional[Dict[str, Any]] = None
llm_t0: Optional[float] = None

total_input_tokens = 0
total_output_tokens = 0
total_tokens = 0
total_api_cost_usd = 0.0
invalid_actions = 0
llm_failures = 0
total_reward = 0.0
run_t0 = time.perf_counter()
phase = "brick_spatiotemporal_context"


# =============================================================================
# MAIN LOOP: build context, prompt LLM, apply action, save artifacts each step.
# =============================================================================

try:
    step = 0
    while step < TOTAL_CONTROL_STEPS:
        step_t0 = time.perf_counter()
        llm_t0 = None
        obs_dict = obs_to_dict(obs, obs_vars)
        timestep = int(info.get("timestep", step))
        ts_now = timestamp_from_obs(obs_dict, timestep)
        current_label = brick_context.current_time_label(obs_dict, step)

        current_by_ref = current_values_with_occupied_pmv(obs_dict)
        runtime_data = runtime_payload(current_by_ref, history_by_ref, step, current_label, history_meta)
        context_text = brick_context.build_clean_context(brick_graph, runtime_data)
        user_content = build_user_prompt(step, current_label, context_text)
        system_content = SYSTEM_PROMPT

        graph_retrieval_rows.append(
            {
                "step_number": step,
                "timestamp": ts_now.isoformat(),
                "current_step_time_label": current_label,
                "brick_ttl": str(ttl_path),
                "spatiotemporal_context": context_text[:200000],
                "full_llm_user_message": user_content[:250000],
            }
        )

        usage_input_tokens = usage_output_tokens = usage_total_tokens = 0
        llm_latency_sec: Optional[float] = None
        raw = ""
        api_success: Optional[bool] = None
        api_error: Optional[str] = None
        response_model: Optional[str] = None
        generation_id: Optional[str] = None
        finish_reason: Optional[str] = None
        step_api_cost: Optional[float] = None
        cost_source: Optional[str] = None
        api_attempted = bool(USE_LLM_API and client is not None)
        active_op_row = {
            "step_number": step, "timestamp": ts_now.isoformat(), "requested_model": MODEL,
            "response_model": None, "generation_id": None, "temperature": TEMPERATURE,
            "reasoning_effort": REASONING_EFFORT,
            "max_tokens": MAX_TOKENS, "api_attempted": api_attempted, "api_success": None,
            "api_status": "pending" if api_attempted else "disabled", "api_error": None,
            "api_latency_sec": None, "finish_reason": None, "input_tokens": 0,
            "output_tokens": 0, "total_tokens": 0, "api_cost_usd": None, "cost_source": None,
            "output_valid": None, "invalid_output_reason": None, "proposed_plan_json": None,
            "submitted_action_json": None, "actual_action_json": None,
            "applied_exactly_as_proposed": None, "fallback_activated": False,
            "fallback_reason": None, "fallback_success": None, "sim_step_completed": False,
        }
        operational_rows.append(active_op_row)
        active_reason = ""
        if api_attempted:
            try:
                kwargs: Dict[str, Any] = {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": TEMPERATURE,
                    "response_format": {"type": "json_object"},
                }
                if REASONING_EFFORT is not None:
                    kwargs["reasoning_effort"] = REASONING_EFFORT
                model_lower = MODEL.lower()
                if model_lower.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
                    kwargs["max_completion_tokens"] = MAX_TOKENS
                else:
                    kwargs["max_tokens"] = MAX_TOKENS
                llm_t0 = time.perf_counter()
                response = client.chat.completions.create(**kwargs)
                api_success = True
                response_model = getattr(response, "model", None)
                generation_id = getattr(response, "id", None)
                choice = response.choices[0]
                finish_reason = getattr(choice, "finish_reason", None)
                raw = (choice.message.content or "").strip()
                usage = getattr(response, "usage", None)
                if usage is not None:
                    usage_input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                    usage_output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                    usage_total_tokens = int(getattr(usage, "total_tokens", usage_input_tokens + usage_output_tokens) or usage_input_tokens + usage_output_tokens)
                    step_api_cost = (
                        usage_input_tokens * OPENAI_INPUT_PRICE_PER_1M / 1_000_000.0
                        + usage_output_tokens * OPENAI_OUTPUT_PRICE_PER_1M / 1_000_000.0
                    )
                    cost_source = "configured_openai_token_price"
            except Exception as exc:
                api_success = False
                api_error = str(exc)
                step_api_cost = 0.0
                cost_source = "api_failure_assumed_zero"
                llm_failures += 1
                active_reason = "api_error"
                print(f"[LLM] call failed at step={step}: {exc}")
            finally:
                if llm_t0 is not None:
                    llm_latency_sec = time.perf_counter() - llm_t0
        else:
            active_reason = "api_disabled"

        proposed_plan: Dict[str, List[List[float]]] = {}
        if api_success is True:
            zone_reasoning, proposed_plan, validation_error = parse_llm_response(raw)
            if validation_error:
                active_reason = f"invalid_output:{validation_error}"
                invalid_actions += 1
                output_valid: Optional[bool] = False
            else:
                output_valid = True
        else:
            zone_reasoning = {}
            validation_error = None
            output_valid = None

        fallback_activated = bool(active_reason)
        if fallback_activated:
            action_source = f"fallback_{active_reason.split(':', 1)[0]}"
            action_dict = fallback_action(obs_dict)
            action_plan = action_plan_from_action(action_dict)
            zone_reasoning = {
                zone: f"Rule-based PMV-aware fallback: {active_reason}." for zone in ZONE_NAMES
            }
        else:
            action_source = "gpt"
            action_plan = proposed_plan
            action_dict = first_action_from_plan(action_plan)

        if api_success is True and step_api_cost is None:
            # Preserve an explicit unknown cost when usage metadata was absent.
            cost_source = "usage_metadata_unavailable"
        if step_api_cost is not None:
            total_api_cost_usd += step_api_cost
        total_input_tokens += usage_input_tokens
        total_output_tokens += usage_output_tokens
        total_tokens += usage_total_tokens

        step_api_cost_for_prompt = step_api_cost
        active_op_row.update(
            response_model=response_model, generation_id=generation_id, api_success=api_success,
            api_status="success" if api_success is True else "failed" if api_attempted else "disabled",
            api_error=api_error, api_latency_sec=llm_latency_sec, finish_reason=finish_reason,
            input_tokens=usage_input_tokens, output_tokens=usage_output_tokens, total_tokens=usage_total_tokens,
            api_cost_usd=step_api_cost, cost_source=cost_source, output_valid=output_valid,
            invalid_output_reason=validation_error if output_valid is False else None,
            proposed_plan_json=json.dumps(proposed_plan) if proposed_plan else None,
            fallback_activated=fallback_activated, fallback_reason=active_reason or None,
        )
        active_prompt_record = {
            "step": step, "model": MODEL, "response_model": response_model,
            "generation_id": generation_id, "temperature": TEMPERATURE,
            "reasoning_effort": REASONING_EFFORT, "max_tokens": MAX_TOKENS,
            "system_prompt": system_content, "user_prompt": user_content, "raw_response": raw,
            "parsed_zone_reasoning": zone_reasoning, "parsed_five_step_setpoint_plan": action_plan,
            "proposed_five_step_setpoint_plan": proposed_plan or None,
            "applied_action_horizon_index": 0, "input_tokens": usage_input_tokens,
            "output_tokens": usage_output_tokens, "total_tokens": usage_total_tokens,
            "api_cost_usd": step_api_cost_for_prompt, "cost_source": cost_source,
            "api_success": api_success, "api_error": api_error, "finish_reason": finish_reason,
            "output_valid": output_valid, "fallback_activated": fallback_activated,
            "action_source": action_source, "fallback_reason": active_reason or None,
        }
        prompt_io_records.append(active_prompt_record)
        write_prompt_jsonl(prompt_io_records)

        if DEBUG_PRINT:
            print("\n" + "=" * 118)
            print(f"[STEP {step}] {current_label} | action_source={action_source} | phase={phase}")
            print("-" * 118)
            print(">>> LLM INPUT (exact user message) <<<")
            print(user_content)
            print("-" * 118)
            print(">>> LLM OUTPUT: zonewise thermoanalysis and goal-aligned action justification <<<")
            for zone in ZONE_NAMES:
                print(f"\n--- {zone} ---")
                print(zone_reasoning.get(zone, "").strip() or "[empty]")
            print(f"\n--- five-step plan (only horizon 0 is applied at step {step}) ---")
            for zi in range(1, 6):
                zone = f"SPACE{zi}-1"
                print(
                    f"  {zone}: "
                    + " | ".join(
                        f"t+{offset}: HSP={pair[0]:.2f}, CSP={pair[1]:.2f}"
                        for offset, pair in enumerate(action_plan[zone])
                    )
                )
            print("=" * 118)

        # Archive the exact values used in this prompt only after prompting.
        # This keeps each next step's history strictly prior-step context.
        for ref_name, value in current_by_ref.items():
            history_by_ref[ref_name].append(value)
        history_meta.append((step, brick_context.history_time_label(obs_dict, step)))

        # Receding-horizon control: apply only plan index 0, then replan next step.
        submitted_action = action_array(action_dict)
        active_op_row["submitted_action_json"] = json.dumps(action_dict)
        obs, reward, terminated, truncated, info = env.step(submitted_action)
        total_reward += float(reward)
        applied_exactly = bool(
            output_valid is True and np.array_equal(action_array(first_action_from_plan(proposed_plan)), submitted_action)
        )
        active_op_row.update(
            actual_action_json=json.dumps(action_dict),
            applied_exactly_as_proposed=applied_exactly if output_valid is True else None,
            fallback_success=True if fallback_activated else None,
            sim_step_completed=True,
        )
        active_prompt_record.update(
            submitted_action=action_dict,
            actual_action=action_dict,
            applied_exactly_as_proposed=applied_exactly if output_valid is True else None,
        )

        obs_after = obs_to_dict(obs, obs_vars)
        timestep_after = int(info.get("timestep", step + 1))
        ts_after = timestamp_from_obs(obs_after, timestep_after)
        expected_after = ts_now + timedelta(minutes=STEP_DURATION_MIN)
        if ts_after != expected_after:
            print(f"[TIME WARNING] step {step}: expected {expected_after.isoformat()} got {ts_after.isoformat()}")

        post_power_w = float(obs_after.get("HVAC_electricity_demand_rate", 0.0) or 0.0)
        if np.isnan(post_power_w):
            post_power_w = 0.0
        post_kwh = post_power_w * STEP_HOURS / 1000.0

        step_runtime_sec = time.perf_counter() - step_t0
        step_runtimes_sec.append(step_runtime_sec)

        row: Dict[str, Any] = {
            "step_number": step,
            "timestamp": ts_now.isoformat(),
            "current_step_time_label": current_label,
            "phase": phase,
            "pmv_source": PMV_SOURCE,
            "reward": float(reward),
            "hvac_power_W_after_step": round(post_power_w, 2),
            "energy_kWh_after_step": round(post_kwh, 6),
            "zonewise_thermoanalysis_and_goal_aligned_action_justification": json.dumps(zone_reasoning, ensure_ascii=True),
            "five_step_setpoint_plan": json.dumps(action_plan, ensure_ascii=True),
            "applied_action_horizon_index": 0,
            "input_tokens": usage_input_tokens,
            "output_tokens": usage_output_tokens,
            "total_tokens": usage_total_tokens,
            "api_cost_usd": round(step_api_cost, 6) if step_api_cost is not None else None,
            "llm_latency_sec": round(llm_latency_sec, 4) if llm_latency_sec is not None else None,
            "invalid_action": int(output_valid is False),
            "api_success": api_success,
            "api_error": api_error,
            "response_model": response_model,
            "generation_id": generation_id,
            "finish_reason": finish_reason,
            "output_valid": output_valid,
            "action_source": action_source,
            "fallback_activated": fallback_activated,
            "fallback_reason": active_reason or None,
            "applied_exactly_as_proposed": applied_exactly if output_valid is True else None,
            "step_runtime_sec": round(step_runtime_sec, 4),
            "llm_system_message": system_content[:50000],
            "llm_user_message": user_content[:100000],
            "raw_llm_response": raw[:12000],
        }
        for zi in range(1, 6):
            row[f"HSP_applied_Z{zi}"] = action_dict[f"hsp_space{zi}"]
            row[f"CSP_applied_Z{zi}"] = action_dict[f"csp_space{zi}"]
            row[f"SPACE{zi}_reasoning"] = zone_reasoning.get(f"SPACE{zi}-1", "")[:12000]
            temp_after = obs_after.get(f"air_temperature_space{zi}")
            row[f"air_temperature_space{zi}_after"] = float(temp_after) if temp_after is not None and np.isfinite(float(temp_after)) else None
            pmv_after = obs_after.get(pmv_observation_name(zi))
            pmv_is_finite = pmv_after is not None and np.isfinite(float(pmv_after))
            occupied_after = float(obs_after.get(f"air_occ_space{zi}", 0.0) or 0.0) > 0.0
            row[f"pmv_space{zi}_after"] = float(pmv_after) if pmv_is_finite else None
            row[f"occupied_space{zi}_after"] = occupied_after
            row[f"pmv_violation_space{zi}_after"] = (
                int(float(pmv_after) < PMV_LOWER or float(pmv_after) > PMV_UPPER)
                if occupied_after and pmv_is_finite else None
            )
        step_log_rows.append(row)
        action_source_rows.append({
            "step_number": step, "timestamp": ts_now.isoformat(), "action_source": action_source,
            "fallback_activated": fallback_activated, "fallback_reason": active_reason or None,
            "api_success": api_success, "output_valid": output_valid,
            "applied_exactly_as_proposed": applied_exactly if output_valid is True else None,
            **{f"HSP_applied_Z{zi}": action_dict[f"hsp_space{zi}"] for zi in range(1, 6)},
            **{f"CSP_applied_Z{zi}": action_dict[f"csp_space{zi}"] for zi in range(1, 6)},
        })

        pd.DataFrame(step_log_rows).to_csv(STEP_LOG_CSV, index=False)
        pd.DataFrame(graph_retrieval_rows).to_csv(GRAPH_RETRIEVAL_CSV, index=False)
        pd.DataFrame(action_source_rows).to_csv(ACTION_SOURCE_CSV, index=False)
        operational_result = write_operational_artifacts(operational_rows, prompt_io_records, len(step_log_rows))
        active_op_row = None
        active_prompt_record = None

        step += 1
        if terminated or truncated:
            if step + 1 < TOTAL_CONTROL_STEPS:
                print(
                    f"[RUN PERIOD WARNING] Environment ended after {step + 1} of "
                    f"{TOTAL_CONTROL_STEPS} configured control intervals."
                )
            break

except KeyboardInterrupt:
    print("\n[INTERRUPT] Stopping; completed-step CSVs and prompt logs will be flushed.")
    if active_op_row is not None and active_op_row.get("api_status") == "pending":
        active_op_row.update(
            api_status="interrupted",
            api_success=False,
            api_error="interrupted before API response was recorded",
        )
        if llm_t0 is not None:
            active_op_row["api_latency_sec"] = time.perf_counter() - llm_t0
        if active_prompt_record is not None:
            active_prompt_record.update(
                api_success=False,
                api_error="interrupted before API response was recorded",
            )

finally:
    # =============================================================================
    # FINALIZATION: save CSVs, prompts, metrics, plots, MLflow, and close env.
    # =============================================================================
    pd.DataFrame(step_log_rows).to_csv(STEP_LOG_CSV, index=False)
    pd.DataFrame(graph_retrieval_rows).to_csv(GRAPH_RETRIEVAL_CSV, index=False)
    pd.DataFrame(action_source_rows).to_csv(ACTION_SOURCE_CSV, index=False)
    write_prompt_jsonl(prompt_io_records)
    operational_result = write_operational_artifacts(operational_rows, prompt_io_records, len(step_log_rows))

    print(f"[SAVED] {STEP_LOG_CSV}")
    print(f"[SAVED] {GRAPH_RETRIEVAL_CSV}")
    print(f"[SAVED] {ACTION_SOURCE_CSV}")
    print(f"[SAVED] {PROMPT_IO_JSONL}")
    print(f"[SAVED] {PROMPT_RESPONSE_CSV}")
    print(f"[SAVED] {OPERATIONAL_STEPS_CSV}")
    print(f"[SAVED] {OPERATIONAL_SUMMARY_CSV}")
    print(
        f"[TOKENS] in={total_input_tokens} out={total_output_tokens} total={total_tokens} "
        f"cost=${total_api_cost_usd:.4f} invalid_actions={invalid_actions} "
        f"llm_failures={llm_failures} runtime={time.perf_counter() - run_t0:.1f}s"
    )

    # CSVLogger writes final monitor/progress files during env.close(), so close
    # before searching for observations.csv and computing post-run metrics.
    try:
        env.close()
    except Exception as exc:
        print(f"[ENV] close failed before metrics: {exc}")

    obs_csv: Optional[str] = None
    try:
        workspace_path = env.get_wrapper_attr("workspace_path")
        candidates = [
            os.path.join(workspace_path, "episode-1", "monitor", "observations.csv"),
            os.path.join(workspace_path, "monitor", "observations.csv"),
        ]
        try:
            episode_path = env.get_wrapper_attr("episode_path")
            candidates.append(os.path.join(episode_path, "monitor", "observations.csv"))
        except Exception:
            pass
        obs_csv = next((path for path in candidates if path and os.path.isfile(path)), None)
    except Exception:
        obs_csv = None

    performance_metrics: Dict[str, Any] = {}
    plot_paths: List[str] = []
    action_traj_csv = os.path.join(METRICS_DIR, "action_trajectory.csv")

    try:
        if step_log_rows:
            cols = ["step_number", "timestamp", "reward", "energy_kWh_after_step"]
            cols += [f"HSP_applied_Z{i}" for i in range(1, 6)]
            cols += [f"CSP_applied_Z{i}" for i in range(1, 6)]
            pd.DataFrame(step_log_rows)[cols].to_csv(action_traj_csv, index=False)

        date_start, date_end = runperiod_to_date_range(BUILDING_CONFIG["runperiod"])
        if obs_csv and os.path.isfile(obs_csv):
            obs_df = pd.read_csv(obs_csv)
            performance_metrics = compute_performance_metrics(
                observations_for_pmv_metrics(obs_df), date_start, date_end
            )
            performance_metrics["pmv_source"] = PMV_SOURCE
            performance_metrics["observations_csv"] = obs_csv
            save_pmv_summary_artifacts(
                performance_metrics,
                label=f"{SAFE_ENV_NAME} | PMV source: {PMV_SOURCE}",
                csv_path=PMV_SUMMARY_CSV,
                txt_path=PMV_SUMMARY_TXT,
                date_start=date_start,
                date_end=date_end,
            )

            if MPL_AVAILABLE:
                temp_cols = [f"air_temperature_space{z}" for z in range(1, 6)]
                if all(col in obs_df.columns for col in temp_cols):
                    fig, ax = plt.subplots(figsize=(12, 5))
                    x = np.arange(len(obs_df))
                    for zi, col in enumerate(temp_cols, start=1):
                        ax.plot(x, pd.to_numeric(obs_df[col], errors="coerce"), label=f"Z{zi}")
                    ax.set_xlabel("Step index")
                    ax.set_ylabel("Zone air temperature (degC)")
                    ax.set_title("Zone temperatures")
                    ax.legend(loc="best", fontsize=8)
                    ax.grid(True, alpha=0.3)
                    out = os.path.join(PLOTS_DIR, "zone_temperatures.png")
                    fig.tight_layout()
                    fig.savefig(out, dpi=150)
                    plt.close(fig)
                    plot_paths.append(out)

                if "HVAC_electricity_demand_rate" in obs_df.columns:
                    power_w = pd.to_numeric(obs_df["HVAC_electricity_demand_rate"], errors="coerce").fillna(0.0)
                    kwh_step = power_w * STEP_HOURS / 1000.0
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
                    x = np.arange(len(obs_df))
                    ax1.plot(x, power_w, color="#BF616A")
                    ax1.set_ylabel("HVAC power (W)")
                    ax1.set_title("HVAC electrical demand")
                    ax1.grid(True, alpha=0.3)
                    ax2.plot(x, kwh_step.cumsum(), color="#5E81AC")
                    ax2.set_xlabel("Step index")
                    ax2.set_ylabel("Cumulative energy (kWh)")
                    ax2.grid(True, alpha=0.3)
                    out = os.path.join(PLOTS_DIR, "hvac_energy.png")
                    fig.tight_layout()
                    fig.savefig(out, dpi=150)
                    plt.close(fig)
                    plot_paths.append(out)

            print("\n" + format_metrics_report(performance_metrics, label=SAFE_ENV_NAME))
            print(f"[METRICS] observations.csv: {obs_csv}")
        else:
            print("[METRICS] observations.csv not found; PMV/energy metrics skipped.")

        report = {
            "run_name": SAFE_ENV_NAME,
            "prompt_version": PROMPT_VERSION_TAG,
            "llm": {
                "model": MODEL,
                "temperature": TEMPERATURE,
                "reasoning_effort": REASONING_EFFORT,
                "max_tokens": MAX_TOKENS,
                "use_llm_api": USE_LLM_API,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "total_api_cost_usd": operational_result["Total API Cost (USD)"],
                "invalid_actions": invalid_actions,
                "llm_failures": llm_failures,
                "gpt_actions_applied": sum(row["action_source"] == "gpt" for row in action_source_rows),
                "fallback_actions_applied": sum(row["action_source"] != "gpt" for row in action_source_rows),
            },
            "llm_operational_summary": operational_result,
            "simulation": {
                "expected_control_steps": TOTAL_CONTROL_STEPS,
                "steps_completed": len(step_log_rows),
                "total_reward": total_reward,
                "mean_reward": float(np.mean([r["reward"] for r in step_log_rows])) if step_log_rows else None,
                "total_runtime_sec": round(sum(step_runtimes_sec), 3),
            },
            "performance_from_observations_csv": performance_metrics,
            "artifact_paths": {
                "step_log_csv": STEP_LOG_CSV,
                "graph_retrieval_csv": GRAPH_RETRIEVAL_CSV,
                "observations_csv": obs_csv,
                "pmv_summary_csv": PMV_SUMMARY_CSV if os.path.isfile(PMV_SUMMARY_CSV) else None,
                "pmv_summary_txt": PMV_SUMMARY_TXT if os.path.isfile(PMV_SUMMARY_TXT) else None,
                "prompt_io_jsonl": PROMPT_IO_JSONL,
                "prompt_response_csv": PROMPT_RESPONSE_CSV,
                "action_source_csv": ACTION_SOURCE_CSV,
                "operational_steps_csv": OPERATIONAL_STEPS_CSV,
                "operational_summary_csv": OPERATIONAL_SUMMARY_CSV,
                "action_trajectory_csv": action_traj_csv if os.path.isfile(action_traj_csv) else None,
                "brick_ttl": str(ttl_path),
                "plots": plot_paths,
            },
        }
        with open(PERFORMANCE_JSON, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=True, default=str)
        print(f"[SAVED] {PERFORMANCE_JSON}")

        artifacts = [
            STEP_LOG_CSV, GRAPH_RETRIEVAL_CSV, ACTION_SOURCE_CSV, PROMPT_IO_JSONL,
            PROMPT_RESPONSE_CSV, OPERATIONAL_STEPS_CSV, OPERATIONAL_SUMMARY_CSV,
            PERFORMANCE_JSON, PMV_SUMMARY_CSV, PMV_SUMMARY_TXT, action_traj_csv,
            obs_csv or "", *plot_paths,
        ]
        log_mlflow(performance_metrics, artifacts, step_log_rows, operational_result)

        print("\nLLM operational performance (all recorded steps)")
        print(pd.DataFrame(list(operational_result.items()), columns=["Metric", "Value"]).to_string(index=False))

    except Exception as exc:
        print(f"[FINALIZE] metrics/MLflow finalization failed: {exc}")
