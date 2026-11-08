#!/usr/bin/env python3
"""Run official DARLIN's text-RAG control loop on the local five-zone Sinergym model.

Each 15-minute decision asks for an eight-pair continuous heating/cooling plan
per zone, preserving DARLIN's two-hour planning horizon. As in the official
EnergyPlus plugin, the first proposal is applied immediately, then later
proposals are applied one decision later. If a control response is unavailable
or invalid, a PMV-aware deterministic fallback applies immediately and clears
that zone's queued action.
``--dry-run`` exercises retrieval and prompt construction without EnergyPlus or
an API. ``--smoke-env`` exercises two EnergyPlus steps without an API.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time as wall_time
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path

import faiss
import gymnasium as gym
import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pythermalcomfort.models import pmv_ppd_iso

import sinergym  # Registers Eplus-5zone-zonal-arizona-continuous-v1.
from darlin import DarlinGenerator  # Reuse the official model API and JSON extraction.
from sinergym.utils.wrappers import CSVLogger, LoggerWrapper


# Paths are inside /workspaces/proj/codes because this script runs in the existing container.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from pmv_energy_metrics import PMV_LOWER, PMV_UPPER

TEXT_DIR = ROOT / "data" / "text"
ENV_ID = "Eplus-5zone-zonal-arizona-continuous-v1"
ZONES = tuple(f"SPACE{i}-1" for i in range(1, 6))
# Model-specific geometry labels: four glazed perimeter zones and one core.
# The EnergyPlus model uses a 30-degree Building North Axis, so avoid guessing
# cardinal directions from the front/right/back/left names.
ZONE_LAYOUT_SOLAR_CONTEXT = {
    "SPACE1-1": "front perimeter zone with exterior glazing",
    "SPACE2-1": "right perimeter zone with exterior glazing",
    "SPACE3-1": "back perimeter zone with exterior glazing",
    "SPACE4-1": "left perimeter zone with exterior glazing",
    "SPACE5-1": "central core with no direct exterior wall or window exposure",
}
HSP_BOUNDS = (12.0, 23.25)  # Continuous heating schedule range in the local YAML.
CSP_BOUNDS = (23.25, 30.0)  # Continuous cooling schedule range in the local YAML.
SIM_STEP_MINUTES = 15
CONTROL_STEPS = 1  # Make one control decision at every Sinergym step.
CONTROL_INTERVAL_MINUTES = SIM_STEP_MINUTES * CONTROL_STEPS
HISTORY_WINDOW_MINUTES = 6 * 60
HISTORY_SAMPLES = HISTORY_WINDOW_MINUTES // CONTROL_INTERVAL_MINUTES
SUMMER_OCCUPIED_PAIR = (23.0, 26.0)  # June-September occupied warm-up.
WINTER_OCCUPIED_PAIR = (20.0, 23.5)  # Other-month occupied warm-up.
UNOCCUPIED_PAIR = (12.0, 30.0)  # Any-month unoccupied warm-up and emergency hold.
PLAN_HORIZON_MINUTES = 120  # Preserve DARLIN's two-hour planning horizon.
PLAN_HORIZON = PLAN_HORIZON_MINUTES // CONTROL_INTERVAL_MINUTES
DEFAULT_MODEL = "google/gemma-4-26b-a4b-it"
DEFAULT_OPENAI_MODEL = "gpt-6-sol"
DEFAULT_API_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_API_BASE_URL = "https://api.openai.com/v1"
REASONING_EFFORT = "none"
MODEL_TEMPERATURE = 0.0
MODEL_MAX_TOKENS = 9000
# OpenAI fallback cost estimate for gpt-6-sol, in USD per 1M tokens.
# A provider-reported request cost takes precedence when available.
OPENAI_INPUT_PRICE_PER_1M = 2.00
OPENAI_OUTPUT_PRICE_PER_1M = 10.00
NEW_PMV_DEFINITION = (
    "ISO 7730 PMV using zone air temperature for both air and radiant temperature, "
    "RH=40%, air speed=0.15 m/s, met=1.1, clo=0.5 in June-September and 1.0 otherwise"
)
# Set OPENROUTER_API_KEY in the environment when using OpenRouter.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
# Set OPENAI_API_KEY in the environment when using OpenAI.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
def require_text_files() -> None:
    """Fail visibly if the author's FAISS asset was not copied into the mounted repo."""
    for name in ("merged_faiss_may26.index", "merged_metadata_may26.json"):
        if not (TEXT_DIR / name).is_file():
            raise FileNotFoundError(f"Missing DARLIN retrieval asset: {TEXT_DIR / name}")


class TextRetriever:
    """The official FAISS L2/text retrieval, without its unavailable time-series DB."""

    def __init__(self) -> None:
        require_text_files()
        self.index = faiss.read_index(str(TEXT_DIR / "merged_faiss_may26.index"))
        with (TEXT_DIR / "merged_metadata_may26.json").open(encoding="utf-8") as stream:
            self.documents = json.load(stream)
        if self.index.ntotal != len(self.documents):
            raise ValueError("FAISS vector count differs from author metadata count")
        # The same 384-dimensional embedding model used in the paper and official code.
        self.encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
        )
        if self.index.d != self.encoder.get_sentence_embedding_dimension():
            raise ValueError("Author FAISS dimension differs from MiniLM embedding dimension")

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Return the closest three paper chunks, preserving scores and metadata."""
        vector = self.encoder.encode([query], convert_to_numpy=True).astype("float32")
        distances, ids = self.index.search(vector, k)
        return [
            {"distance_l2": float(distance), **self.documents[int(idx)]}
            for distance, idx in zip(distances[0], ids[0]) if idx >= 0
        ]


def calculate_new_pmv(temperature_c: float, month: int) -> float:
    """Calculate the ISO 7730 value exposed to DARLIN as new_pmv_space1..5."""
    temperature_c = float(temperature_c)
    if not math.isfinite(temperature_c):
        raise ValueError("Cannot calculate new PMV from a non-finite zone temperature")
    clothing = 0.5 if int(month) in (6, 7, 8, 9) else 1.0
    result = pmv_ppd_iso(
        tdb=temperature_c, tr=temperature_c, vr=0.15, rh=40.0,
        met=1.1, clo=clothing, limit_inputs=False, round_output=False,
    )
    value = float(result.pmv)
    if not math.isfinite(value):
        raise ValueError("ISO 7730 PMV calculation returned a non-finite value")
    return value


def as_obs_dict(observation: np.ndarray, names: list[str]) -> dict[str, float]:
    """Map simulator variables and derive the five YAML-defined new PMVs."""
    if len(observation) != len(names):
        raise ValueError(f"Observation length {len(observation)} != names {len(names)}")
    obs = {name: float(value) for name, value in zip(names, observation)}
    month = int(obs["month"])
    for zone_number in range(1, 6):
        obs[f"new_pmv_space{zone_number}"] = calculate_new_pmv(
            obs[f"air_temperature_space{zone_number}"], month
        )
        # Do not let downstream controller code consume EnergyPlus Fanger PMV.
        obs.pop(f"pmv_space{zone_number}", None)
    return obs


def action_array(pairs: dict[str, tuple[float, float]]) -> np.ndarray:
    """Map five independently selected (H,C) pairs to Sinergym action order."""
    return np.asarray(
        [value for zone in ZONES for value in pairs[zone]],
        dtype=np.float32,
    )


def warmup_pair(month: int, occupants: float) -> tuple[float, float]:
    """Choose a legal per-zone warm-up pair from current occupancy and month."""
    if occupants <= 0:
        return UNOCCUPIED_PAIR
    if month in (6, 7, 8, 9):
        return SUMMER_OCCUPIED_PAIR
    return WINTER_OCCUPIED_PAIR


def current_time(obs: dict[str, float], step: int) -> dict[str, int]:
    """Use simulated time from the observation, never host clock time."""
    return {**{key: int(obs[key]) for key in ("month", "day_of_month", "hour")},
            "minute": (step % 4) * 15}


def zone_state(obs: dict[str, float], zone_number: int, pair: tuple[float, float],
               cooling_energy_since_decision_j: float,
               heating_energy_since_decision_j: float,
               zone_reheat_electric_j: float) -> dict:
    """Capture the measured variables relevant to one DARLIN decision."""
    suffix = f"space{zone_number}"
    return {
        "outdoor_c": obs["outdoor_temperature"],
        "indoor_c": obs[f"air_temperature_{suffix}"],
        "heating_setpoint_c": pair[0],
        "cooling_setpoint_c": pair[1],
        "cooling_energy_j_building": cooling_energy_since_decision_j,
        "heating_energy_j_building": heating_energy_since_decision_j,
        "reheat_electric_energy_j_zone": zone_reheat_electric_j,
        "hvac_power_w_building": obs["HVAC_electricity_demand_rate"],
        "relative_humidity_pct": obs[f"air_humidity_{suffix}"],
        "occupants": obs[f"air_occ_{suffix}"],
        "new_pmv": obs[f"new_pmv_{suffix}"],
    }


def zone_layout_solar_context(zone: str) -> str:
    """Describe model geometry while avoiding unsupported solar assumptions."""
    layout = ZONE_LAYOUT_SOLAR_CONTEXT.get(zone, "zone geometry not specified")
    return (
        f"{zone}: {layout}. The model Building North Axis is set to 30 degrees; "
        "front/right/back/left are model-face labels, not asserted compass bearings. "
        "Solar gains vary with sun position and weather, but this prompt has no "
        "direct solar-radiation or sun-position measurement. Treat solar gain as "
        "context only; do not assume fixed morning/afternoon exposure or invent shading."
    )


def build_query(zone: str, states: list[dict], time: dict[str, int]) -> str:
    """Ask the official text DB for guidance based on trend, setpoint and weather."""
    now = states[-1]
    prior = states[-2] if len(states) > 1 else now
    indoor_change = now["indoor_c"] - prior["indoor_c"]
    outdoor_change = now["outdoor_c"] - prior["outdoor_c"]
    pmv_context = f"{now['new_pmv']:.2f}" if now["occupants"] > 0 else "unoccupied"
    return (
        f"Energy-efficient HVAC heating and cooling setpoint strategies for {zone} in a five-zone "
        f"office building, Current simulated time: "
        f"month {time['month']} day {time['day_of_month']} "
        f"time {time['hour']:02d}:{time['minute']:02d}. "
        f"Indoor {now['indoor_c']:.2f} C, {CONTROL_INTERVAL_MINUTES}-minute change "
        f"{indoor_change:+.2f} C; "
        f"outdoor {now['outdoor_c']:.2f} C, change {outdoor_change:+.2f} C; "
        f"heating/cooling setpoints {now['heating_setpoint_c']:.2f}/"
        f"{now['cooling_setpoint_c']:.2f} C; "
        f"humidity {now['relative_humidity_pct']:.1f}%; "
        f"building HVAC power {now['hvac_power_w_building']:.0f} W; "
        f"occupants {now['occupants']:.0f}; occupied new PMV {pmv_context}. "
        f"Zone layout and solar context: {zone_layout_solar_context(zone)} "
        "How can both setpoints save maximum HVAC energy while "
        "maintaining PMV within -0.5 to 0.5 in occupied periods?"
    )


def state_table(states: list[dict]) -> str:
    """Serialize recent states at the current control interval."""
    header = ("Ago | Outdoor C | Indoor C | RH % | Heating SP C | Cooling SP C | "
              "Building HVAC W | Building cooling J | Building heating J | "
              "Zone reheat electric J | Occupants | New PMV (ISO 7730)")
    rows = [header]
    for offset, state in enumerate(states):
        ago = (len(states) - offset - 1) * CONTROL_INTERVAL_MINUTES
        pmv_context = (f"{state['new_pmv']:.2f}"
                       if state["occupants"] > 0 else "unoccupied")
        rows.append(
            f"-{ago} min | {state['outdoor_c']:.2f} | {state['indoor_c']:.2f} | "
            f"{state['relative_humidity_pct']:.1f} | "
            f"{state['heating_setpoint_c']:.2f} | {state['cooling_setpoint_c']:.2f} | "
            f"{state['hvac_power_w_building']:.0f} | "
            f"{state['cooling_energy_j_building']:.0f} | "
            f"{state['heating_energy_j_building']:.0f} | "
            f"{state['reheat_electric_energy_j_zone']:.0f} | "
            f"{state['occupants']:.0f} | {pmv_context}"
        )
    return "\n".join(rows)


def extraction_prompt(query: str, hits: list[dict]) -> str:
    """Retain DARLIN's first LLM stage: distill retrieved papers into guidance."""
    literature = "\n\n".join(hit["text"] for hit in hits)
    return (
        "You are analyzing HVAC technical literature to extract actionable heating and cooling "
        "control strategies.\nCurrent conditions and question:\n"
        f"{query}\nRetrieved technical literature:\n{literature}\n"
        "Extract 3-5 concise, specific strategies relevant to these conditions. "
        "Save HVAC energy while maintaining PMV within -0.5 to 0.5 in occupied zones. "
        "Return only short bullet points."
    )


def control_prompt(zone: str, states: list[dict], time: dict[str, int],
                   insights: str, scheduled_pair: tuple[float, float]) -> str:
    """Ask for a continuous two-hour heating/cooling plan at the control cadence."""
    schema_pairs = ", ".join(
        f"[hsp{index}, csp{index}]" for index in range(1, PLAN_HORIZON + 1)
    )
    return (
        "# Heating and Cooling Setpoint Optimizer\n"
        "Role:You are an expert thermodynamic analyst and constraint-aware HVAC control engineer specializing in energy-efficient thermal comfort control of a five-zone VAV office building.;\n"
        "Goal:Maintain PMV thermal comfort in occupied zones within the PMV range [-0.5 to 0.5] while saving maximum HVAC energy.\n"
        f"Control only {zone}; all five zones are controlled independently.\n"
        "### Zone Layout and Solar-Exposure Context\n"
        f"{zone_layout_solar_context(zone)}\n"
        "PMV is unitless: negative means cold and positive means warm. The "
        "occupied PMV comfort interval is [-0.5, 0.5], matching evaluation. "
        "Use PMV for occupied-zone comfort decisions; unoccupied PMV is not a comfort "
        "violation. Heating and cooling setpoints influence PMV indirectly.\n"
        f"Each row below is {CONTROL_INTERVAL_MINUTES} minutes apart. Building total hvac electricity "
        "is shared across zones."
        f"{state_table(states)}\n"
        "Relevant literature guidance retrieved for the current conditions:\n"
        f"{insights}\n"
        "### Instructions for Applying This Knowledge\n"
        "1. Identify applicable strategies: determine which retrieved strategies "
        "directly apply to the current PMV, occupancy, and thermal trends.\n"
        "2. Adapt to current context: use applicable literature principles while "
        "accounting for current indoor/outdoor conditions and observed trends.\n"
        "3. Prioritize the stated objective and action constraints when selecting "
        "setpoints.\n"
        "4. Cite the applicable literature-based strategy in the reason field; "
        "do not claim a strategy that is not present in the retrieved guidance.\n"
        "Choose both heating and cooling setpoints. "
        "Adjust HSP and CSP when PMV, occupancy or thermal trends justify it. Account "
        "for thermal lag, zone layout and solar context, weather and recent setpoint response. Explain the "
        "expected PMV and energy effect of the current and future pairs.\n"
        f"Heating setpoint may be ANY finite real value in [{HSP_BOUNDS[0]}, "
        f"{HSP_BOUNDS[1]}] C. Cooling setpoint may be ANY finite real value in "
        f"[{CSP_BOUNDS[0]}, {CSP_BOUNDS[1]}] C. Heating must not exceed cooling. "
        "Every action plan satisfy these allowed hsp/csp ranges and the heating<cooling pair order. "
        "constraints."
        f"Currently applied heating/cooling pair: "
        f"[{states[-1]['heating_setpoint_c']:.4f}, "
        f"{states[-1]['cooling_setpoint_c']:.4f}] C. "
        f"Pair scheduled to be applied at this decision boundary: "
        f"[{scheduled_pair[0]:.4f}, {scheduled_pair[1]:.4f}] C.\n"
        f"Choose {PLAN_HORIZON} heating/cooling pairs for successive "
        f"{CONTROL_INTERVAL_MINUTES}-minute intervals spanning two hours. "
        "Only the first pair of this plan is actionable; the next decision replans."
        f"Return valid JSON only with exactly {PLAN_HORIZON} [heating, cooling] pairs and "
        "a nonempty short concise reason explaining both controls and citing the applicable "
        "retrieved strategy. Use this output schema, with your own selected values:\n"
        f'{{"optimal_setpoint_pairs": [{schema_pairs}], "reason": "..."}}'
    )


def validate_pair(raw: object) -> tuple[float, float]:
    """Accept exact continuous values only when both lie within Sinergym bounds."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("Each action must be [heating_setpoint, cooling_setpoint]")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in raw):
        raise ValueError("Both setpoints must be finite numbers")
    heating, cooling = map(float, raw)
    if not HSP_BOUNDS[0] <= heating <= HSP_BOUNDS[1]:
        raise ValueError(f"Heating setpoint {heating} outside {HSP_BOUNDS}")
    if not CSP_BOUNDS[0] <= cooling <= CSP_BOUNDS[1]:
        raise ValueError(f"Cooling setpoint {cooling} outside {CSP_BOUNDS}")
    if heating > cooling:
        raise ValueError("Heating setpoint exceeds cooling setpoint")
    return heating, cooling


def validate_plan(response: object) -> list[tuple[float, float]]:
    """Check the full continuous plan against actuator bounds and pair order."""
    if not isinstance(response, dict):
        raise ValueError("LLM output must be a JSON object")
    raw_plan = response.get("optimal_setpoint_pairs")
    if not isinstance(raw_plan, list) or len(raw_plan) != PLAN_HORIZON:
        raise ValueError(f"Expected {PLAN_HORIZON} heating/cooling pairs")
    if not isinstance(response.get("reason"), str) or not response["reason"].strip():
        raise ValueError("Expected a nonempty reason")
    plan = []
    for raw_pair in raw_plan:
        plan.append(validate_pair(raw_pair))
    return plan


def classify_invalid_plan(error: str) -> str:
    """Split invalid LLM plans into bounds, crossed-pair, or format failures."""
    message = error.lower()
    if "outside" in message:
        return "out_of_bounds"
    if "exceeds cooling" in message:
        return "crossed_setpoints"
    return "malformed_or_schema"


def sanitize_fallback_pair(heating: float, cooling: float) -> tuple[float, float]:
    """Clip fallback values to actuator limits and repair crossed setpoints."""
    try:
        heating = float(np.clip(heating, HSP_BOUNDS[0], HSP_BOUNDS[1]))
        cooling = float(np.clip(cooling, CSP_BOUNDS[0], CSP_BOUNDS[1]))
        if not math.isfinite(heating) or not math.isfinite(cooling) or heating >= cooling:
            return UNOCCUPIED_PAIR
        return validate_pair((heating, cooling))
    except (TypeError, ValueError):
        return UNOCCUPIED_PAIR


def pmv_aware_fallback_pair(obs: dict[str, float], zone_number: int
                            ) -> tuple[float, float]:
    """Mirror the proposed controller's per-zone occupancy/PMV fallback."""
    month = int(obs.get("month", 1) or 1)
    occupants = float(obs.get(f"air_occ_space{zone_number}", 0.0) or 0.0)
    if occupants <= 0.0:
        return UNOCCUPIED_PAIR

    low, high = (SUMMER_OCCUPIED_PAIR if month in (6, 7, 8, 9)
                 else WINTER_OCCUPIED_PAIR)
    temperature = float(obs.get(f"air_temperature_space{zone_number}", float("nan")))
    pmv_value = float(obs.get(f"new_pmv_space{zone_number}", float("nan")))
    pmv = pmv_value if math.isfinite(pmv_value) else None
    heating, cooling = low, high
    if pmv is not None and pmv < PMV_LOWER:
        heating = max(low, temperature + 0.5)
    elif pmv is not None and pmv > PMV_UPPER:
        heating = HSP_BOUNDS[0]
        cooling = min(high, temperature - 0.5)
    return sanitize_fallback_pair(heating, cooling)


class HeatingCoolingGenerator(DarlinGenerator):
    """Call either OpenRouter or OpenAI and parse continuous pair plans."""

    def __init__(self, *args, api_provider: str = "openrouter",
                 call_log_path: Path | None = None, **kwargs):
        """Keep the official credential setup and record the selected endpoint."""
        super().__init__(*args, **kwargs)
        self.api_provider = api_provider
        self.call_log_path = call_log_path
        self.api_events: list[dict] = []
        self.call_context: dict = {}

    def set_call_context(self, step: int, zone: str, stage: str, attempt: int = 1) -> None:
        """Label the next model call so extraction and control remain distinct."""
        self.call_context = {"step": step, "zone": zone, "stage": stage,
                             "attempt": attempt}

    def mark_control_validation(self, valid: bool, error: str | None = None) -> None:
        """Attach JSON/action validation to the most recent successful HTTP call."""
        if not self.api_events or self.api_events[-1]["stage"] != "control":
            return
        event = self.api_events[-1]
        event["output_valid"] = valid
        event["invalid_output_reason"] = error
        event["invalid_action_category"] = (
            None if valid else classify_invalid_plan(error or "")
        )
        if self.call_log_path is not None:
            write_jsonl(self.call_log_path, {
                "event": "validation", "call_id": event["call_id"],
                "output_valid": valid, "invalid_output_reason": error,
                "invalid_action_category": event["invalid_action_category"],
            })

    def _call_chat_api(self, prompt_text: str) -> str:
        """Print and persist every prompt, response, failure, usage and latency."""
        context = self.call_context
        call_id = len(self.api_events) + 1
        event = {
            "call_id": call_id, "step": context.get("step"),
            "zone": context.get("zone"), "stage": context.get("stage"),
            "attempt": context.get("attempt", 1),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "api_provider": self.api_provider,
            "api_base_url": self.api_base,
            "requested_model": self.model_name, "response_model": None,
            "generation_id": None,
            "temperature": MODEL_TEMPERATURE,
            "reasoning_effort": REASONING_EFFORT,
            "max_tokens": MODEL_MAX_TOKENS,
            "prompt": prompt_text, "raw_response": None,
            "api_success": False, "api_error": None, "api_latency_sec": None,
            "finish_reason": None, "input_tokens": None,
            "output_tokens": None, "total_tokens": None,
            "reasoning_tokens": None, "api_cost_usd": None,
            "cost_source": None, "output_valid": None,
            "invalid_output_reason": None, "invalid_action_category": None,
        }
        print(f"\n=== LLM PROMPT step={event['step']} zone={event['zone']} "
              f"stage={event['stage']} attempt={event['attempt']} ===\n"
              f"{prompt_text}\n=== END PROMPT ===", flush=True)
        if self.call_log_path is not None:
            write_jsonl(self.call_log_path, {"event": "request", **event})
        payload = {
            "model": self.model_name,
            "stream": False,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": MODEL_TEMPERATURE,
        }
        if self.api_provider == "openrouter":
            # Keep the existing OpenRouter Gemma request settings unchanged.
            payload.update({
                "max_tokens": MODEL_MAX_TOKENS,
                "reasoning": {"effort": REASONING_EFFORT},
                "provider": {"require_parameters": True},
            })
        else:
            # OpenAI uses a snake_case reasoning field in Chat Completions.
            payload["max_completion_tokens"] = MODEL_MAX_TOKENS
            payload["reasoning_effort"] = REASONING_EFFORT
        started = wall_time.perf_counter()
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            reply = choice["message"]["content"].strip()
            usage = body.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            detail_reasoning = (details.get("reasoning_tokens")
                                if isinstance(details, dict) else None)
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            reported_cost = usage.get("cost")
            cost_source = (f"{self.api_provider}_usage_cost"
                           if reported_cost is not None else None)
            if (reported_cost is None and self.api_provider == "openai"
                    and OPENAI_INPUT_PRICE_PER_1M is not None
                    and OPENAI_OUTPUT_PRICE_PER_1M is not None
                    and input_tokens is not None and output_tokens is not None):
                reported_cost = (
                    int(input_tokens) * OPENAI_INPUT_PRICE_PER_1M / 1_000_000.0
                    + int(output_tokens) * OPENAI_OUTPUT_PRICE_PER_1M / 1_000_000.0
                )
                cost_source = "configured_openai_token_price"
            event.update(
                api_success=True, raw_response=reply,
                response_model=body.get("model"),
                generation_id=body.get("id"),
                finish_reason=choice.get("finish_reason"),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                reasoning_tokens=(detail_reasoning if detail_reasoning is not None
                                  else usage.get("reasoning_tokens")),
                api_cost_usd=reported_cost,
                cost_source=cost_source,
            )
            return reply
        except Exception as exc:
            event["api_error"] = f"{type(exc).__name__}: {exc}"
            event["api_cost_usd"] = 0.0
            event["cost_source"] = "api_failure_assumed_zero"
            raise
        except KeyboardInterrupt:
            # Preserve a distinct interrupted-call record; the finally block
            # still appends its elapsed time and writes the audit event.
            event["api_error"] = "KeyboardInterrupt: interrupted during API request"
            event["api_cost_usd"] = 0.0
            event["cost_source"] = "api_failure_assumed_zero"
            raise
        finally:
            event["api_latency_sec"] = wall_time.perf_counter() - started
            self.api_events.append(event)
            if self.call_log_path is not None:
                write_jsonl(self.call_log_path, {"event": "result", **event})
            print(f"[LLM {event['stage']}] step={event['step']} zone={event['zone']} "
                  f"api_success={event['api_success']} "
                  f"latency={event['api_latency_sec']:.2f}s", flush=True)

    def generate_pair_response(self, prompt: str,
                               fallback_pair: tuple[float, float]) -> dict:
        """Make one control call, then return either its valid plan or fallback."""
        reply = None
        reply_received = False
        try:
            self.call_context["attempt"] = 1
            reply = self._call_chat_api(prompt)
            reply_received = True
            try:
                result = json.loads(reply)
            except json.JSONDecodeError:
                result = self.extract_json_from_text(reply)
            plan = validate_plan(result)
            self.mark_control_validation(True)
            return {"optimal_setpoint_pairs": [list(pair) for pair in plan],
                    "reason": result["reason"], "raw_response": reply,
                    "attempt": 1, "fallback_activated": False,
                    "fallback_type": None, "fallback_reason": None}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if reply_received:
                self.mark_control_validation(False, error)
            fallback_type = "invalid_output" if reply_received else "api_failure"
            try:
                fallback = validate_pair(fallback_pair)
            except (TypeError, ValueError):
                fallback = UNOCCUPIED_PAIR
            reason = (f"PMV-aware {fallback_type} fallback; {error}")
            print(f"[ACTION FALLBACK] {reason} pair={fallback}; no LLM retry",
                  flush=True)
            return {"optimal_setpoint_pairs": [list(fallback)] * PLAN_HORIZON,
                    "reason": reason, "raw_response": reply,
                    "attempt": 1, "fallback_activated": True,
                    "fallback_type": fallback_type,
                    "fallback_reason": error}


def write_jsonl(path: Path, record: dict) -> None:
    """Append every prompt, retrieved hit and action for later audit."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def finite_json(value):
    """Represent unavailable per-zone metrics as JSON null, never NaN tokens."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


CALL_COLUMNS = (
    "call_id", "step", "zone", "stage", "attempt", "timestamp_utc",
    "api_provider", "api_base_url", "requested_model", "response_model", "generation_id",
    "temperature", "reasoning_effort",
    "max_tokens", "api_success", "api_error", "api_latency_sec",
    "finish_reason", "input_tokens", "output_tokens", "total_tokens",
    "reasoning_tokens", "api_cost_usd", "cost_source", "output_valid",
    "invalid_output_reason", "invalid_action_category", "prompt", "raw_response",
)
DECISION_COLUMNS = (
    "decision_id", "step", "zone", "simulated_time_json", "status",
    "error", "output_valid", "extraction_fallback", "fallback_activated",
    "fallback_type", "fallback_reason", "control_attempts", "api_latency_sec",
    "decision_latency_sec",
    "action_plan_json", "reason", "proposed_pair_json", "applied_pair_json",
    "applied_from_decision_id", "proposal_applied_now",
    "proposal_eventually_applied", "proposal_applied_step",
    "proposal_superseded_by_fallback", "proposal_superseded_at_step",
    "sim_step_completed",
)


def write_csv_records(path: Path, columns: tuple[str, ...], records: list[dict]) -> None:
    """Write a stable-header CSV even when an offline smoke test has no LLM calls."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def operational_summary(api_events: list[dict], decisions: list[dict],
                        completed_steps: int, model: str, run_completed: bool,
                        mode: str = "run", provider: str = "openrouter",
                        step_runtimes: list[float] | None = None) -> dict:
    """Use the proposed controller's rates, accounting for DARLIN's action delay.

    Application means a proposed first pair reached a completed Sinergym step,
    either immediately or at the following decision. The final queued proposal
    can remain unapplied when the test ends; it is counted separately.
    """
    successful_calls = [event for event in api_events if event["api_success"]]
    successful_controls = [event for event in successful_calls
                           if event["stage"] == "control"]
    extraction_calls = [event for event in api_events if event["stage"] == "extraction"]
    control_calls = [event for event in api_events if event["stage"] == "control"]
    extraction_failures = [event for event in extraction_calls
                           if not event["api_success"]]
    control_failures = [event for event in control_calls if not event["api_success"]]
    invalid_controls = [event for event in successful_controls
                        if event["output_valid"] is not True]
    out_of_bounds_controls = [event for event in invalid_controls
                              if event.get("invalid_action_category") == "out_of_bounds"]
    crossed_controls = [event for event in invalid_controls
                        if event.get("invalid_action_category") == "crossed_setpoints"]
    malformed_controls = [event for event in invalid_controls
                          if event.get("invalid_action_category") == "malformed_or_schema"]
    valid_decisions = [row for row in decisions if row["output_valid"] is True]
    active_valid_decisions = [row for row in valid_decisions
                              if not row.get("proposal_superseded_by_fallback")]
    applied = [row for row in active_valid_decisions
               if row["proposal_eventually_applied"]]
    immediate = [row for row in applied if row["proposal_applied_now"]]
    fallback_decisions = [row for row in decisions if row.get("fallback_activated")]
    invalid_fallbacks = [row for row in fallback_decisions
                         if row.get("fallback_type") == "invalid_output"]
    api_fallbacks = [row for row in fallback_decisions
                     if row.get("fallback_type") == "api_failure"]
    fallback_applied = [row for row in fallback_decisions
                        if row["proposal_eventually_applied"]]
    latencies = [float(event["api_latency_sec"]) for event in api_events
                 if event["api_latency_sec"] is not None]
    decision_latencies = [float(row["decision_latency_sec"]) for row in decisions
                          if row.get("decision_latency_sec") is not None]
    cost_known = bool(api_events) and all(
        event["api_cost_usd"] is not None for event in api_events)
    total_cost = (sum(float(event["api_cost_usd"]) for event in api_events)
                  if cost_known else None)
    tokens_known = bool(api_events) and all(
        event["input_tokens"] is not None and event["output_tokens"] is not None
        for event in api_events)
    token_events = [event for event in api_events
                    if event["input_tokens"] is not None
                    and event["output_tokens"] is not None]
    step_runtimes = [float(value) for value in (step_runtimes or [])]
    simulated_hours = completed_steps * 0.25

    def rate(count: int, denominator: int) -> float | None:
        return 100.0 * count / denominator if denominator else None

    return {
        "API Provider": provider,
        "Requested Model": model,
        "Response Model(s)": ", ".join(sorted({str(event["response_model"])
                                           for event in successful_calls
                                           if event["response_model"]})) or None,
        "Temperature": MODEL_TEMPERATURE,
        "Reasoning Effort": REASONING_EFFORT,
        "Max Output Tokens": MODEL_MAX_TOKENS,
        "Execution Mode": mode,
        "Run Completed": run_completed,
        "Completed Control Steps": completed_steps,
        "Simulated Hours": simulated_hours,
        "Total LLM Calls": len(api_events),
        "Extraction Calls": sum(event["stage"] == "extraction" for event in api_events),
        "Control Calls": sum(event["stage"] == "control" for event in api_events),
        "Extraction API Failures": len(extraction_failures),
        "Extraction API Failure Rate (%)": rate(len(extraction_failures),
                                                  len(extraction_calls)),
        "Control API Failures": len(control_failures),
        "Control API Failure Rate (%)": rate(len(control_failures),
                                               len(control_calls)),
        "Extraction Fallbacks": sum(bool(row.get("extraction_fallback"))
                                    for row in decisions),
        "Mean Latency (s)": float(np.mean(latencies)) if latencies else None,
        "P95 Latency (s)": float(np.percentile(latencies, 95)) if latencies else None,
        "Mean Simulation Step Runtime (s)": (float(np.mean(step_runtimes))
                                             if step_runtimes else None),
        "P95 Simulation Step Runtime (s)": (float(np.percentile(step_runtimes, 95))
                                            if step_runtimes else None),
        "Mean Decision Latency (s)": (float(np.mean(decision_latencies))
                                      if decision_latencies else None),
        "P95 Decision Latency (s)": (float(np.percentile(decision_latencies, 95))
                                     if decision_latencies else None),
        "Average Input Tokens/Request": (sum(int(event["input_tokens"])
                                             for event in api_events) / len(api_events)
                                         if tokens_known else None),
        "Average Output Tokens/Request": (sum(int(event["output_tokens"])
                                              for event in api_events) / len(api_events)
                                          if tokens_known else None),
        "Total Input Tokens": sum(int(event["input_tokens"]) for event in token_events),
        "Total Output Tokens": sum(int(event["output_tokens"]) for event in token_events),
        "Total Tokens": sum(int(event["total_tokens"] or 0) for event in token_events),
        "Token Usage Coverage (%)": rate(len(token_events), len(api_events)),
        "API Cost/Request (USD)": (total_cost / len(api_events)
                                   if total_cost is not None else None),
        "API Cost/24 h (USD)": (total_cost * 24.0 / simulated_hours
                               if total_cost is not None and simulated_hours else None),
        "API Failure Rate (%)": rate(len(api_events) - len(successful_calls),
                                     len(api_events)),
        "Invalid Output Rate (%)": rate(len(invalid_controls),
                                        len(successful_controls)),
        "Out-of-Bounds Plan Rate (%)": rate(len(out_of_bounds_controls),
                                             len(successful_controls)),
        "Crossed-Setpoint Plan Rate (%)": rate(len(crossed_controls),
                                                len(successful_controls)),
        "Malformed/Schema Plan Rate (%)": rate(len(malformed_controls),
                                                len(successful_controls)),
        "Fallback Activation Rate (%)": rate(len(fallback_decisions), len(decisions)),
        "Invalid-Output Fallback Rate (%)": rate(len(invalid_fallbacks),
                                                  len(decisions)),
        "API-Failure Fallback Rate (%)": rate(len(api_fallbacks), len(decisions)),
        "LLM Action Application Rate (%)": rate(len(applied),
                                                  len(active_valid_decisions)),
        "Same-Decision Application Rate (%)": rate(len(immediate),
                                                     len(active_valid_decisions)),
        "Fallback Action Application Rate (%)": rate(len(fallback_applied),
                                                       len(fallback_decisions)),
        "Fallback Success Rate (%)": rate(len(fallback_applied),
                                          len(fallback_decisions)),
        "Total API Cost (USD)": total_cost,
        "Cost Coverage (%)": rate(sum(event["cost_source"] is not None
                                      for event in successful_calls),
                                  len(successful_calls)),
        "Valid LLM Actions": len(valid_decisions),
        "Exactly Applied LLM Actions": len(applied),
        "Delayed Applied Actions": len(applied) - len(immediate),
        "Pending Unapplied Actions": len(active_valid_decisions) - len(applied),
        "Superseded LLM Actions": sum(bool(row.get("proposal_superseded_by_fallback"))
                                       for row in valid_decisions),
        "Successful API Calls": len(successful_calls),
        "Invalid API Outputs": len(invalid_controls),
        "Out-of-Bounds Plans": len(out_of_bounds_controls),
        "Crossed-Setpoint Plans": len(crossed_controls),
        "Malformed/Schema Plans": len(malformed_controls),
        "Fallback Activations": len(fallback_decisions),
        "Invalid-Output Fallbacks": len(invalid_fallbacks),
        "API-Failure Fallbacks": len(api_fallbacks),
        "Fallback Actions Applied": len(fallback_applied),
        "Pending Fallback Actions": len(fallback_decisions) - len(fallback_applied),
        "Failed Control Decisions": sum(row["status"] == "failed" for row in decisions),
        "Interrupted Control Decisions": sum(row["status"] == "interrupted"
                                               for row in decisions),
        "Control Decisions": len(decisions),
    }


def write_operational_artifacts(out: Path, api_events: list[dict],
                                decisions: list[dict], completed_steps: int,
                                model: str, run_completed: bool,
                                mode: str = "run",
                                provider: str = "openrouter",
                                step_runtimes: list[float] | None = None) -> dict:
    """Save call/decision CSVs and print the cumulative operational summary."""
    write_csv_records(out / "operational_calls.csv", CALL_COLUMNS, api_events)
    write_csv_records(out / "operational_decisions.csv", DECISION_COLUMNS, decisions)
    summary = operational_summary(api_events, decisions, completed_steps,
                                  model, run_completed, mode, provider,
                                  step_runtimes)
    write_csv_records(out / "operational_summary.csv", tuple(summary), [summary])
    print("\n=== OPERATIONAL SUMMARY ===", flush=True)
    for name, value in summary.items():
        print(f"{name:<38} | {value if value is not None else 'N/A'}", flush=True)
    for name in ("operational_calls.csv", "operational_decisions.csv",
                 "operational_summary.csv"):
        print(f"[SAVED] {out / name}", flush=True)
    return summary


def compute_pmv_energy_only_metrics(observations, start: list[int], end: list[int]) -> dict:
    """Report HVAC kWh and occupied ISO-derived new PMV, not temperature bands."""
    from pmv_energy_metrics import (
        PMV_LOWER, PMV_UPPER, df_in_date_range, pmv_stats,
        total_hvac_energy_kwh, zone_pmv_for_occupied_steps,
    )

    # The raw monitor contains EnergyPlus Fanger PMV, not the YAML's custom
    # new_pmv_space1..5. Derive the custom values and route them through the
    # shared occupancy filtering/statistics implementation under its PMV names.
    observations = observations.copy()
    for zone_number in range(1, 6):
        new_column = f"new_pmv_space{zone_number}"
        observations[new_column] = [
            calculate_new_pmv(temperature, int(month))
            for temperature, month in zip(
                observations[f"air_temperature_space{zone_number}"],
                observations["month"],
            )
        ]
        observations[f"pmv_space{zone_number}"] = observations[new_column]
    period = df_in_date_range(observations, start, end)
    per_zone = {}
    occupied_arrays = []
    for zone_number in range(1, 6):
        pmv, count = zone_pmv_for_occupied_steps(period, zone_number)
        if pmv.size:
            occupied_arrays.append(pmv)
        pmv_mean, pmv_std, violation_pct = pmv_stats(pmv)
        per_zone[f"Z{zone_number}"] = {
            "occupied_pmv_steps": count,
            "pmv_mean": pmv_mean,
            "pmv_std": pmv_std,
            "pmv_violation_pct": violation_pct,
            "mean_abs_pmv": float(np.mean(np.abs(pmv))) if pmv.size else None,
            "mean_pmv_excess_outside_band": float(np.mean(
                np.maximum(PMV_LOWER - pmv, 0.0) +
                np.maximum(pmv - PMV_UPPER, 0.0)
            )) if pmv.size else None,
        }
    all_pmv = np.concatenate(occupied_arrays) if occupied_arrays else np.array([])
    pmv_mean_all, pmv_std_all, violation_pct_all = pmv_stats(all_pmv)
    return {
        "metric_scope": "HVAC electrical energy and occupied-zone new PMV only",
        "pmv_definition": NEW_PMV_DEFINITION,
        "pmv_source_columns": [f"new_pmv_space{i}" for i in range(1, 6)],
        "date_start": start, "date_end": end,
        "rows_in_range": int(len(period)),
        "total_hvac_energy_kwh": total_hvac_energy_kwh(period),
        "pmv_band": [PMV_LOWER, PMV_UPPER],
        "occupied_pmv_steps_all": int(all_pmv.size),
        "pmv_mean_all": pmv_mean_all,
        "pmv_std_all": pmv_std_all,
        "pmv_violation_pct_all": violation_pct_all,
        "mean_abs_pmv_all": float(np.mean(np.abs(all_pmv))) if all_pmv.size else None,
        "mean_pmv_excess_outside_band_all": float(np.mean(
            np.maximum(PMV_LOWER - all_pmv, 0.0) +
            np.maximum(all_pmv - PMV_UPPER, 0.0)
        )) if all_pmv.size else None,
        "zone_metrics": per_zone,
    }


def parse_args() -> argparse.Namespace:
    """Expose offline verification, environment smoke test, and configured run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Offline RAG/prompt check; no EnergyPlus or API")
    parser.add_argument("--print-prompts", action="store_true",
                        help="With --dry-run, print sample extraction and control prompts")
    parser.add_argument("--smoke-env", action="store_true", help="Two Sinergym steps, fixed action, no API")
    parser.add_argument("--run", action="store_true",
                        help="Run the configured Arizona period with DARLIN LLM control")
    parser.add_argument("--provider", choices=("openrouter", "openai"),
                        default="openrouter",
                        help="LLM API provider; defaults to the existing OpenRouter setup")
    parser.add_argument("--api-base-url", default=None,
                        help="Optional API base URL override for the selected provider")
    parser.add_argument("--model", default=None,
                        help="Model name; OpenAI requires an explicit model name")
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    if sum((args.dry_run, args.smoke_env, args.run)) != 1:
        parser.error("Choose exactly one of --dry-run, --smoke-env, or --run")
    if args.print_prompts and not args.dry_run:
        parser.error("--print-prompts requires --dry-run")
    return args


def main() -> None:
    """Execute a mode without ever reading a key during offline checks."""
    args = parse_args()
    if args.dry_run:
        retriever = TextRetriever()
        states = [{"outdoor_c": 38.0, "indoor_c": 25.0,
                   "heating_setpoint_c": 18.0, "cooling_setpoint_c": 26.0,
                   "cooling_energy_j_building": 50000.0,
                   "heating_energy_j_building": 0.0,
                   "hvac_power_w_building": 1800.0,
                   "relative_humidity_pct": 35.0,
                   "reheat_electric_energy_j_zone": 0.0, "occupants": 12.0,
                   "new_pmv": 0.4} for _ in range(HISTORY_SAMPLES)]
        time = {"month": 7, "day_of_month": 1, "hour": 12, "minute": 0}
        query = build_query(ZONES[0], states, time)
        hits = retriever.search(query)
        scheduled_pair = (18.0, 26.0)
        prompt = control_prompt(ZONES[0], states, time,
                                "[offline: extraction skipped]", scheduled_pair)
        sample_plan = {"optimal_setpoint_pairs":
                       [[18.375, 26.125] for _ in range(PLAN_HORIZON)],
                       "reason": "Offline validation of continuous pairs."}
        validated = validate_plan(sample_plan)
        print(json.dumps({"faiss_vectors": retriever.index.ntotal,
                          "top_source": hits[0].get("source") if hits else None,
                          "hit_count": len(hits), "prompt_chars": len(prompt),
                          "model": args.model or DEFAULT_MODEL,
                          "reasoning_effort": REASONING_EFFORT,
                          "action_shape": list(action_array({z: validated[0] for z in ZONES}).shape),
                          "example_continuous_pair": list(validated[0])}, indent=2))
        if args.print_prompts:
            print("\n=== LITERATURE EXTRACTION PROMPT (offline sample) ===\n")
            print(extraction_prompt(query, hits))
            print("\n=== CONTROL PROMPT (offline sample; extraction skipped) ===\n")
            print(prompt)
        return

    # Select credentials, model and endpoint together so provider keys cannot
    # accidentally be sent to the other provider. A pasted source key takes
    # precedence; otherwise the matching environment/.env variable is used.
    provider_settings = {
        "openrouter": {
            "key_env": "OPENROUTER_API_KEY",
            "hardcoded_key": OPENROUTER_API_KEY,
            "default_base": DEFAULT_API_BASE_URL,
            "default_model": DEFAULT_MODEL,
        },
        "openai": {
            "key_env": "OPENAI_API_KEY",
            "hardcoded_key": OPENAI_API_KEY,
            "default_base": DEFAULT_OPENAI_API_BASE_URL,
            "default_model": DEFAULT_OPENAI_MODEL,
        },
    }
    provider = provider_settings[args.provider]
    if args.run:
        hardcoded_key = provider["hardcoded_key"].strip()
        if hardcoded_key:
            os.environ[provider["key_env"]] = hardcoded_key
        else:
            load_dotenv(ROOT / ".env")
        if not os.getenv(provider["key_env"]):
            raise RuntimeError(
                f"{provider['key_env']} is missing; paste it into the matching "
                "HARDCODED_*_API_KEY setting or configure it in .env/environment"
            )
    selected_model = args.model or provider["default_model"]
    selected_model = selected_model or DEFAULT_MODEL
    selected_api_base = args.api_base_url or provider["default_base"]
    # Set the evaluation dates here; EnergyPlus uses
    # (start_day, start_month, start_year, end_day, end_month, end_year).
    runperiod = (1, 1, 1991, 7, 1, 1991)
    start_date = date(runperiod[2], runperiod[1], runperiod[0])
    end_date = date(runperiod[5], runperiod[4], runperiod[3])
    run_days = (end_date - start_date).days + 1
    start_tag = start_date.strftime("%b%d").lower()
    end_tag = end_date.strftime("%b%d").lower()
    retriever = TextRetriever() if args.run else None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.output / f"darlin_text_5zone_arizona_winerWEEK_7jandays_{start_tag}_to_{end_tag}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    decisions_path = out / "decisions.jsonl"
    steps_path = out / "steps.csv"
    generator = (HeatingCoolingGenerator(
        api_key_env=provider["key_env"], api_base_url=selected_api_base,
        model_name=selected_model, llm_provider="openai", api_provider=args.provider,
        call_log_path=out / "llm_calls.jsonl") if args.run else None)
    api_events = generator.api_events if generator is not None else []
    decision_rows: list[dict] = []
    config = {"timesteps_per_hour": 4, "runperiod": runperiod}
    env = CSVLogger(LoggerWrapper(gym.make(ENV_ID, env_name=out.name,
                                           building_config=config)))
    workspace = Path(env.get_wrapper_attr("workspace_path"))
    # The smoke-only pair checks that EnergyPlus accepts non-grid values on both actuators.
    smoke_pair = (18.375, 26.125)
    pairs = {zone: smoke_pair for zone in ZONES}
    pending = {zone: None for zone in ZONES}  # Official plugin's last_setpoints.
    pending_rows = {zone: None for zone in ZONES}  # Provenance of queued first pairs.
    buffers = {zone: deque(maxlen=HISTORY_SAMPLES) for zone in ZONES}
    cooling_energy_since_decision_j = 0.0
    heating_energy_since_decision_j = 0.0
    reheat_electric_since_decision_j = {zone: 0.0 for zone in ZONES}
    rows: list[dict] = []
    run_completed = False
    try:
        observation, _ = env.reset(seed=0)
        names = list(env.get_wrapper_attr("observation_variables"))
        if not args.smoke_env:
            # Initialize from the actual reset occupancy, then refresh the
            # warm-up action at every 15-minute observation until LLM control.
            reset_obs = as_obs_dict(observation, names)
            pairs = {zone: warmup_pair(int(reset_obs["month"]),
                                      reset_obs[f"air_occ_space{zi}"])
                     for zi, zone in enumerate(ZONES, start=1)}
        initial_pairs = {zone: list(pair) for zone, pair in pairs.items()}
        # Run all requested 15-minute steps; smoke mode remains two steps.
        limit = 2 if args.smoke_env else run_days * 24 * 4
        for step in range(limit):
            step_started = wall_time.perf_counter()
            obs = as_obs_dict(observation, names)
            step_decision_rows: list[dict] = []
            step_application_links: list[dict] = []
            if step % CONTROL_STEPS == 0:
                time = current_time(obs, step)
                for zi, zone in enumerate(ZONES, start=1):
                    states = buffers[zone]
                    states.append(zone_state(
                        obs, zi, pairs[zone], cooling_energy_since_decision_j,
                        heating_energy_since_decision_j,
                        reheat_electric_since_decision_j[zone]))
                    if args.smoke_env or len(states) < HISTORY_SAMPLES:
                        continue
                    assert retriever is not None and generator is not None
                    decision = {
                        "decision_id": f"{step}:{zone}", "step": step, "zone": zone,
                        "simulated_time_json": json.dumps(time), "status": "pending",
                        "error": None, "output_valid": None,
                        "extraction_fallback": False,
                        "fallback_activated": False, "fallback_type": None,
                        "fallback_reason": None,
                        "control_attempts": 0,
                        "api_latency_sec": None, "decision_latency_sec": None,
                        "action_plan_json": None,
                        "reason": None, "proposed_pair_json": None,
                        "applied_pair_json": None,
                        "applied_from_decision_id": None,
                        "proposal_applied_now": False,
                        "proposal_eventually_applied": False,
                        "proposal_applied_step": None,
                        "proposal_superseded_by_fallback": False,
                        "proposal_superseded_at_step": None,
                        "sim_step_completed": False,
                    }
                    decision_rows.append(decision)
                    event_start = len(api_events)
                    decision_started = wall_time.perf_counter()
                    try:
                        query = build_query(zone, list(states), time)
                        hits = retriever.search(query)
                        extraction = extraction_prompt(query, hits)
                        generator.set_call_context(step, zone, "extraction")
                        try:
                            insights = generator._call_chat_api(extraction)
                        except Exception as exc:
                            # Match the official text RAG fallback: keep raw
                            # retrieved knowledge if insight extraction fails.
                            insights = "\n\n".join(hit["text"] for hit in hits)
                            decision["extraction_fallback"] = True
                            print(f"[EXTRACTION FALLBACK] step={step} zone={zone}: "
                                  f"{type(exc).__name__}: {exc}; using raw text",
                                  flush=True)
                        scheduled_pair = (pending[zone] if pending[zone] is not None
                                          else pairs[zone])
                        prompt = control_prompt(zone, list(states), time,
                                                insights, scheduled_pair)
                        fallback_pair = pmv_aware_fallback_pair(obs, zi)
                        generator.set_call_context(step, zone, "control")
                        result = generator.generate_pair_response(prompt, fallback_pair)
                        proposal = tuple(result["optimal_setpoint_pairs"][0])
                        # The final safety gate applies equally to valid LLM
                        # output, queued actions and deterministic fallback.
                        proposal = validate_pair(proposal)
                        if result["fallback_activated"]:
                            # Apply the PMV-aware fallback now and discard any
                            # stale queued action for this zone.
                            if pending_rows[zone] is not None:
                                pending_rows[zone]["proposal_superseded_by_fallback"] = True
                                pending_rows[zone]["proposal_superseded_at_step"] = step
                            applied = proposal
                            applied_source = decision
                            proposal_applied_now = True
                        else:
                            applied = (pending[zone] if pending[zone] is not None
                                       else proposal)
                            applied_source = pending_rows[zone] or decision
                            proposal_applied_now = pending_rows[zone] is None
                        applied = validate_pair(applied)
                        decision.update(
                            status=("fallback" if result["fallback_activated"]
                                    else "validated"),
                            output_valid=not result["fallback_activated"],
                            fallback_activated=result["fallback_activated"],
                            fallback_type=result["fallback_type"],
                            fallback_reason=result["fallback_reason"],
                            control_attempts=result["attempt"],
                            action_plan_json=json.dumps(result["optimal_setpoint_pairs"]),
                            reason=result["reason"],
                            proposed_pair_json=json.dumps(proposal),
                            applied_pair_json=json.dumps(applied),
                            applied_from_decision_id=applied_source["decision_id"],
                            proposal_applied_now=proposal_applied_now,
                        )
                        print(f"\n=== ACTION PLAN step={step} zone={zone} ===\n"
                              f"{json.dumps(result['optimal_setpoint_pairs'], indent=2)}\n"
                              f"Reason: {result['reason']}\n"
                              f"Proposed first pair: {proposal}; "
                              f"applied now: {applied} "
                              f"(from {applied_source['decision_id']})\n"
                              "=== END ACTION PLAN ===", flush=True)
                        write_jsonl(decisions_path, {
                            "step": step, "zone": zone, "simulated_time": time,
                            "query": query, "retrieved_hits": hits,
                            "extraction_prompt": extraction,
                            "insights": insights, "control_prompt": prompt,
                            "response": result, "valid_proposed_pair_c": proposal,
                            "extraction_fallback": decision["extraction_fallback"],
                            "fallback_activated": result["fallback_activated"],
                            "fallback_type": result["fallback_type"],
                            "fallback_reason": result["fallback_reason"],
                            "applied_pair_c": applied,
                            "applied_from_decision_id": applied_source["decision_id"],
                            "scheduled_pair_before_call_c": scheduled_pair,
                            "previous_applied_pair_c": pairs[zone],
                        })
                        pairs[zone] = applied
                        if result["fallback_activated"]:
                            pending[zone] = None
                            pending_rows[zone] = None
                        else:
                            pending[zone] = proposal
                            pending_rows[zone] = decision
                        step_decision_rows.append(decision)
                        step_application_links.append(applied_source)
                    except KeyboardInterrupt as exc:
                        decision["status"] = "interrupted"
                        decision["error"] = f"KeyboardInterrupt: {exc}"
                        print(f"[DECISION INTERRUPTED] step={step} zone={zone}",
                              flush=True)
                        raise
                    except Exception as exc:
                        decision["status"] = "failed"
                        decision["error"] = f"{type(exc).__name__}: {exc}"
                        print(f"[DECISION FAILED] step={step} zone={zone}: "
                              f"{decision['error']}", flush=True)
                        raise
                    finally:
                        events = api_events[event_start:]
                        decision["api_latency_sec"] = sum(
                            event["api_latency_sec"] or 0.0 for event in events)
                        decision["decision_latency_sec"] = (
                            wall_time.perf_counter() - decision_started)
                        decision["control_attempts"] = sum(
                            event["stage"] == "control" for event in events)
                cooling_energy_since_decision_j = 0.0
                heating_energy_since_decision_j = 0.0
                reheat_electric_since_decision_j = {zone: 0.0 for zone in ZONES}
            if not args.smoke_env and len(buffers[ZONES[0]]) < HISTORY_SAMPLES:
                # Refresh the seasonal occupancy-based warm-up on every step.
                for zi, zone in enumerate(ZONES, start=1):
                    pairs[zone] = warmup_pair(int(obs["month"]),
                                              obs[f"air_occ_space{zi}"])
            applied_action = action_array(pairs)
            next_observation, _reward, terminated, truncated, _ = env.step(applied_action)
            next_obs = as_obs_dict(next_observation, names)
            cooling_energy_since_decision_j += next_obs["cooling_coil_electricity_energy"]
            heating_energy_since_decision_j += next_obs["heating_electricity"]
            for zi, zone in enumerate(ZONES, start=1):
                reheat_electric_since_decision_j[zone] += next_obs[f"reheat_energy_space{zi}"]
            rows.append({"step": step, "step_runtime_sec":
                         wall_time.perf_counter() - step_started,
                         "month": obs["month"],
                         "day_of_month": obs["day_of_month"], "hour": obs["hour"],
                         "hvac_power_w_after": next_obs["HVAC_electricity_demand_rate"],
                         "hvac_energy_kwh_after":
                         next_obs["HVAC_electricity_demand_rate"] * 0.25 / 1000.0,
                         **{f"heating_sp_space{i}": pairs[zone][0]
                            for i, zone in enumerate(ZONES, start=1)},
                         **{f"cooling_sp_space{i}": pairs[zone][1]
                            for i, zone in enumerate(ZONES, start=1)},
                         **{f"new_pmv_space{i}_after": next_obs[f"new_pmv_space{i}"]
                            for i in range(1, 6)},
                         **{f"occupied_abs_new_pmv_space{i}_after":
                            abs(next_obs[f"new_pmv_space{i}"])
                            if next_obs[f"air_occ_space{i}"] > 0 else ""
                            for i in range(1, 6)},
                         **{f"occupants_space{i}_after": next_obs[f"air_occ_space{i}"]
                            for i in range(1, 6)}})
            for decision in step_decision_rows:
                decision["sim_step_completed"] = True
            for source in step_application_links:
                source["proposal_eventually_applied"] = True
                source["proposal_applied_step"] = step
            observation = next_observation
            if terminated or truncated:
                break
        run_completed = True
        print(f"Completed {len(rows)} Sinergym steps; output: {out}")
    finally:
        # Preserve operational evidence even when a later model request fails.
        try:
            if rows:
                with steps_path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            write_operational_artifacts(out, api_events, decision_rows,
                                        len(rows), selected_model, run_completed,
                                        "run" if args.run else "smoke-env",
                                        args.provider if args.run else "none",
                                        [row["step_runtime_sec"] for row in rows])
        finally:
            env.close()
    # Reuse shared occupied-PMV/HVAC-energy calculations with custom new-PMV inputs.
    observations_csv = workspace / "episode-1" / "monitor" / "observations.csv"
    if observations_csv.is_file():
        sys.path.insert(0, str(ROOT.parent))
        import pandas as pd

        metrics = compute_pmv_energy_only_metrics(
            pd.read_csv(observations_csv),
            [runperiod[0], runperiod[1], runperiod[2]],
            [runperiod[3], runperiod[4], runperiod[5]],

        )
        with (out / "metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(finite_json(metrics), stream, indent=2, allow_nan=False)
    # Record the exact configuration and EnergyPlus monitor location for comparison.
    with (out / "run_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump({"method": "DARLIN Text-only", "environment_id": ENV_ID,
                   "runperiod": list(config["runperiod"]), "timesteps_per_hour": 4,
                   "control_interval_minutes": CONTROL_INTERVAL_MINUTES,
                   "history_samples": HISTORY_SAMPLES,
                   "history_interval_minutes": CONTROL_INTERVAL_MINUTES,
                   "history_window_minutes": HISTORY_WINDOW_MINUTES,
                   "plan_horizon_steps": PLAN_HORIZON,
                   "plan_horizon_minutes": PLAN_HORIZON_MINUTES,
                   "zones": list(ZONES),
                   "plugin_one_decision_delay": True,
                   "action_mode": "independent continuous heating and cooling",
                   "heating_bounds_c": list(HSP_BOUNDS),
                   "cooling_bounds_c": list(CSP_BOUNDS),
                   "initial_pairs_c": initial_pairs,
                   "warmup_policy": {
                       "summer_occupied_months": [6, 7, 8, 9],
                       "summer_occupied_pair_c": list(SUMMER_OCCUPIED_PAIR),
                       "other_occupied_pair_c": list(WINTER_OCCUPIED_PAIR),
                       "unoccupied_pair_c": list(UNOCCUPIED_PAIR),
                       "recompute_every_simulation_step": True,
                   },
                   "fallback_policy": {
                       "method": "proposed-controller seasonal occupancy and PMV rule",
                       "summer_occupied_months": [6, 7, 8, 9],
                       "summer_occupied_pair_c": list(SUMMER_OCCUPIED_PAIR),
                       "other_occupied_pair_c": list(WINTER_OCCUPIED_PAIR),
                       "unoccupied_pair_c": list(UNOCCUPIED_PAIR),
                       "pmv_band": [PMV_LOWER, PMV_UPPER],
                       "api_or_invalid_output_fallback": True,
                       "control_retries": 0,
                       "apply_immediately": True,
                       "clear_pending_action_for_zone": True,
                   },
                   "evaluation_objectives": ["total_hvac_energy_kwh",
                                             "occupied_pmv_deviation"],
                   "pmv_observation_names": [f"new_pmv_space{i}" for i in range(1, 6)],
                   "pmv_definition": NEW_PMV_DEFINITION,
                   "author_text_index": str(TEXT_DIR),
                   "api_provider": args.provider if args.run else None,
                   "model": selected_model if args.run else None,
                   "reasoning_effort": REASONING_EFFORT if args.run else None,
                   "model_temperature": MODEL_TEMPERATURE if args.run else None,
                   "model_max_tokens": MODEL_MAX_TOKENS if args.run else None,
                   "api_base_url": selected_api_base if args.run else None,
                   "llm_calls_jsonl": (str(out / "llm_calls.jsonl")
                                       if (out / "llm_calls.jsonl").is_file() else None),
                   "operational_calls_csv": str(out / "operational_calls.csv"),
                   "operational_decisions_csv": str(out / "operational_decisions.csv"),
                   "operational_summary_csv": str(out / "operational_summary.csv"),
                   "energyplus_workspace": str(workspace),
                   "observations_csv": str(observations_csv),
                   "smoke_only": bool(args.smoke_env)}, stream, indent=2)


if __name__ == "__main__":
    main()
