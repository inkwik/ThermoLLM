"""Shared PMV, comfort, and HVAC energy metrics for simulation result data.

Calculate ASHRAE 55 PMV with ``pythermalcomfort`` using zone air temperature for both
``tdb`` and ``tr``, observed relative humidity, ``vr=0.1 m/s``, ``met=1.2``,
``wme=0``, and ``clo=0.5`` in June–September (summer months) or ``1.0`` otherwise. PMV comfort
is assessed against [-0.5, 0.5]. HVAC energy integrates 15-minute power values.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pythermalcomfort.models import pmv_ppd_ashrae

ZONES = [1, 2, 3, 4, 5]
HVAC_POWER_COL = "HVAC_electricity_demand_rate"
STEP_HOURS = 15.0 / 60.0

SUMMER_MONTHS = {6, 7, 8, 9}

WINTER_CLO = 1.0
SUMMER_CLO = 0.5
MET_RATE = 1.2
PMV_LOWER = -0.5
PMV_UPPER = 0.5

WINTER_COMFORT_LOW = 20.0
WINTER_COMFORT_HIGH = 23.5
SUMMER_COMFORT_LOW = 23.0
SUMMER_COMFORT_HIGH = 26.0


def runperiod_to_date_range(runperiod: Tuple[int, ...]) -> Tuple[List[int], List[int]]:
    """Convert a run period (d,m,y,d,m,y) to start and end dates."""
    sd, sm, sy, ed, em, ey = runperiod
    return [sd, sm, sy], [ed, em, ey]


def _pmv_from_result(result: Any) -> float:
    if isinstance(result, dict):
        return float(result.get("pmv"))
    return float(getattr(result, "pmv"))


def _clo_for_month(month: int) -> float:
    return SUMMER_CLO if month in SUMMER_MONTHS else WINTER_CLO


def clo_for_month(month: int) -> float:
    """Public accessor for the seasonal clothing insulation used in PMV."""
    return _clo_for_month(int(month))


def pmv_for_conditions(
    tdb: float,
    rh: float,
    month: int,
    *,
    tr: Optional[float] = None,
    vr: float = 0.1,
    wme: float = 0.0,
) -> Optional[float]:
    """
    Single-point ASHRAE PMV using the same assumptions as the batch metrics
    (met=MET_RATE, seasonal clo, mean radiant temp = air temp, vr=0.1 m/s).

    Returns None if inputs are missing/invalid.
    """
    try:
        tdb_f = float(tdb)
        rh_f = float(rh)
        month_i = int(month)
    except (TypeError, ValueError):
        return None
    if np.isnan(tdb_f) or np.isnan(rh_f):
        return None
    clo = _clo_for_month(month_i)
    res = pmv_ppd_ashrae(
        tdb=tdb_f,
        tr=tdb_f if tr is None else float(tr),
        vr=vr,
        rh=rh_f,
        met=MET_RATE,
        clo=float(clo),
        wme=wme,
    )
    pmv = _pmv_from_result(res)
    if np.isnan(pmv):
        return None
    return pmv


def is_pmv_violation(pmv: Optional[float]) -> Optional[bool]:
    """True if PMV is outside the comfort band [PMV_LOWER, PMV_UPPER]; None if pmv is None."""
    if pmv is None:
        return None
    return bool(pmv < PMV_LOWER or pmv > PMV_UPPER)


def _comfort_band_for_month(month: int) -> Tuple[float, float]:
    if month in SUMMER_MONTHS:
        return SUMMER_COMFORT_LOW, SUMMER_COMFORT_HIGH
    return WINTER_COMFORT_LOW, WINTER_COMFORT_HIGH


def _day_column(df: pd.DataFrame) -> str:
    if "day" in df.columns and "day_of_month" not in df.columns:
        return "day"
    if "day_of_month" in df.columns:
        return "day_of_month"
    return "day"


def _mask_date_range(df: pd.DataFrame, start: List[int], end: List[int]) -> pd.Series:
    s_day, s_month, s_year = start
    e_day, e_month, e_year = end
    day_col = _day_column(df)
    dates = pd.to_datetime(
        {"year": s_year, "month": df["month"], "day": df[day_col]},
        errors="coerce",
    )
    start_ts = pd.Timestamp(year=s_year, month=s_month, day=s_day)
    end_ts = pd.Timestamp(year=e_year, month=e_month, day=e_day)
    return (dates >= start_ts) & (dates <= end_ts)


def df_in_date_range(df: pd.DataFrame, start: List[int], end: List[int]) -> pd.DataFrame:
    day_col = _day_column(df)
    if day_col not in df.columns or "month" not in df.columns:
        return df.iloc[0:0]
    return df.loc[_mask_date_range(df, start, end)].copy()


def total_hvac_energy_kwh(df_range: pd.DataFrame) -> float:
    if HVAC_POWER_COL not in df_range.columns:
        return 0.0
    power_w = pd.to_numeric(df_range[HVAC_POWER_COL], errors="coerce").fillna(0.0)
    return float((power_w * STEP_HOURS / 1000.0).sum())


def zone_pmv_for_occupied_steps(
    df_range: pd.DataFrame, zone: int
) -> Tuple[np.ndarray, int]:
    """
    PMV values for occupied rows in one zone.

    Calculate PMV using the local ASHRAE method.
    """
    pmv_col = f"pmv_space{zone}"
    occ_col = f"air_occ_space{zone}"
    if pmv_col in df_range.columns and occ_col in df_range.columns:
        zdf = df_range[df_range[occ_col] > 0].copy()
        if zdf.empty:
            return np.array([], dtype=float), 0
        pmv = pd.to_numeric(zdf[pmv_col], errors="coerce").dropna().to_numpy(dtype=float)
        return pmv, int(pmv.size)

    t_col = f"air_temperature_space{zone}"
    rh_col = f"air_humidity_space{zone}"
    for c in (t_col, rh_col, occ_col, "month"):
        if c not in df_range.columns:
            return np.array([], dtype=float), 0
    zdf = df_range[df_range[occ_col] > 0].copy()
    if zdf.empty:
        return np.array([], dtype=float), 0
    temps = pd.to_numeric(zdf[t_col], errors="coerce")
    hums = pd.to_numeric(zdf[rh_col], errors="coerce")
    months = pd.to_numeric(zdf["month"], errors="coerce").astype("Int64")
    out: List[float] = []
    for t_val, rh, month in zip(temps, hums, months):
        if pd.isna(t_val) or pd.isna(rh) or pd.isna(month):
            continue
        pmv = pmv_for_conditions(float(t_val), float(rh), int(month))
        if pmv is None:
            continue
        out.append(pmv)
    arr = np.array(out, dtype=float)
    return arr, int(arr.size)


def pmv_stats(pmv: np.ndarray) -> Tuple[float, float, float]:
    if pmv.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(pmv))
    std = float(np.std(pmv))
    viol = 100.0 * float(((pmv < PMV_LOWER) | (pmv > PMV_UPPER)).sum()) / pmv.size
    return round(mean, 3), round(std, 3), round(viol, 2)


def zone_comfort_violations_for_occupied_steps(
    df_range: pd.DataFrame, zone: int
) -> Tuple[np.ndarray, int]:
    t_col = f"air_temperature_space{zone}"
    occ_col = f"air_occ_space{zone}"
    for c in (t_col, occ_col, "month"):
        if c not in df_range.columns:
            return np.array([], dtype=bool), 0
    zdf = df_range[df_range[occ_col] > 0].copy()
    if zdf.empty:
        return np.array([], dtype=bool), 0
    temps = pd.to_numeric(zdf[t_col], errors="coerce")
    months = pd.to_numeric(zdf["month"], errors="coerce").astype("Int64")
    violations: List[bool] = []
    for temp, month in zip(temps, months):
        if pd.isna(temp) or pd.isna(month):
            continue
        low, high = _comfort_band_for_month(int(month))
        violations.append(bool(temp < low or temp > high))
    arr = np.array(violations, dtype=bool)
    return arr, int(arr.size)


def comfort_violation_pct(violations: np.ndarray) -> float:
    if violations.size == 0:
        return float("nan")
    return round(100.0 * float(violations.sum()) / violations.size, 2)


def compute_performance_metrics(
    df: pd.DataFrame,
    date_start: List[int],
    date_end: List[int],
) -> Dict[str, Any]:
    """
    Compute run-level PMV, comfort, and energy metrics for a date range.

    Returns a flat dict suitable for MLflow logging plus per-zone breakdown.
    """
    sub = df_in_date_range(df, date_start, date_end)
    metrics: Dict[str, Any] = {
        "date_start": date_start,
        "date_end": date_end,
        "period_label": label_for_date_range(date_start, date_end),
        "total_hvac_energy_kwh": total_hvac_energy_kwh(sub),
        "pmv_mean_all": float("nan"),
        "pmv_std_all": float("nan"),
        "pmv_violation_pct_all": float("nan"),
        "comfort_violation_pct_all": float("nan"),
        "occupied_pmv_steps_all": 0,
    }
    zone_metrics: Dict[str, Dict[str, Any]] = {}

    all_pmvs: List[np.ndarray] = []
    all_comfort_viol: List[np.ndarray] = []
    for z in ZONES:
        pmv_arr, n_pmv = zone_pmv_for_occupied_steps(sub, z)
        comfort_viol, _ = zone_comfort_violations_for_occupied_steps(sub, z)
        mean, std, pmv_viol = pmv_stats(pmv_arr)
        comfort_viol_pct_val = comfort_violation_pct(comfort_viol)
        zone_metrics[f"Z{z}"] = {
            "occupied_pmv_steps": n_pmv,
            "pmv_mean": mean,
            "pmv_std": std,
            "pmv_violation_pct": pmv_viol,
            "comfort_violation_pct": comfort_viol_pct_val,
        }
        if pmv_arr.size:
            all_pmvs.append(pmv_arr)
        if comfort_viol.size:
            all_comfort_viol.append(comfort_viol)

    if all_pmvs:
        stacked = np.concatenate(all_pmvs)
        m, s, pmv_viol_all = pmv_stats(stacked)
        metrics["pmv_mean_all"] = m
        metrics["pmv_std_all"] = s
        metrics["pmv_violation_pct_all"] = pmv_viol_all
        metrics["occupied_pmv_steps_all"] = int(stacked.size)
    if all_comfort_viol:
        metrics["comfort_violation_pct_all"] = comfort_violation_pct(
            np.concatenate(all_comfort_viol)
        )

    metrics["zone_metrics"] = zone_metrics
    metrics["rows_in_range"] = int(len(sub))
    metrics["summary_table"] = build_pmv_summary_table(metrics)
    return metrics


def label_for_date_range(start: List[int], end: List[int]) -> str:
    sd, sm, sy = start
    ed, em, ey = end
    if sm == em and sy == ey:
        return f"{sd:02d}-{ed:02d}/{sm:02d}/{sy}"
    return f"{sd:02d}/{sm:02d}/{sy}–{ed:02d}/{em:02d}/{ey}"


def build_pmv_summary_table(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return tabular PMV summary rows for each zone and the whole building."""
    rows: List[Dict[str, Any]] = []
    zone_metrics = metrics.get("zone_metrics") or {}

    for z in ZONES:
        zm = zone_metrics.get(f"Z{z}", {})
        rows.append(
            {
                "zone": f"Zone {z}",
                "n_steps": zm.get("occupied_pmv_steps", 0),
                "mean_PMV": zm.get("pmv_mean", float("nan")),
                "std_PMV": zm.get("pmv_std", float("nan")),
                "pmv_viol_pct": zm.get("pmv_violation_pct", float("nan")),
                "comfort_viol_pct": zm.get("comfort_violation_pct", float("nan")),
            }
        )

    rows.append(
        {
            "zone": "ALL",
            "n_steps": metrics.get("occupied_pmv_steps_all", 0),
            "mean_PMV": metrics.get("pmv_mean_all", float("nan")),
            "std_PMV": metrics.get("pmv_std_all", float("nan")),
            "pmv_viol_pct": metrics.get("pmv_violation_pct_all", float("nan")),
            "comfort_viol_pct": metrics.get("comfort_violation_pct_all", float("nan")),
        }
    )
    return rows


def pmv_summary_table_to_dataframe(metrics: Dict[str, Any]) -> pd.DataFrame:
    rows = metrics.get("summary_table") or build_pmv_summary_table(metrics)
    return pd.DataFrame(rows)


def format_pmv_summary_report(
    metrics: Dict[str, Any],
    *,
    label: str = "Run",
    date_start: Optional[List[int]] = None,
    date_end: Optional[List[int]] = None,
) -> str:
    """Format run-level PMV and HVAC energy metrics as readable text."""
    period = metrics.get("period_label")
    if not period and date_start and date_end:
        period = label_for_date_range(date_start, date_end)
    period = period or "unknown"

    e_kwh = metrics.get("total_hvac_energy_kwh", float("nan"))
    lines = [
        "=" * 60,
        f"{label}  |  Period: {period}",
        "=" * 60,
        f"Total HVAC electrical energy (all steps in range): {e_kwh:.4f} kWh",
        f"(from column {HVAC_POWER_COL!r}, W × {STEP_HOURS} h / 1000 per row)",
        "",
        "PMV (occupied steps only, per zone: air_occ_space* > 0):",
        f"  pmv_viol_%     : |PMV| outside [{PMV_LOWER}, {PMV_UPPER}]",
        (
            f"  comfort_viol_%: air temp outside winter [{WINTER_COMFORT_LOW}, "
            f"{WINTER_COMFORT_HIGH}] °C or summer [{SUMMER_COMFORT_LOW}, "
            f"{SUMMER_COMFORT_HIGH}] °C"
        ),
        f"{'Zone':<8} {'n_steps':>8} {'mean_PMV':>10} {'std_PMV':>10} "
        f"{'pmv_viol_%':>10} {'comfort_viol_%':>14}",
    ]

    for row in metrics.get("summary_table") or build_pmv_summary_table(metrics):
        zone = row["zone"]
        n_steps = int(row.get("n_steps") or 0)
        mean = row.get("mean_PMV", float("nan"))
        std = row.get("std_PMV", float("nan"))
        pmv_viol = row.get("pmv_viol_pct", float("nan"))
        comfort_viol = row.get("comfort_viol_pct", float("nan"))

        mean_s = f"{mean:.3f}" if not (isinstance(mean, float) and np.isnan(mean)) else "nan"
        std_s = f"{std:.3f}" if not (isinstance(std, float) and np.isnan(std)) else "nan"
        pmv_viol_s = (
            f"{pmv_viol:.2f}" if not (isinstance(pmv_viol, float) and np.isnan(pmv_viol)) else "nan"
        )
        comfort_viol_s = (
            f"{comfort_viol:.2f}"
            if not (isinstance(comfort_viol, float) and np.isnan(comfort_viol))
            else "nan"
        )
        zone_label = str(zone)
        lines.append(
            f"{zone_label:<8} {n_steps:8d}  {mean_s:>10}  {std_s:>10}  "
            f"{pmv_viol_s:>10}  {comfort_viol_s:>14}"
        )

    return "\n".join(lines)


def save_pmv_summary_artifacts(
    metrics: Dict[str, Any],
    *,
    label: str,
    csv_path: str,
    txt_path: str,
    date_start: Optional[List[int]] = None,
    date_end: Optional[List[int]] = None,
) -> Tuple[str, str]:
    """Save PMV summary as CSV table + terminal-style text report."""
    import os

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pmv_summary_table_to_dataframe(metrics)
    df.to_csv(csv_path, index=False)

    report_text = format_pmv_summary_report(
        metrics, label=label, date_start=date_start, date_end=date_end
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    return csv_path, txt_path


def format_metrics_report(metrics: Dict[str, Any], label: str = "Run") -> str:
    return format_pmv_summary_report(metrics, label=label)
