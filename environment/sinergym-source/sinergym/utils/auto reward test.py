import os
import numpy as np
import pandas as pd

# ==============================================================================
# 1. DATA SCHEMA CONFIGURATION (Adjust names to match your CSV headers)
# ==============================================================================
CSV_NAME = "rbc_observations.csv"
GLOBAL_ENERGY_COL = "total_electricity_HVAC"  # Expected in Joules

# Map your 5 zones to their respective temperature and binary occupancy columns
ZONE_COLUMNS = {
    1: {"temp": "Zone_1_Air_Temperature", "occ": "Zone_1_Occupancy"},
    2: {"temp": "Zone_2_Air_Temperature", "occ": "Zone_2_Occupancy"},
    3: {"temp": "Zone_3_Air_Temperature", "occ": "Zone_3_Occupancy"},
    4: {"temp": "Zone_4_Air_Temperature", "occ": "Zone_4_Occupancy"},
    5: {"temp": "Zone_5_Air_Temperature", "occ": "Zone_5_Occupancy"},
}

# Strict Winter Boundaries
COMFORT_LOW = 20.0
COMFORT_HIGH = 23.5
STEPS_PER_DAY = 96  # 15-minute timesteps (4 * 24)

# ==============================================================================
# 2. INITIALIZATION
# ==============================================================================
os.makedirs("success_trajectories", exist_ok=True)
os.makedirs("failure_trajectories", exist_ok=True)

print(f"Loading {CSV_NAME}...")
df = pd.read_csv(CSV_NAME)

# Create a clean grouping index for days based on sequential steps
df["day_index"] = df.index // STEPS_PER_DAY

daily_metrics = []
grouped_days = df.groupby("day_index")

# ==============================================================================
# 3. FIRST PASS: CALCULATE METRICS PER DAY
# ==============================================================================
for day_idx, daily_df in grouped_days:
    # A. Global Daily Energy Calculation (Joules to kWh)
    total_day_energy_kwh = daily_df[GLOBAL_ENERGY_COL].sum() / 3600000.0

    # B. Spatial-Temporal Zone-Step Pool Comfort Calculation
    total_occupied_zone_steps = 0
    total_comfort_deviation = 0.0

    for _, row in daily_df.iterrows():
        for zone_id, cols in ZONE_COLUMNS.items():
            # Check occupancy flag for this specific zone-step
            if row[cols["occ"]] == 1:
                total_occupied_zone_steps += 1
                zone_temp = row[cols["temp"]]

                # Accumulate out-of-bounds deviation
                if zone_temp < COMFORT_LOW:
                    total_comfort_deviation += COMFORT_LOW - zone_temp
                elif zone_temp > COMFORT_HIGH:
                    total_comfort_deviation += zone_temp - COMFORT_HIGH

    # Compute Normalized Daily Comfort Score
    if total_occupied_zone_steps > 0:
        daily_comfort_score = (
            total_comfort_deviation / total_occupied_zone_steps
        )
    else:
        daily_comfort_score = 0.0  # Safe default if entirely vacant (weekends)

    daily_metrics.append(
        {
            "day_index": day_idx,
            "comfort_score": daily_comfort_score,
            "total_energy": total_day_energy_kwh,
            "raw_dataframe": daily_df,
        }
    )

summary_df = pd.DataFrame(daily_metrics)

# ==============================================================================
# 4. STATISTICAL QUANTILE DERIVATION (Season-wide Benchmarks)
# ==============================================================================
# Success Target: Median energy of winter days that achieved flawless tracking
perfect_comfort_days = summary_df[summary_df["comfort_score"] == 0.0]
if not perfect_comfort_days.empty:
    Q50_SUCCESS_ENERGY = np.percentile(perfect_comfort_days["total_energy"], 50)
else:
    Q50_SUCCESS_ENERGY = np.percentile(summary_df["total_energy"], 50)

# Failure Targets: Statistical worst-performance boundaries
Q90_FAILURE_COMFORT = np.percentile(summary_df["comfort_score"], 90)
Q85_FAILURE_ENERGY = np.percentile(summary_df["total_energy"], 85)

print("\n--- Dynamically Extracted Operational Limits ---")
print(f"Success Energy Benchmark (Q50):  {Q50_SUCCESS_ENERGY:.2f} kWh")
print(f"Failure Discomfort Boundary (Q90): {Q90_FAILURE_COMFORT:.4f}°C/step")
print(f"Failure Energy Boundary (Q85):     {Q85_FAILURE_ENERGY:.2f} kWh\n")

# ==============================================================================
# 5. SECOND PASS: FILTERING ENGINE & EXPORT (Trinary Sorting)
# ==============================================================================
success_count = 0
failure_count = 0

for day in daily_metrics:
    day_idx = day["day_index"]
    comfort = day["comfort_score"]
    energy = day["total_energy"]
    raw_data = day["raw_dataframe"].drop(columns=["day_index"])

    file_name = f"winter_day_{day_idx + 1}.csv"

    # Perfect tracking performance underneath historical energy spend median
    if comfort == 0.0 and energy <= Q50_SUCCESS_ENERGY:
        raw_data.to_csv(f"success_trajectories/{file_name}", index=False)
        success_count += 1

    # Outlier performance representing true operational failures
    elif comfort > Q90_FAILURE_COMFORT or energy > Q85_FAILURE_ENERGY:
        raw_data.to_csv(f"failure_trajectories/{file_name}", index=False)
        failure_count += 1

print("--- Trinary Library Generation Complete ---")
print(f"Saved {success_count} Optimal Trajectories to /success_trajectories")
print(f"Saved {failure_count} Failure Trajectories to /failure_trajectories")
print(
    f"Discarded {len(daily_metrics) - (success_count + failure_count)} Mediocre Buffer Days."
)