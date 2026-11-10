# ThermoLLM

ThermoLLM is an LLM-based controller for five-zone HVAC control. This repository contains the controller, baseline implementations, shared evaluation utilities, and the custom Sinergym source and environment files used in the paper.

<img width="50%" alt="image" src="https://github.com/user-attachments/assets/014fc5bc-f7da-4a32-9e71-6117484a00c2" />

## Repository layout

- `src/thermollm/` — ThermoLLM controller and Brick context retrieval
- `src/baselines/` — DARLIN, PPO, RBC, and MB2C baselines
- `src/common/` — shared PMV and energy metrics
- `environment/sinergym-source/` — Sinergym package source, environment configuration, building models, weather files, and occupancy schedules

## Environment setup

Install EnergyPlus 25.1.0 and the Python dependencies required by Sinergym and the selected method. The 5-zone model reads five occupancy schedule CSV files. Update the `Schedule:File` paths in `environment/sinergym-source/sinergym/data/buildings/5ZoneAutoDXVAVnw.epJSON` to match their location on your system.

## API credentials

LLM methods require a provider API key. Set it as an environment variable before running the method. Do not commit API keys or other credentials.

## DARLIN retrieval assets

DARLIN also requires its FAISS index, corresponding metadata JSON file, and `DarlinGenerator` dependency. These components are not included in this repository due to their large size. Please contact the authors via email to obtain them.
