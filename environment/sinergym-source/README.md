# Paper Sinergym source

This folder contains the Sinergym Python package source used for the paper, with its environment configurations, building models, Brick data, and weather files. The `sinergym/` directory is the package source.

The model uses the occupancy CSVs in `occupancy_schedules_15min/`. Update the five `Schedule:File` paths in `sinergym/data/buildings/5ZoneAutoDXVAVnw.epJSON` to the directory where you place those files before running.

Git metadata, Python caches, and unrelated `.agents` and plugin directories are excluded. This is the package source snapshot, not the complete upstream build repository; its runtime dependencies and EnergyPlus setup still need to be installed as described in the paper setup.
