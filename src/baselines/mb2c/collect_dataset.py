#!/usr/bin/env python3
"""Collect the 12/3/3 seeded-TMY MB2C dataset for three climates."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from mb2c_5zone_multiclimate import collection_specs, collect_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = args.output_root.resolve()
    for folder in (root / "development", root / "test", root / "sinergym_collection"):
        folder.mkdir(parents=True, exist_ok=True)

    specs, manifest = collection_specs(root)
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with mp.get_context("spawn").Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(collect_one, specs):
            print(f"Collected: {result}", flush=True)
    print(f"Collected {sum(len(v) for v in manifest['splits'].values())} trajectories at {root}", flush=True)


if __name__ == "__main__":
    main()
