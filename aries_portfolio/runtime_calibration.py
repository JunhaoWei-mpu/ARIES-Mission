from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import pandas as pd

from .baselines import iterated_local_search, multi_start_two_opt
from .types import MissionInstance


SYNTHETIC_SIZES = [11, 11, 16, 12, 12, 11, 8, 19, 13, 11, 11, 15, 11, 5, 9,
                   17, 7, 17, 6, 10, 7, 42, 4, 4, 6, 10, 5, 10, 20, 12]


def _synthetic_instances() -> list[MissionInstance]:
    instances = []
    for task_id, n_targets in enumerate(SYNTHETIC_SIZES, start=1):
        rng = np.random.default_rng(np.random.SeedSequence([20260821, task_id, n_targets]))
        offsets = rng.uniform(-0.02, 0.02, size=(n_targets, 2))
        targets = tuple((43.0 + float(lat), -80.0 + float(lon)) for lat, lon in offsets)
        instances.append(
            MissionInstance(
                task_id=task_id,
                home_gps=(43.0, -80.0),
                targets_gps=targets,
                detection_order=tuple(range(n_targets)),
                metadata={"development_only": True},
            )
        )
    return instances


def calibrate(root: Path) -> dict[str, object]:
    config = json.loads((root / "configs" / "experiment.json").read_text())
    protocol = config["classical_controls"]
    primary_starts = int(protocol["multi_start_2opt"]["starts_per_task_seed"])
    primary_iterations = int(protocol["ils"]["iterations_per_task_seed"])
    serial = pd.read_csv(root / "results" / "latency_serial_seed42.csv")
    target_runtime = float(serial.loc[serial.method == "Portfolio", "runtime_s"].mean())

    instances = _synthetic_instances()
    ms_times = [multi_start_two_opt(item, 0, starts=primary_starts).runtime_s for item in instances]
    ils_times = [iterated_local_search(item, 0, iterations=primary_iterations).runtime_s for item in instances]
    mean_ms = statistics.fmean(ms_times)
    mean_ils = statistics.fmean(ils_times)
    recommended_starts = max(
        primary_starts + 1,
        int(round(primary_starts * target_runtime / max(mean_ms, 1e-9))),
    )
    recommended_iterations = max(
        primary_iterations + 1,
        int(round(primary_iterations * target_runtime / max(mean_ils, 1e-9))),
    )
    report = {
        "status": "development-only runtime calibration; frozen task coordinates and objectives were not read",
        "synthetic_sizes": SYNTHETIC_SIZES,
        "seed": 0,
        "portfolio_reference_mean_runtime_s": target_runtime,
        "primary_multi_start_starts": primary_starts,
        "primary_ils_iterations": primary_iterations,
        "synthetic_primary_multi_start_mean_runtime_s": mean_ms,
        "synthetic_primary_ils_mean_runtime_s": mean_ils,
        "recommended_runtime_matched_multi_start_starts": recommended_starts,
        "recommended_runtime_matched_ils_iterations": recommended_iterations,
        "rule": "linear operation-budget scaling from a synthetic mission-size-matched calibration",
    }
    (root / "results" / "runtime_match_calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(calibrate(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()

