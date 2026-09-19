from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .baselines import cheapest_insertion, nearest_neighbor, two_opt
from .io import load_frozen_instances


def build_runtime_reports(root: Path) -> None:
    instances = load_frozen_instances(root / "results" / "frozen_instances_gps.json")
    quick_rows = []
    for instance in instances:
        for result in [nearest_neighbor(instance), cheapest_insertion(instance), two_opt(instance)]:
            quick_rows.append(
                {
                    "task_id": instance.task_id,
                    "num_targets": len(instance.targets_gps),
                    "seed": result.seed,
                    "method": result.method,
                    "runtime_s": result.runtime_s,
                    "length_km": result.length_km,
                    "evaluations": result.evaluations,
                }
            )
    quick = pd.DataFrame(quick_rows)
    quick.to_csv(root / "results" / "latency_deterministic_serial.csv", index=False)

    original = pd.read_csv(root / "results" / "latency_serial_seed42.csv")
    original = original[original.method.isin(["Raw", "ACO", "DE", "PSO", "ACO×3", "Portfolio"])]
    local = pd.read_csv(root / "results" / "latency_classical_controls_serial_seed42.csv")
    combined = pd.concat([quick, original, local], ignore_index=True)
    summary = combined.groupby("method", as_index=False).agg(
        n_tasks=("task_id", "nunique"),
        mean_runtime_s=("runtime_s", "mean"),
        median_runtime_s=("runtime_s", "median"),
        minimum_runtime_s=("runtime_s", "min"),
        maximum_runtime_s=("runtime_s", "max"),
        mean_evaluations=("evaluations", "mean"),
    )
    summary.to_csv(root / "results" / "runtime_summary.csv", index=False)

    classical = pd.read_csv(root / "results" / "routes_classical_controls.csv")
    rows = []
    for method, frame in classical.groupby("method"):
        metadata = frame.metadata.map(json.loads)
        row = {
            "method": method,
            "n_rows": len(frame),
            "mean_initial_solutions": metadata.map(lambda item: item.get("initial_solutions", 0)).mean(),
            "mean_candidate_moves_evaluated": metadata.map(lambda item: item.get("candidate_moves_evaluated", 0)).mean(),
            "mean_accepted_2opt_moves": metadata.map(lambda item: item.get("accepted_moves", 0)).mean(),
            "mean_completed_local_search_passes": metadata.map(lambda item: item.get("completed_local_search_passes", 0)).mean(),
            "mean_perturbations": metadata.map(lambda item: item.get("perturbations", 0)).mean(),
            "mean_full_route_objective_evaluations": metadata.map(
                lambda item: item.get("full_route_objective_evaluations", 0)
            ).mean(),
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / "results" / "search_accounting_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    build_runtime_reports(args.root.resolve())


if __name__ == "__main__":
    main()

