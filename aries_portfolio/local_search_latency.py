from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .baselines import insertion_two_opt, iterated_local_search, multi_start_two_opt
from .io import load_frozen_instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / "configs" / "experiment.json").read_text())
    protocol = config["classical_controls"]
    primary_starts = int(protocol["multi_start_2opt"]["starts_per_task_seed"])
    primary_iterations = int(protocol["ils"]["iterations_per_task_seed"])
    runtime_starts = int(protocol["runtime_matched"]["multi_start_2opt_starts"])
    runtime_iterations = int(protocol["runtime_matched"]["ils_iterations"])
    instances = load_frozen_instances(root / "results" / "frozen_instances_gps.json")
    rows = []
    for instance in instances:
        results = [
            insertion_two_opt(instance),
            multi_start_two_opt(instance, args.seed, primary_starts, "MultiStart2opt"),
            iterated_local_search(instance, args.seed, primary_iterations, "ILS"),
            multi_start_two_opt(instance, args.seed, runtime_starts, "MultiStart2opt-RT"),
            iterated_local_search(instance, args.seed, runtime_iterations, "ILS-RT"),
        ]
        for result in results:
            rows.append(
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
        print(f"serial local-search timing task={instance.task_id:02d}/30", flush=True)
    path = root / "results" / "latency_classical_controls_serial_seed42.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

