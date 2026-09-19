from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .baselines import two_opt
from .io import load_detection_instances
from .solvers import solve_aco, solve_de, solve_pso, solve_raw, select_multistart, select_portfolio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=7500)
    args = parser.parse_args()
    root = args.root
    instances = load_detection_instances(
        root / "results" / "detections_buildings.json",
        root / "data" / "nano30" / "img_lat_long_data.txt",
        root / "data" / "nano30" / "images",
    )
    rows = []
    for instance in instances:
        raw = solve_raw(instance, seed=args.seed)
        opt2 = two_opt(instance)
        opt2.seed = args.seed
        aco_runs = [
            solve_aco(instance, seed=args.seed + offset, eval_budget=args.budget, pop_size=50)
            for offset in [0, 20, 40]
        ]
        aco = aco_runs[0]
        de = solve_de(instance, seed=args.seed, eval_budget=args.budget, pop_size=50)
        pso = solve_pso(instance, seed=args.seed, eval_budget=args.budget, pop_size=50)
        portfolio = select_portfolio(
            instance,
            [raw, opt2, aco, de, pso],
            args.seed,
            priority=["Raw", "NN+2opt", "ACO", "DE", "PSO"],
            method="Portfolio",
        )
        aco_multistart = select_multistart(instance, aco_runs, args.seed)
        for result in [raw, opt2, aco, de, pso, portfolio, aco_multistart]:
            rows.append(
                {
                    "task_id": instance.task_id,
                    "num_targets": len(instance.targets_gps),
                    "method": result.method,
                    "runtime_s": result.runtime_s,
                    "length_km": result.length_km,
                    "evaluations": result.evaluations,
                    "selected_method": result.selected_method,
                }
            )
        print(f"latency task={instance.task_id:02d}", flush=True)
    with (root / "results" / "latency_serial_seed42.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
