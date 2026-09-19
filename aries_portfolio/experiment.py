from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy

from .baselines import cheapest_insertion, held_karp, nearest_neighbor, two_opt
from .external_solvers import concorde_exact, resolve_binary
from .geo import closed_loop_length_km, crossing_count, gps_to_local_xy, node_distance_matrix
from .io import (
    load_detection_instances,
    load_frozen_instances,
    load_human_route,
    write_frozen_instances,
    write_results_csv,
    write_results_json,
)
from .solvers import solve_aco, solve_de, solve_pso, solve_raw, select_multistart, select_portfolio
from .types import RouteResult


def _run_stochastic_job(payload: tuple) -> list[RouteResult]:
    instance, seed, config = payload
    raw = solve_raw(instance, seed=seed)
    aco = solve_aco(
        instance,
        seed=seed,
        eval_budget=int(config["solver_evaluation_budget"]),
        pop_size=int(config["solver_population"]),
    )
    de = solve_de(
        instance,
        seed=seed,
        eval_budget=int(config["solver_evaluation_budget"]),
        pop_size=int(config["solver_population"]),
    )
    pso = solve_pso(
        instance,
        seed=seed,
        eval_budget=int(config["solver_evaluation_budget"]),
        pop_size=int(config["solver_population"]),
    )
    opt2 = two_opt(instance)
    opt2.seed = seed
    candidates = [raw, opt2, aco, de, pso]
    portfolio = select_portfolio(
        instance,
        candidates,
        seed=seed,
        tie_tolerance_km=float(config["tie_tolerance_km"]),
        priority=config["portfolio_priority"],
        method="Portfolio",
    )
    offsets = [int(value) for value in config["aco_multistart_seed_offsets"]]
    aco_runs = [aco]
    for offset in offsets[1:]:
        aco_runs.append(
            solve_aco(
                instance,
                seed=seed + offset,
                eval_budget=int(config["solver_evaluation_budget"]),
                pop_size=int(config["solver_population"]),
            )
        )
    aco_multistart = select_multistart(
        instance,
        aco_runs,
        seed=seed,
        tie_tolerance_km=float(config["tie_tolerance_km"]),
    )
    return [*candidates, portfolio, aco_multistart]


def _run_baseline_job(payload: tuple) -> list[RouteResult]:
    instance, config, root = payload
    nn = nearest_neighbor(instance)
    insertion = cheapest_insertion(instance)
    opt2 = two_opt(instance, nn.order)
    results = [nn, insertion, opt2]
    exact_config = config["exact"]
    scale = int(exact_config["integer_scale_per_km"])
    concorde = resolve_binary(root, exact_config["binary"])
    reference = concorde_exact(
        instance,
        binary=concorde,
        output_dir=root / "results" / "exact_logs" / f"task{instance.task_id:02d}",
        scale_per_km=scale,
        seed=int(exact_config["seed"]),
    )
    results.append(reference)
    if len(instance.targets_gps) <= 18:
        held_karp_check = held_karp(instance, 18)
        held_karp_check.method = "HeldKarpCheck"
        results.append(held_karp_check)

    return results


def _hardware_metadata() -> dict[str, object]:
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    ram_bytes = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                ram_bytes = int(line.split()[1]) * 1024
                break
    return {
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "ram_bytes": ram_bytes,
        "operating_system": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
    }


def run_experiment(root: Path, config_path: Path) -> list[RouteResult]:
    config = json.loads(config_path.read_text())
    data_dir = root / "data" / "nano30"
    frozen_path = root / "results" / "frozen_instances_gps.json"
    if frozen_path.exists():
        instances = load_frozen_instances(frozen_path, data_dir / "images")
    else:
        instances = load_detection_instances(
            root / "results" / "detections_buildings.json",
            data_dir / "img_lat_long_data.txt",
            data_dir / "images",
        )
        write_frozen_instances(frozen_path, instances)
    if len(instances) != 30:
        raise RuntimeError(f"Expected 30 detection records, found {len(instances)}")
    output: list[RouteResult] = []
    task_features = []
    baselines = []
    human_rows = []
    for instance in instances:
        matrix = node_distance_matrix(instance.home_gps, instance.targets_gps)
        raw = solve_raw(instance)
        local_xy = gps_to_local_xy(instance.home_gps, instance.targets_gps)
        task_features.append(
            {
                "task_id": instance.task_id,
                "num_targets": len(instance.targets_gps),
                "raw_length_km": raw.length_km,
                "raw_crossings": crossing_count(raw.order, local_xy),
            }
        )
        human_path = data_dir / "human_waypoints" / f"{instance.task_id}.waypoints"
        if human_path.exists():
            human_home, human_targets = load_human_route(human_path)
            human_matrix = node_distance_matrix(human_home, human_targets)
            human_order = tuple(range(len(human_targets)))
            human_nodes = [0, *[index + 1 for index in human_order]]
            open_length = float(sum(human_matrix[a, b] for a, b in zip(human_nodes[:-1], human_nodes[1:])))
            closing_edge = float(human_matrix[human_nodes[-1], 0]) if human_nodes else 0.0
            human_rows.append(
                {
                    "task_id": instance.task_id,
                    "num_waypoints": len(human_targets),
                    "open_length_km": open_length,
                    "closing_edge_km": closing_edge,
                    "length_km": closed_loop_length_km(human_order, human_matrix),
                }
            )

    with ProcessPoolExecutor(max_workers=int(config["baseline_workers"])) as executor:
        future_map = {
            executor.submit(_run_baseline_job, (instance, config, root)): instance.task_id
            for instance in instances
        }
        for future in as_completed(future_map):
            baselines.extend(future.result())
            print(f"completed baselines task={future_map[future]:02d}", flush=True)

    jobs = [(instance, seed, config) for instance in instances for seed in config["solver_seeds"]]
    completed = 0
    with ProcessPoolExecutor(max_workers=int(config["stochastic_workers"])) as executor:
        futures = [executor.submit(_run_stochastic_job, job) for job in jobs]
        for future in as_completed(futures):
            output.extend(future.result())
            completed += 1
            if completed % 100 == 0:
                write_results_json(root / "results" / "routes_stochastic.partial.json", output)
            if completed % 20 == 0 or completed == len(jobs):
                print(f"completed stochastic jobs={completed}/{len(jobs)}", flush=True)

    results_dir = root / "results"
    output.sort(key=lambda result: (result.task_id, result.seed, result.method))
    baselines.sort(key=lambda result: (result.task_id, result.method, result.seed))
    write_results_json(results_dir / "routes_stochastic.json", output)
    write_results_csv(results_dir / "routes_stochastic.csv", output)
    write_results_json(results_dir / "routes_baselines.json", baselines)
    write_results_csv(results_dir / "routes_baselines.csv", baselines)
    tables = [("task_features.csv", task_features)]
    if human_rows:
        tables.append(("human_routes.csv", human_rows))
    for name, rows in tables:
        with (results_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    meta = {
        "hardware_software": _hardware_metadata(),
        "config": config,
        # Keep the manifest shareable: command options are useful provenance,
        # whereas the launcher's absolute path only identifies one workstation.
        "commands": {"argv": [Path(sys.argv[0]).name, *sys.argv[1:]]},
    }
    try:
        meta["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except subprocess.CalledProcessError:
        meta["git_commit"] = "uncommitted"
    (results_dir / "run_manifest.json").write_text(json.dumps(meta, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = args.config or args.root / "configs" / "experiment.json"
    run_experiment(args.root, config)


if __name__ == "__main__":
    main()
