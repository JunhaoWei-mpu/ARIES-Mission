"""Reproduce and verify the reported five-candidate ARIES-Mission routes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
from typing import Iterable

from .components import solve_nn_two_opt
from .solvers import solve_aco, solve_de, solve_pso, solve_raw, select_portfolio
from .types import MissionInstance, RouteResult


def load_instances(path: Path) -> list[MissionInstance]:
    payload = json.loads(path.read_text())
    return [
        MissionInstance(
            task_id=int(item["task_id"]),
            home_gps=tuple(map(float, item["home_gps"])),
            targets_gps=tuple(tuple(map(float, point)) for point in item["targets_gps"]),
            target_type=str(item.get("target_type", "buildings")),
            detection_order=tuple(map(int, item["detection_order"])),
        )
        for item in payload
    ]


def _run_one(payload: tuple[MissionInstance, int, dict]) -> list[RouteResult]:
    instance, seed, config = payload
    budget = int(config["solver_evaluation_budget"])
    population = int(config["solver_population"])
    candidates = [
        solve_raw(instance, seed=seed),
        solve_nn_two_opt(instance, seed=seed),
        solve_aco(instance, seed=seed, eval_budget=budget, pop_size=population),
        solve_de(instance, seed=seed, eval_budget=budget, pop_size=population),
        solve_pso(instance, seed=seed, eval_budget=budget, pop_size=population),
    ]
    portfolio = select_portfolio(
        instance,
        candidates,
        seed=seed,
        tie_tolerance_km=float(config["tie_tolerance_km"]),
        priority=config["portfolio_priority"],
    )
    return [*candidates, portfolio]


def _serialise(results: Iterable[RouteResult]) -> list[dict]:
    rows = []
    for result in results:
        row = result.to_dict()
        row["order"] = json.dumps(row["order"], separators=(",", ":"))
        row["history_km"] = json.dumps(row["history_km"], separators=(",", ":"))
        row["metadata"] = json.dumps(row["metadata"], sort_keys=True, separators=(",", ":"))
        rows.append(row)
    return rows


def write_results(output_dir: Path, results: list[RouteResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(results, key=lambda item: (item.task_id, item.seed, item.method))
    (output_dir / "portfolio_routes.json").write_text(
        json.dumps([item.to_dict() for item in ordered], indent=2) + "\n"
    )
    rows = _serialise(ordered)
    with (output_dir / "portfolio_routes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_reported(root: Path, results: list[RouteResult], tolerance: float) -> None:
    expected_path = root / "results" / "reported_portfolio_routes.csv"
    if not expected_path.exists():
        print("reported-route verification skipped: expected file absent")
        return
    expected: dict[tuple[int, int, str], dict[str, str]] = {}
    with expected_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            expected[(int(row["task_id"]), int(row["seed"]), row["method"])] = row
    for result in results:
        key = (result.task_id, result.seed, result.method)
        row = expected.get(key)
        if row is None:
            raise AssertionError(f"Missing reported route: {key}")
        reported_order = tuple(map(int, json.loads(row["order"])))
        if result.order != reported_order:
            raise AssertionError(f"Route mismatch: {key}")
        if abs(result.length_km - float(row["length_km"])) > tolerance:
            raise AssertionError(f"Length mismatch: {key}")
        if result.method == "Portfolio" and result.selected_method != row["selected_method"]:
            raise AssertionError(f"Selected-candidate mismatch: {key}")
    print(f"reported-route verification: PASS ({len(results)} records)")


def portfolio_total(results: list[RouteResult]) -> float:
    by_task: dict[int, list[float]] = {}
    for result in results:
        if result.method == "Portfolio":
            by_task.setdefault(result.task_id, []).append(result.length_km)
    return sum(sum(values) / len(values) for values in by_task.values())


def parse_seeds(text: str | None, configured: list[int]) -> list[int]:
    if text is None:
        return list(map(int, configured))
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def parse_tasks(text: str | None, instances: list[MissionInstance]) -> list[MissionInstance]:
    if text is None:
        return instances
    selected = {int(value.strip()) for value in text.split(",") if value.strip()}
    filtered = [instance for instance in instances if instance.task_id in selected]
    if {instance.task_id for instance in filtered} != selected:
        raise ValueError(f"Unknown task id in {sorted(selected)}")
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tasks", help="comma-separated subset; default: all 30 tasks")
    parser.add_argument("--seeds", help="comma-separated subset; default: configured seeds 0--19")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    config_path = args.config or root / "configs" / "portfolio.json"
    config = json.loads(config_path.read_text())
    seeds = parse_seeds(args.seeds, config["solver_seeds"])
    instances = parse_tasks(
        args.tasks,
        load_instances(root / "results" / "frozen_instances_gps.json"),
    )
    jobs = [(instance, seed, config) for instance in instances for seed in seeds]
    if args.workers == 1:
        batches = map(_run_one, jobs)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        batches = executor.map(_run_one, jobs)
    try:
        results = [result for batch in batches for result in batch]
    finally:
        if args.workers != 1:
            executor.shutdown()

    output = args.output or root / "reproduced_results"
    write_results(output, results)
    verify_reported(root, results, float(config["tie_tolerance_km"]))
    print(f"tasks={len(instances)} seeds={len(seeds)} records={len(results)}")
    print(f"Portfolio task-aggregated total={portfolio_total(results):.12f} km")


if __name__ == "__main__":
    main()
