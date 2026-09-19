from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from .baselines import insertion_two_opt, iterated_local_search, multi_start_two_opt
from .io import load_frozen_instances, write_results_csv, write_results_json
from .types import MissionInstance, RouteResult


def _seeded_job(payload: tuple[MissionInstance, int, dict]) -> list[RouteResult]:
    instance, seed, protocol = payload
    primary_starts = int(protocol["multi_start_2opt"]["starts_per_task_seed"])
    primary_iterations = int(protocol["ils"]["iterations_per_task_seed"])
    runtime_protocol = protocol["runtime_matched"]
    runtime_starts = runtime_protocol["multi_start_2opt_starts"]
    runtime_iterations = runtime_protocol["ils_iterations"]
    if runtime_starts is None or runtime_iterations is None:
        raise RuntimeError("Runtime-matched budgets must be frozen before formal execution")
    return [
        multi_start_two_opt(instance, seed, starts=primary_starts, method="MultiStart2opt"),
        iterated_local_search(instance, seed, iterations=primary_iterations, method="ILS"),
        multi_start_two_opt(instance, seed, starts=int(runtime_starts), method="MultiStart2opt-RT"),
        iterated_local_search(instance, seed, iterations=int(runtime_iterations), method="ILS-RT"),
    ]


def run_formal_controls(root: Path, config_path: Path) -> list[RouteResult]:
    config = json.loads(config_path.read_text())
    protocol = config["classical_controls"]
    instances = load_frozen_instances(root / "results" / "frozen_instances_gps.json")
    if len(instances) != 30:
        raise RuntimeError(f"Expected 30 frozen tasks, found {len(instances)}")

    results: list[RouteResult] = [insertion_two_opt(instance) for instance in instances]
    jobs = [(instance, int(seed), protocol) for instance in instances for seed in config["solver_seeds"]]
    completed = 0
    with ProcessPoolExecutor(max_workers=int(protocol["workers"])) as executor:
        futures = [executor.submit(_seeded_job, job) for job in jobs]
        for future in as_completed(futures):
            results.extend(future.result())
            completed += 1
            if completed % 20 == 0 or completed == len(jobs):
                print(f"completed formal local-search jobs={completed}/{len(jobs)}", flush=True)

    results.sort(key=lambda item: (item.task_id, item.method, item.seed))
    output_json = root / "results" / "routes_classical_controls.json"
    output_csv = root / "results" / "routes_classical_controls.csv"
    write_results_json(output_json, results)
    write_results_csv(output_csv, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config or root / "configs" / "experiment.json"
    results = run_formal_controls(root, config_path)
    counts: dict[str, int] = {}
    for result in results:
        counts[result.method] = counts.get(result.method, 0) + 1
    print(json.dumps({"rows": len(results), "methods": counts}, indent=2))


if __name__ == "__main__":
    main()

