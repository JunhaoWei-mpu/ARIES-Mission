from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .geo import closed_loop_length_km, node_distance_matrix, route_is_valid
from .io import load_frozen_instances


def audit_results(root: Path) -> dict[str, object]:
    config = json.loads((root / "configs" / "experiment.json").read_text())
    tolerance = float(config["tie_tolerance_km"])
    stochastic = pd.read_csv(root / "results" / "routes_stochastic.csv")
    baselines = pd.read_csv(root / "results" / "routes_baselines.csv")
    classical = pd.read_csv(root / "results" / "routes_classical_controls.csv")
    formal = pd.read_csv(root / "results" / "formal_source_data.csv")
    expected_runs = 30 * len(config["solver_seeds"])
    expected_methods = {"Raw", "NN+2opt", "ACO", "DE", "PSO", "Portfolio", "ACO×3"}
    counts = stochastic.method.value_counts().to_dict()
    if set(counts) != expected_methods or any(counts[method] != expected_runs for method in expected_methods):
        raise AssertionError(f"Unexpected stochastic method counts: {counts}")
    if not stochastic.valid.all():
        raise AssertionError("At least one stochastic route is invalid")

    candidate_methods = config["portfolio_priority"]
    priority = {method: index for index, method in enumerate(candidate_methods)}
    candidate = stochastic[stochastic.method.isin(candidate_methods)]
    candidate_lengths = candidate.pivot(index=["task_id", "seed"], columns="method", values="length_km")
    portfolio = stochastic[stochastic.method == "Portfolio"].set_index(["task_id", "seed"])
    expected_length = candidate_lengths.min(axis=1)
    if not np.all(np.abs(portfolio.length_km - expected_length) <= tolerance):
        raise AssertionError("Five-candidate terminal selection is not the per-run minimum")
    expected_selected = candidate_lengths.apply(
        lambda row: min(
            [method for method in candidate_methods if abs(row[method] - row.min()) <= tolerance],
            key=lambda method: priority[method],
        ),
        axis=1,
    )
    if not (portfolio.selected_method == expected_selected).all():
        raise AssertionError("Five-candidate selected_method does not match the declared tie rule")

    multistart = stochastic[stochastic.method == "ACO×3"]
    expected_multistart_evaluations = len(config["aco_multistart_seed_offsets"]) * int(
        config["solver_evaluation_budget"]
    )
    if not (multistart.evaluations == expected_multistart_evaluations).all():
        raise AssertionError("ACO×3 does not use the declared matched total budget")

    exact = baselines[baselines.method == "ReferenceExact"]
    if len(exact) != 30 or not exact.valid.all():
        raise AssertionError("Expected 30 valid Concorde exact routes")
    if not exact.metadata.map(lambda value: json.loads(value).get("certified") is True).all():
        raise AssertionError("At least one Concorde route lacks a recorded certificate")

    expected_classical = {
        "Insertion+2opt": 30,
        "MultiStart2opt": expected_runs,
        "ILS": expected_runs,
        "MultiStart2opt-RT": expected_runs,
        "ILS-RT": expected_runs,
    }
    classical_counts = classical.method.value_counts().to_dict()
    if classical_counts != expected_classical or not classical.valid.all():
        raise AssertionError(f"Unexpected classical-control records: {classical_counts}")
    metadata = classical.metadata.map(json.loads)
    primary_ms = classical[classical.method == "MultiStart2opt"].metadata.map(json.loads)
    primary_ils = classical[classical.method == "ILS"].metadata.map(json.loads)
    runtime_ms = classical[classical.method == "MultiStart2opt-RT"].metadata.map(json.loads)
    runtime_ils = classical[classical.method == "ILS-RT"].metadata.map(json.loads)
    protocol = config["classical_controls"]
    if not primary_ms.map(lambda item: item["initial_solutions"] == protocol["multi_start_2opt"]["starts_per_task_seed"]).all():
        raise AssertionError("Primary Multi-start 2-opt budget drifted")
    if not primary_ils.map(lambda item: item["perturbations"] == protocol["ils"]["iterations_per_task_seed"]).all():
        raise AssertionError("Primary ILS budget drifted")
    if not runtime_ms.map(lambda item: item["initial_solutions"] == protocol["runtime_matched"]["multi_start_2opt_starts"]).all():
        raise AssertionError("Runtime-matched Multi-start 2-opt budget drifted")
    if not runtime_ils.map(lambda item: item["perturbations"] == protocol["runtime_matched"]["ils_iterations"]).all():
        raise AssertionError("Runtime-matched ILS budget drifted")

    if formal.method.astype(str).str.contains("LKH", case=False, regex=False).any():
        raise AssertionError("Unreported exploratory method leaked into formal source data")
    instances = {item.task_id: item for item in load_frozen_instances(root / "results" / "frozen_instances_gps.json")}
    matrices = {
        task_id: node_distance_matrix(instance.home_gps, instance.targets_gps)
        for task_id, instance in instances.items()
    }
    max_rescore_difference = 0.0
    for row in formal.itertuples(index=False):
        order = tuple(map(int, json.loads(row.route_indices)))
        instance = instances[int(row.task_id)]
        if not route_is_valid(order, len(instance.targets_gps)):
            raise AssertionError(f"Invalid formal route for task={row.task_id}, method={row.method}, seed={row.seed}")
        rescored = closed_loop_length_km(order, matrices[int(row.task_id)])
        difference = abs(rescored - float(row.route_length_km))
        max_rescore_difference = max(max_rescore_difference, difference)
        if difference > tolerance:
            raise AssertionError(f"Common-evaluator mismatch for task={row.task_id}, method={row.method}, seed={row.seed}")

    crosscheck = pd.read_csv(root / "results" / "exact_crosscheck.csv")
    if len(crosscheck) != 27 or crosscheck.absolute_difference_km.max() > 1e-12:
        raise AssertionError("Held-Karp/Concorde cross-check failed")

    report = {
        "stochastic_rows": int(len(stochastic)),
        "rows_per_stochastic_method": {key: int(value) for key, value in counts.items()},
        "valid_stochastic_routes": int(stochastic.valid.sum()),
        "five_candidate_selections_checked": int(len(portfolio)),
        "aco_multistart_budget": expected_multistart_evaluations,
        "concorde_certified_tasks": int(len(exact)),
        "classical_control_rows": int(len(classical)),
        "rows_per_classical_method": {key: int(value) for key, value in classical_counts.items()},
        "formal_source_rows": int(len(formal)),
        "formal_routes_rescored": int(len(formal)),
        "maximum_formal_rescore_difference_km": float(max_rescore_difference),
        "held_karp_crosschecks": int(len(crosscheck)),
        "maximum_held_karp_difference_km": float(crosscheck.absolute_difference_km.max()),
        "status": "pass",
    }
    (root / "results" / "audit_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(audit_results(args.root), indent=2))


if __name__ == "__main__":
    main()
