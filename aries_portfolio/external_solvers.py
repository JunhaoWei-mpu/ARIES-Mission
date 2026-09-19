from __future__ import annotations

import hashlib
import re
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .geo import closed_loop_length_km, node_distance_matrix, route_is_valid
from .types import MissionInstance, RouteResult


def resolve_binary(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"Required solver binary not found: {path}")
    return path.resolve()


def integer_distance_matrix(instance: MissionInstance, scale_per_km: int) -> tuple[np.ndarray, np.ndarray]:
    matrix_km = node_distance_matrix(instance.home_gps, instance.targets_gps)
    matrix_int = np.rint(matrix_km * scale_per_km).astype(np.int64)
    np.fill_diagonal(matrix_int, 0)
    if matrix_int.max(initial=0) > np.iinfo(np.int32).max:
        raise ValueError("Scaled edge weight exceeds the Concorde 32-bit integer range")
    return matrix_km, matrix_int


def write_explicit_tsplib(path: Path, name: str, matrix_int: np.ndarray) -> None:
    rows = [
        f"NAME: {name}",
        "TYPE: TSP",
        f"DIMENSION: {len(matrix_int)}",
        "EDGE_WEIGHT_TYPE: EXPLICIT",
        "EDGE_WEIGHT_FORMAT: FULL_MATRIX",
        "EDGE_WEIGHT_SECTION",
    ]
    rows.extend(" ".join(map(str, row)) for row in matrix_int.tolist())
    rows.append("EOF")
    path.write_text("\n".join(rows) + "\n")


def _pad_with_home_duplicates(matrix_int: np.ndarray, minimum_nodes: int = 10) -> np.ndarray:
    """Avoid Concorde's legacy <10-node short-edge special case.

    Duplicate home nodes preserve a metric TSP optimum because they can be
    placed next to the original home at zero cost.  The returned tour is
    checked against the unpadded objective before it is accepted.
    """
    original_nodes = len(matrix_int)
    if original_nodes >= minimum_nodes:
        return matrix_int
    padded = np.zeros((minimum_nodes, minimum_nodes), dtype=np.int64)
    padded[:original_nodes, :original_nodes] = matrix_int
    for dummy in range(original_nodes, minimum_nodes):
        padded[dummy, :original_nodes] = matrix_int[0, :]
        padded[:original_nodes, dummy] = matrix_int[:, 0]
    return padded


def _cycle_to_target_order(nodes: Sequence[int], n_nodes: int) -> tuple[int, ...]:
    cycle = [int(node) for node in nodes]
    if len(cycle) != n_nodes or sorted(cycle) != list(range(n_nodes)):
        raise ValueError(f"Invalid external-solver tour: {cycle}")
    home_position = cycle.index(0)
    rotated = cycle[home_position:] + cycle[:home_position]
    order = tuple(node - 1 for node in rotated[1:])
    if not route_is_valid(order, n_nodes - 1):
        raise ValueError(f"External tour does not map to a target permutation: {order}")
    return order


def _parse_concorde_tour(path: Path, n_nodes: int) -> tuple[int, ...]:
    values = [int(value) for value in re.findall(r"-?\d+", path.read_text())]
    if values and values[0] == n_nodes:
        values = values[1:]
    return _cycle_to_target_order(values[:n_nodes], n_nodes)


def _parse_padded_concorde_tour(path: Path, original_nodes: int, padded_nodes: int) -> tuple[int, ...]:
    values = [int(value) for value in re.findall(r"-?\d+", path.read_text())]
    if values and values[0] == padded_nodes:
        values = values[1:]
    cycle = values[:padded_nodes]
    if len(cycle) != padded_nodes or sorted(cycle) != list(range(padded_nodes)):
        raise ValueError(f"Invalid padded Concorde tour: {cycle}")
    original_cycle = [node for node in cycle if node < original_nodes]
    return _cycle_to_target_order(original_cycle, original_nodes)


def _tour_integer_length(order: Sequence[int], matrix_int: np.ndarray) -> int:
    nodes = [0, *[int(index) + 1 for index in order], 0]
    return int(sum(int(matrix_int[a, b]) for a, b in zip(nodes[:-1], nodes[1:])))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def concorde_exact(
    instance: MissionInstance,
    binary: Path,
    output_dir: Path,
    scale_per_km: int = 1_000_000,
    seed: int = 20260820,
) -> RouteResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_km, matrix_int = integer_distance_matrix(instance, scale_per_km)
    concorde_matrix = _pad_with_home_duplicates(matrix_int)
    stem = f"task{instance.task_id:02d}"
    problem = output_dir / f"{stem}.tsp"
    solution = output_dir / f"{stem}.sol"
    log = output_dir / f"{stem}.concorde.log"
    write_explicit_tsplib(problem, stem, concorde_matrix)

    command = [str(binary), "-x", "-s", str(seed), "-o", solution.name, problem.name]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=output_dir, text=True, capture_output=True, check=False)
    runtime = time.perf_counter() - started
    combined_log = completed.stdout + ("\nSTDERR:\n" + completed.stderr if completed.stderr else "")
    log.write_text(combined_log)
    if not solution.exists():
        raise RuntimeError(f"Concorde did not write a tour for task {instance.task_id}; see {log}")

    if len(concorde_matrix) == len(matrix_int):
        order = _parse_concorde_tour(solution, len(matrix_int))
    else:
        order = _parse_padded_concorde_tour(solution, len(matrix_int), len(concorde_matrix))
    integer_length = _tour_integer_length(order, matrix_int)
    lower_match = re.search(r"Final lower bound\s+([0-9.]+)", completed.stdout)
    optimum_match = re.search(r"Optimal Solution:\s*([0-9.]+)", completed.stdout)
    certified = lower_match is not None and optimum_match is not None
    if not certified:
        raise RuntimeError(f"Concorde output lacks an optimality certificate for task {instance.task_id}; see {log}")
    reported_optimum = int(round(float(optimum_match.group(1))))
    if reported_optimum != integer_length:
        raise RuntimeError(
            f"Concorde tour/objective mismatch for task {instance.task_id}: "
            f"tour={integer_length}, reported={reported_optimum}"
        )

    return RouteResult(
        task_id=instance.task_id,
        method="ReferenceExact",
        seed=seed,
        order=order,
        length_km=closed_loop_length_km(order, matrix_km),
        runtime_s=runtime,
        evaluations=0,
        valid=True,
        metadata={
            "solver": "Concorde 03.12.19 with QSopt",
            "certified": True,
            "integer_scale_per_km": scale_per_km,
            "integer_objective": integer_length,
            "rounding_bound_km_per_tour": len(matrix_int) * 0.5 / scale_per_km,
            "original_nodes": len(matrix_int),
            "concorde_nodes": len(concorde_matrix),
            "home_duplicate_padding": len(concorde_matrix) - len(matrix_int),
            "problem_sha256": _sha256(problem),
            "solution_sha256": _sha256(solution),
            "log": str(log.relative_to(output_dir.parents[1])),
            "process_returncode": completed.returncode,
            "nonzero_returncode_with_certificate": completed.returncode != 0,
        },
    )
