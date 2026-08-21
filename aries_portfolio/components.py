"""Deterministic component required by the frozen five-candidate Portfolio."""

from __future__ import annotations

import time

from .geo import closed_loop_length_km, node_distance_matrix, route_is_valid
from .types import MissionInstance, RouteResult


def _nearest_neighbor_order(instance: MissionInstance) -> tuple[int, ...]:
    matrix = node_distance_matrix(instance.home_gps, instance.targets_gps)
    remaining = set(range(len(instance.targets_gps)))
    order: list[int] = []
    node = 0
    while remaining:
        nxt = min(remaining, key=lambda index: (matrix[node, index + 1], index))
        order.append(nxt)
        remaining.remove(nxt)
        node = nxt + 1
    return tuple(order)


def solve_nn_two_opt(instance: MissionInstance, seed: int = 0) -> RouteResult:
    """Run deterministic nearest-neighbour initialisation and first-improvement 2-opt."""

    started = time.perf_counter()
    matrix = node_distance_matrix(instance.home_gps, instance.targets_gps)
    route = list(_nearest_neighbor_order(instance))
    best = closed_loop_length_km(route, matrix)
    evaluations = 1
    improved = True
    while improved:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                candidate = route[:i] + list(reversed(route[i : j + 1])) + route[j + 1 :]
                score = closed_loop_length_km(candidate, matrix)
                evaluations += 1
                if score < best - 1e-12:
                    route = candidate
                    best = score
                    improved = True
                    break
            if improved:
                break
    order = tuple(route)
    return RouteResult(
        task_id=instance.task_id,
        method="NN+2opt",
        seed=int(seed),
        order=order,
        length_km=closed_loop_length_km(order, matrix),
        runtime_s=time.perf_counter() - started,
        evaluations=evaluations,
        valid=route_is_valid(order, len(instance.targets_gps)),
        metadata={
            "initialisation": "nearest neighbour",
            "local_search": "deterministic first-improvement 2-opt",
            "tie_tolerance_km": 1e-12,
        },
    )
