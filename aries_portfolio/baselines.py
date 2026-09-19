from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from .geo import closed_loop_length_km, node_distance_matrix
from .types import MissionInstance, RouteResult


def _route_result(instance: MissionInstance, method: str, order: Sequence[int], started: float, evaluations: int) -> RouteResult:
    matrix = node_distance_matrix(instance.home_gps, instance.targets_gps)
    order_t = tuple(map(int, order))
    return RouteResult(
        task_id=instance.task_id,
        method=method,
        seed=0,
        order=order_t,
        length_km=closed_loop_length_km(order_t, matrix),
        runtime_s=time.perf_counter() - started,
        evaluations=evaluations,
        valid=sorted(order_t) == list(range(len(instance.targets_gps))),
    )


def nearest_neighbor(instance: MissionInstance) -> RouteResult:
    started = time.perf_counter()
    matrix = node_distance_matrix(instance.home_gps, instance.targets_gps)
    remaining = set(range(len(instance.targets_gps)))
    order: list[int] = []
    node = 0
    evaluations = 0
    while remaining:
        nxt = min(remaining, key=lambda i: (matrix[node, i + 1], i))
        evaluations += len(remaining)
        order.append(nxt)
        remaining.remove(nxt)
        node = nxt + 1
    return _route_result(instance, "NearestNeighbor", order, started, evaluations)


def cheapest_insertion(instance: MissionInstance) -> RouteResult:
    started = time.perf_counter()
    matrix = node_distance_matrix(instance.home_gps, instance.targets_gps)
    n = len(instance.targets_gps)
    if n == 0:
        return _route_result(instance, "CheapestInsertion", (), started, 0)
    first = min(range(n), key=lambda i: (matrix[0, i + 1], i))
    route = [first]
    remaining = set(range(n)) - {first}
    evaluations = n
    while remaining:
        best: tuple[float, int, int] | None = None
        cycle_nodes = [0, *[i + 1 for i in route], 0]
        for target in sorted(remaining):
            target_node = target + 1
            for pos, (a, b) in enumerate(zip(cycle_nodes[:-1], cycle_nodes[1:])):
                delta = matrix[a, target_node] + matrix[target_node, b] - matrix[a, b]
                candidate = (float(delta), target, pos)
                evaluations += 1
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        _, target, pos = best
        route.insert(pos, target)
        remaining.remove(target)
    return _route_result(instance, "CheapestInsertion", route, started, evaluations)


def two_opt(instance: MissionInstance, initial: Sequence[int] | None = None) -> RouteResult:
    started = time.perf_counter()
    matrix = node_distance_matrix(instance.home_gps, instance.targets_gps)
    route = list(initial if initial is not None else nearest_neighbor(instance).order)
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
                    route, best, improved = candidate, score, True
                    break
            if improved:
                break
    return _route_result(instance, "NN+2opt", route, started, evaluations)


def _first_improvement_two_opt(
    initial: Sequence[int],
    matrix: np.ndarray,
    tolerance_km: float = 1e-12,
) -> tuple[tuple[int, ...], float, dict[str, int]]:
    """Run deterministic first-improvement 2-opt to a local optimum.

    Candidate moves are scanned lexicographically by ``(i, j)``.  Every
    candidate is rescored with the common closed-loop evaluator.  This is less
    optimized than a four-edge delta implementation but makes the accounting
    and objective definition directly auditable.
    """

    route = list(map(int, initial))
    best = closed_loop_length_km(route, matrix)
    candidate_moves = 0
    accepted_moves = 0
    completed_passes = 0
    full_route_evaluations = 1
    improved = True
    while improved:
        improved = False
        completed_passes += 1
        for i in range(max(0, len(route) - 1)):
            for j in range(i + 1, len(route)):
                candidate = route[:i] + list(reversed(route[i : j + 1])) + route[j + 1 :]
                score = closed_loop_length_km(candidate, matrix)
                candidate_moves += 1
                full_route_evaluations += 1
                if score < best - tolerance_km:
                    route = candidate
                    best = score
                    accepted_moves += 1
                    improved = True
                    break
            if improved:
                break
    return (
        tuple(route),
        float(best),
        {
            "candidate_moves_evaluated": candidate_moves,
            "accepted_moves": accepted_moves,
            "completed_local_search_passes": completed_passes,
            "full_route_objective_evaluations": full_route_evaluations,
        },
    )


def _merge_counts(total: dict[str, int], update: dict[str, int]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0) + int(value)


def _formal_result(
    instance: MissionInstance,
    method: str,
    seed: int,
    order: Sequence[int],
    matrix: np.ndarray,
    started: float,
    counts: dict[str, int],
    metadata: dict[str, Any],
) -> RouteResult:
    order_t = tuple(map(int, order))
    valid = sorted(order_t) == list(range(len(instance.targets_gps)))
    length_km = closed_loop_length_km(order_t, matrix) if valid else float("inf")
    full_route_evaluations = int(counts.get("full_route_objective_evaluations", 0)) + 1
    return RouteResult(
        task_id=instance.task_id,
        method=method,
        seed=int(seed),
        order=order_t,
        length_km=length_km,
        runtime_s=time.perf_counter() - started,
        evaluations=full_route_evaluations,
        valid=bool(valid and np.isfinite(length_km)),
        metadata={
            **metadata,
            **{key: int(value) for key, value in counts.items()},
            "final_common_evaluator_calls": 1,
            "full_route_objective_evaluations": full_route_evaluations,
            "tie_tolerance_km": 1e-12,
        },
    )


def insertion_two_opt(instance: MissionInstance) -> RouteResult:
    """Cheapest insertion followed by deterministic first-improvement 2-opt."""

    started = time.perf_counter()
    matrix = node_distance_matrix(instance.home_gps, instance.targets_gps)
    initial = cheapest_insertion(instance)
    order, _, counts = _first_improvement_two_opt(initial.order, matrix)
    counts["initial_solutions"] = 1
    counts["insertion_candidate_evaluations"] = int(initial.evaluations)
    return _formal_result(
        instance,
        "Insertion+2opt",
        0,
        order,
        matrix,
        started,
        counts,
        {
            "initialization": "cheapest insertion",
            "local_search": "deterministic first-improvement 2-opt",
            "budget_rule": "one insertion route; 2-opt until local optimum",
        },
    )


def multi_start_two_opt(
    instance: MissionInstance,
    seed: int,
    starts: int = 20,
    method: str = "MultiStart2opt",
) -> RouteResult:
    """Multi-start 2-opt with two constructive and seeded random starts."""

    if starts < 2:
        raise ValueError("Multi-start 2-opt requires at least two starts")
    started = time.perf_counter()
    matrix = node_distance_matrix(instance.home_gps, instance.targets_gps)
    nn = nearest_neighbor(instance)
    insertion = cheapest_insertion(instance)
    initial_orders: list[tuple[int, ...]] = [nn.order, insertion.order]
    rng = np.random.default_rng(np.random.SeedSequence([20260821, instance.task_id, int(seed)]))
    n = len(instance.targets_gps)
    for _ in range(starts - 2):
        initial_orders.append(tuple(map(int, rng.permutation(n))))

    best_order: tuple[int, ...] | None = None
    best_score = float("inf")
    best_start = -1
    counts: dict[str, int] = {
        "initial_solutions": starts,
        "constructive_starts": 2,
        "seeded_random_starts": starts - 2,
    }
    for start_index, initial in enumerate(initial_orders):
        order, score, local_counts = _first_improvement_two_opt(initial, matrix)
        _merge_counts(counts, local_counts)
        if score < best_score - 1e-12:
            best_order = order
            best_score = score
            best_start = start_index
    if best_order is None:
        raise RuntimeError("Multi-start 2-opt did not produce a route")
    return _formal_result(
        instance,
        method,
        seed,
        best_order,
        matrix,
        started,
        counts,
        {
            "initialization": "nearest neighbour, cheapest insertion, then seeded random permutations",
            "local_search": "deterministic first-improvement 2-opt",
            "starts_per_task_seed": starts,
            "best_start_index": best_start,
            "budget_rule": f"{starts} starts for every task and seed",
        },
    )


def _double_bridge(route: Sequence[int], rng: np.random.Generator) -> tuple[int, ...]:
    n = len(route)
    if n < 8:
        raise ValueError("Double-bridge perturbation requires at least eight targets")
    a, b, c, d = sorted(map(int, rng.choice(np.arange(1, n), size=4, replace=False)))
    values = list(map(int, route))
    return tuple(values[:a] + values[c:d] + values[b:c] + values[a:b] + values[d:])


def _small_instance_perturbation(route: Sequence[int], iteration: int) -> tuple[int, ...]:
    """Deterministic validity-preserving fallback for fewer than eight targets."""

    values = list(map(int, route))
    n = len(values)
    if n <= 1:
        return tuple(values)
    if n == 2:
        return tuple(reversed(values))
    i = int(iteration % n)
    offset = 2 if n > 3 else 1
    j = int((i + offset + (iteration // n)) % n)
    if i == j:
        j = (j + 1) % n
    values[i], values[j] = values[j], values[i]
    return tuple(values)


def iterated_local_search(
    instance: MissionInstance,
    seed: int,
    iterations: int = 250,
    method: str = "ILS",
) -> RouteResult:
    """Plain ILS with double-bridge perturbation and best-so-far archiving."""

    if iterations < 1:
        raise ValueError("ILS requires at least one perturbation iteration")
    started = time.perf_counter()
    matrix = node_distance_matrix(instance.home_gps, instance.targets_gps)
    rng = np.random.default_rng(np.random.SeedSequence([20260821, instance.task_id, int(seed)]))
    initial = nearest_neighbor(instance)
    current, current_score, initial_counts = _first_improvement_two_opt(initial.order, matrix)
    best_order = current
    best_score = current_score
    counts: dict[str, int] = {
        "initial_solutions": 1,
        "perturbations": 0,
        "double_bridge_perturbations": 0,
        "small_instance_fallback_perturbations": 0,
        "accepted_ils_states": 0,
        "best_archive_updates": 0,
    }
    _merge_counts(counts, initial_counts)

    for iteration in range(iterations):
        if len(current) >= 8:
            perturbed = _double_bridge(current, rng)
            counts["double_bridge_perturbations"] += 1
        else:
            perturbed = _small_instance_perturbation(current, iteration)
            counts["small_instance_fallback_perturbations"] += 1
        counts["perturbations"] += 1
        candidate, candidate_score, local_counts = _first_improvement_two_opt(perturbed, matrix)
        _merge_counts(counts, local_counts)
        # Random-walk acceptance is deliberately simple; the best-so-far route
        # is retained independently and returned at termination.
        current = candidate
        current_score = candidate_score
        counts["accepted_ils_states"] += 1
        if current_score < best_score - 1e-12:
            best_order = current
            best_score = current_score
            counts["best_archive_updates"] += 1

    return _formal_result(
        instance,
        method,
        seed,
        best_order,
        matrix,
        started,
        counts,
        {
            "initialization": "nearest neighbour followed by first-improvement 2-opt",
            "perturbation": "double bridge for N>=8; deterministic swap fallback otherwise",
            "acceptance": "accept every perturbed local optimum; return best-so-far",
            "iterations_per_task_seed": iterations,
            "budget_rule": f"{iterations} perturbation/local-search cycles for every task and seed",
        },
    )


def held_karp(instance: MissionInstance, max_targets: int = 18) -> RouteResult:
    started = time.perf_counter()
    matrix = node_distance_matrix(instance.home_gps, instance.targets_gps)
    n = len(instance.targets_gps)
    if n > max_targets:
        raise ValueError(f"Held-Karp limited to {max_targets} targets, got {n}")
    if n <= 1:
        return _route_result(instance, "HeldKarp", tuple(range(n)), started, 1)

    states = 1 << n
    dp = np.full((states, n), np.inf, dtype=np.float64)
    parent = np.full((states, n), -1, dtype=np.int16)
    for j in range(n):
        dp[1 << j, j] = matrix[0, j + 1]
    evaluations = n
    for mask in range(1, states):
        bits = [j for j in range(n) if mask & (1 << j)]
        if len(bits) <= 1:
            continue
        for j in bits:
            prev_mask = mask ^ (1 << j)
            best_cost = np.inf
            best_k = -1
            for k in bits:
                if k == j:
                    continue
                cost = dp[prev_mask, k] + matrix[k + 1, j + 1]
                evaluations += 1
                if cost < best_cost:
                    best_cost, best_k = cost, k
            dp[mask, j] = best_cost
            parent[mask, j] = best_k
    full = states - 1
    last = min(range(n), key=lambda j: dp[full, j] + matrix[j + 1, 0])
    route_rev: list[int] = []
    mask = full
    while last >= 0:
        route_rev.append(last)
        prev = int(parent[mask, last])
        mask ^= 1 << last
        last = prev
    order = tuple(reversed(route_rev))
    return _route_result(instance, "HeldKarp", order, started, evaluations)
