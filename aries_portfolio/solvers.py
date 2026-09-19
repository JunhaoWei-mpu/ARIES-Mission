from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np

from .geo import closed_loop_length_km, node_distance_matrix, route_is_valid
from .types import MissionInstance, RouteResult


def _result(
    instance: MissionInstance,
    method: str,
    seed: int,
    order: Sequence[int],
    matrix: np.ndarray,
    started: float,
    evaluations: int,
    history: Sequence[float] = (),
    metadata: dict | None = None,
) -> RouteResult:
    order_t = tuple(map(int, order))
    valid = route_is_valid(order_t, len(instance.targets_gps))
    length = closed_loop_length_km(order_t, matrix) if valid else float("inf")
    return RouteResult(
        task_id=instance.task_id,
        method=method,
        seed=seed,
        order=order_t,
        length_km=length,
        runtime_s=time.perf_counter() - started,
        evaluations=evaluations,
        valid=bool(valid and np.isfinite(length)),
        history_km=tuple(map(float, history)),
        metadata=metadata or {},
    )


def solve_raw(instance: MissionInstance, seed: int = 0, **_: object) -> RouteResult:
    started = time.perf_counter()
    matrix = node_distance_matrix(instance.home_gps, instance.targets_gps)
    return _result(instance, "Raw", seed, instance.detection_order, matrix, started, 1)


def solve_aco(
    instance: MissionInstance,
    seed: int,
    eval_budget: int = 7500,
    pop_size: int = 50,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation: float = 0.5,
) -> RouteResult:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    matrix = node_distance_matrix(instance.home_gps, instance.targets_gps)
    n = len(instance.targets_gps)
    if n <= 1:
        return _result(instance, "ACO", seed, tuple(range(n)), matrix, started, 1)

    pheromone = np.ones((n + 1, n + 1), dtype=np.float64)
    heuristic = np.divide(1.0, matrix + 1e-12, out=np.zeros_like(matrix), where=matrix > 0)
    np.fill_diagonal(heuristic, 0.0)
    # Keep ACO independent from the Raw recognition-order branch.  The
    # incumbent is established only by sampled ant tours, so the stated
    # objective-evaluation budget counts every route scored by ACO.
    best_order: tuple[int, ...] | None = None
    best_score = float("inf")
    evaluations = 0
    history: list[float] = []

    while evaluations < eval_budget:
        batch = min(pop_size, eval_budget - evaluations)
        tours: list[tuple[tuple[int, ...], float]] = []
        for _ in range(batch):
            current_node = 0
            unvisited = set(range(n))
            order: list[int] = []
            while unvisited:
                candidates = np.asarray(sorted(unvisited), dtype=int)
                candidate_nodes = candidates + 1
                weights = (pheromone[current_node, candidate_nodes] ** alpha) * (
                    heuristic[current_node, candidate_nodes] ** beta
                )
                if not np.isfinite(weights).all() or weights.sum() <= 0:
                    weights = np.ones_like(weights, dtype=np.float64)
                chosen = int(rng.choice(candidates, p=weights / weights.sum()))
                order.append(chosen)
                unvisited.remove(chosen)
                current_node = chosen + 1
            order_t = tuple(order)
            score = closed_loop_length_km(order_t, matrix)
            tours.append((order_t, score))
            evaluations += 1
            if score < best_score:
                best_order, best_score = order_t, score

        pheromone *= 1.0 - evaporation
        for order_t, score in tours:
            nodes = [0, *[i + 1 for i in order_t], 0]
            deposit = 1.0 / max(score, 1e-12)
            for a, b in zip(nodes[:-1], nodes[1:]):
                pheromone[a, b] += deposit
                pheromone[b, a] += deposit
        history.append(best_score)

    if best_order is None:
        raise RuntimeError("ACO did not evaluate any route")
    return _result(
        instance,
        "ACO",
        seed,
        best_order,
        matrix,
        started,
        evaluations,
        history,
        metadata={
            "alpha": alpha,
            "beta": beta,
            "evaporation": evaporation,
            "evaluation_budget": eval_budget,
            "population": pop_size,
        },
    )


def solve_pso(
    instance: MissionInstance,
    seed: int,
    eval_budget: int = 7500,
    pop_size: int = 50,
    w_max: float = 0.9,
    w_min: float = 0.4,
    c1: float = 2.0,
    c2: float = 2.0,
    v_max: float = 0.2,
) -> RouteResult:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    matrix = node_distance_matrix(instance.home_gps, instance.targets_gps)
    n = len(instance.targets_gps)
    if n <= 1:
        return _result(instance, "PSO", seed, tuple(range(n)), matrix, started, 1)

    size = min(pop_size, eval_budget)
    x = rng.random((size, n))
    v = rng.uniform(-v_max, v_max, size=(size, n))
    scores = np.asarray([closed_loop_length_km(np.argsort(row), matrix) for row in x])
    evaluations = size
    pbest_x = x.copy()
    pbest_scores = scores.copy()
    best_idx = int(np.argmin(pbest_scores))
    gbest_x = pbest_x[best_idx].copy()
    gbest_score = float(pbest_scores[best_idx])
    history = [gbest_score]
    generation = 0
    max_generations = max(1, int(np.ceil(eval_budget / size)))

    while evaluations < eval_budget:
        progress = generation / max(1, max_generations - 1)
        w = w_max - (w_max - w_min) * progress
        r1 = rng.random((size, n))
        r2 = rng.random((size, n))
        v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)
        v = np.clip(v, -v_max, v_max)
        x = np.clip(x + v, 0.0, 1.0)
        batch = min(size, eval_budget - evaluations)
        for i in range(batch):
            score = closed_loop_length_km(np.argsort(x[i]), matrix)
            evaluations += 1
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_x[i] = x[i].copy()
                if score < gbest_score:
                    gbest_score = float(score)
                    gbest_x = x[i].copy()
        history.append(gbest_score)
        generation += 1

    return _result(instance, "PSO", seed, np.argsort(gbest_x), matrix, started, evaluations, history)


def solve_de(
    instance: MissionInstance,
    seed: int,
    eval_budget: int = 7500,
    pop_size: int = 50,
    mutation: float = 0.5,
    crossover: float = 0.7,
) -> RouteResult:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    matrix = node_distance_matrix(instance.home_gps, instance.targets_gps)
    n = len(instance.targets_gps)
    if n <= 1:
        return _result(instance, "DE", seed, tuple(range(n)), matrix, started, 1)

    size = max(4, min(pop_size, eval_budget))
    population = rng.random((size, n))
    scores = np.asarray([closed_loop_length_km(np.argsort(row), matrix) for row in population])
    evaluations = size
    best_idx = int(np.argmin(scores))
    best_vector = population[best_idx].copy()
    best_score = float(scores[best_idx])
    history = [best_score]

    while evaluations < eval_budget:
        for i in range(size):
            if evaluations >= eval_budget:
                break
            pool = np.delete(np.arange(size), i)
            a, b, c = population[rng.choice(pool, size=3, replace=False)]
            mutant = np.clip(a + mutation * (b - c), 0.0, 1.0)
            mask = rng.random(n) < crossover
            mask[rng.integers(0, n)] = True
            trial = np.where(mask, mutant, population[i])
            trial_score = closed_loop_length_km(np.argsort(trial), matrix)
            evaluations += 1
            if trial_score <= scores[i]:
                population[i] = trial
                scores[i] = trial_score
                if trial_score < best_score:
                    best_score = float(trial_score)
                    best_vector = trial.copy()
        history.append(best_score)

    return _result(instance, "DE", seed, np.argsort(best_vector), matrix, started, evaluations, history)


def select_portfolio(
    instance: MissionInstance,
    candidates: Sequence[RouteResult],
    seed: int,
    tie_tolerance_km: float = 1e-12,
    priority: Sequence[str] | None = None,
    method: str = "Portfolio",
) -> RouteResult:
    started = time.perf_counter()
    priority_names = tuple(priority or ("Raw", "NN+2opt", "ACO", "DE", "PSO"))
    priority_rank = {name: rank for rank, name in enumerate(priority_names)}
    by_method = {r.method: r for r in candidates}
    required = set(priority_rank)
    if set(by_method) != required:
        raise ValueError(f"Portfolio requires exactly {sorted(required)}, received {sorted(by_method)}")
    if not all(r.valid for r in candidates):
        raise ValueError("Portfolio received an invalid route")
    shortest = min(r.length_km for r in candidates)
    tied = [r for r in candidates if abs(r.length_km - shortest) <= tie_tolerance_km]
    selected = min(tied, key=lambda r: priority_rank[r.method])
    return RouteResult(
        task_id=instance.task_id,
        method=method,
        seed=seed,
        order=selected.order,
        length_km=selected.length_km,
        runtime_s=sum(r.runtime_s for r in candidates) + (time.perf_counter() - started),
        evaluations=sum(r.evaluations for r in candidates),
        valid=True,
        selected_method=selected.method,
        metadata={
            "candidate_lengths_km": {r.method: r.length_km for r in candidates},
            "candidate_evaluations": {r.method: r.evaluations for r in candidates},
            "priority": list(priority_names),
            "tie_tolerance_km": tie_tolerance_km,
        },
    )


def select_multistart(
    instance: MissionInstance,
    candidates: Sequence[RouteResult],
    seed: int,
    method: str = "ACO×3",
    tie_tolerance_km: float = 1e-12,
) -> RouteResult:
    """Select the shortest route from repeated runs of one solver.

    Unlike :func:`select_portfolio`, method names need not be unique.  The
    first run wins a numerical tie, which makes the control deterministic.
    """
    if not candidates or not all(result.valid for result in candidates):
        raise ValueError("Multistart selection requires valid candidates")
    shortest = min(result.length_km for result in candidates)
    selected_index = next(
        index
        for index, result in enumerate(candidates)
        if abs(result.length_km - shortest) <= tie_tolerance_km
    )
    selected = candidates[selected_index]
    return RouteResult(
        task_id=instance.task_id,
        method=method,
        seed=seed,
        order=selected.order,
        length_km=selected.length_km,
        runtime_s=sum(result.runtime_s for result in candidates),
        evaluations=sum(result.evaluations for result in candidates),
        valid=True,
        selected_method=f"ACO stream {selected_index + 1}",
        metadata={
            "candidate_seeds": [result.seed for result in candidates],
            "candidate_lengths_km": [result.length_km for result in candidates],
            "tie_tolerance_km": tie_tolerance_km,
        },
    )


SOLVERS: dict[str, Callable[..., RouteResult]] = {
    "Raw": solve_raw,
    "ACO": solve_aco,
    "DE": solve_de,
    "PSO": solve_pso,
}
