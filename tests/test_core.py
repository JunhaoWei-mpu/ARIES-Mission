from __future__ import annotations

import math

import pytest

from aries_portfolio.baselines import (
    held_karp,
    insertion_two_opt,
    iterated_local_search,
    multi_start_two_opt,
    nearest_neighbor,
    two_opt,
)
from aries_portfolio.geo import closed_loop_length_km, haversine_km, node_distance_matrix, route_is_valid
from aries_portfolio.solvers import (
    select_multistart,
    select_portfolio,
    solve_aco,
    solve_de,
    solve_pso,
    solve_raw,
)
from aries_portfolio.types import MissionInstance


def square_instance() -> MissionInstance:
    return MissionInstance(
        task_id=1,
        home_gps=(0.0, 0.0),
        targets_gps=((0.0, 0.01), (0.01, 0.01), (0.01, 0.0), (0.005, 0.005)),
        detection_order=(0, 2, 1, 3),
    )


def test_haversine_known_scale() -> None:
    assert haversine_km((0, 0), (0, 1)) == pytest.approx(111.195, rel=1e-3)


def test_closed_loop_and_validity() -> None:
    instance = square_instance()
    matrix = node_distance_matrix(instance.home_gps, instance.targets_gps)
    assert closed_loop_length_km((0, 1, 2, 3), matrix) > 0
    assert route_is_valid((0, 1, 2, 3), 4)
    assert not route_is_valid((0, 1, 1, 3), 4)


@pytest.mark.parametrize("solver", [solve_aco, solve_de, solve_pso])
def test_solver_valid_and_deterministic(solver) -> None:
    instance = square_instance()
    first = solver(instance, seed=7, eval_budget=120, pop_size=10)
    second = solver(instance, seed=7, eval_budget=120, pop_size=10)
    assert first.valid and second.valid
    assert first.order == second.order
    assert first.length_km == pytest.approx(second.length_km)
    assert first.evaluations == 120


def test_aco_budget_contains_only_sampled_ant_tours(monkeypatch) -> None:
    instance = square_instance()
    from aries_portfolio import solvers

    original = solvers.closed_loop_length_km
    calls = 0

    def counted(order, matrix):
        nonlocal calls
        calls += 1
        return original(order, matrix)

    monkeypatch.setattr(solvers, "closed_loop_length_km", counted)
    result = solvers.solve_aco(instance, seed=7, eval_budget=120, pop_size=10)
    # 120 sampled ant tours plus one final validation in _result.
    assert result.evaluations == 120
    assert calls == 121


def test_five_candidate_portfolio_dominates_candidates_and_tie_prefers_raw() -> None:
    instance = square_instance()
    candidates = [
        solve_raw(instance, seed=3),
        two_opt(instance),
        solve_aco(instance, seed=3, eval_budget=120, pop_size=10),
        solve_de(instance, seed=3, eval_budget=120, pop_size=10),
        solve_pso(instance, seed=3, eval_budget=120, pop_size=10),
    ]
    portfolio = select_portfolio(instance, candidates, seed=3)
    assert portfolio.length_km <= min(r.length_km for r in candidates) + 1e-12
    tied = [solve_raw(instance, seed=3) for _ in range(5)]
    for result, method in zip(tied, ["Raw", "NN+2opt", "ACO", "DE", "PSO"]):
        result.method = method
    assert select_portfolio(instance, tied, seed=3).selected_method == "Raw"


def test_aco_multistart_has_matched_total_budget() -> None:
    instance = square_instance()
    runs = [solve_aco(instance, seed=seed, eval_budget=120, pop_size=10) for seed in [7, 27, 47]]
    result = select_multistart(instance, runs, seed=7)
    assert result.evaluations == 360
    assert result.length_km <= min(run.length_km for run in runs) + 1e-12


def test_exact_reference_no_worse_than_local_baselines() -> None:
    instance = square_instance()
    exact = held_karp(instance)
    assert exact.length_km <= nearest_neighbor(instance).length_km + 1e-12
    assert exact.length_km <= two_opt(instance).length_km + 1e-12


def test_insertion_two_opt_is_valid_and_no_worse_than_its_initial_route() -> None:
    instance = square_instance()
    result = insertion_two_opt(instance)
    from aries_portfolio.baselines import cheapest_insertion

    assert result.valid
    assert result.length_km <= cheapest_insertion(instance).length_km + 1e-12
    assert result.metadata["completed_local_search_passes"] >= 1


def test_multi_start_two_opt_is_seeded_valid_and_contains_constructive_starts() -> None:
    instance = square_instance()
    first = multi_start_two_opt(instance, seed=5, starts=6)
    second = multi_start_two_opt(instance, seed=5, starts=6)
    assert first.valid and second.valid
    assert first.order == second.order
    assert first.length_km == pytest.approx(second.length_km)
    assert first.metadata["constructive_starts"] == 2
    assert first.metadata["seeded_random_starts"] == 4
    assert first.length_km <= two_opt(instance).length_km + 1e-12


def test_ils_is_seeded_valid_and_records_budget() -> None:
    instance = square_instance()
    first = iterated_local_search(instance, seed=9, iterations=12)
    second = iterated_local_search(instance, seed=9, iterations=12)
    assert first.valid and second.valid
    assert first.order == second.order
    assert first.length_km == pytest.approx(second.length_km)
    assert first.metadata["perturbations"] == 12
    assert first.metadata["small_instance_fallback_perturbations"] == 12
    assert sorted(first.order) == list(range(len(instance.targets_gps)))
