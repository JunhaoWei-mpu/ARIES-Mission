from __future__ import annotations

import pytest

from aries_portfolio import (
    MissionInstance,
    select_portfolio,
    solve_aco,
    solve_de,
    solve_nn_two_opt,
    solve_pso,
    solve_raw,
)
from aries_portfolio.geo import haversine_km, route_is_valid


def square_instance() -> MissionInstance:
    return MissionInstance(
        task_id=1,
        home_gps=(0.0, 0.0),
        targets_gps=((0.0, 0.01), (0.01, 0.01), (0.01, 0.0), (0.005, 0.005)),
        detection_order=(0, 2, 1, 3),
    )


def test_haversine_and_validity() -> None:
    assert haversine_km((0, 0), (0, 1)) == pytest.approx(111.195, rel=1e-3)
    assert route_is_valid((0, 1, 2, 3), 4)
    assert not route_is_valid((0, 1, 1, 3), 4)


@pytest.mark.parametrize("solver", [solve_aco, solve_de, solve_pso])
def test_stochastic_candidate_is_seeded_and_valid(solver) -> None:
    instance = square_instance()
    first = solver(instance, seed=7, eval_budget=120, pop_size=10)
    second = solver(instance, seed=7, eval_budget=120, pop_size=10)
    assert first.valid and second.valid
    assert first.order == second.order
    assert first.length_km == pytest.approx(second.length_km)


def test_five_candidate_terminal_selection() -> None:
    instance = square_instance()
    candidates = [
        solve_raw(instance, seed=3),
        solve_nn_two_opt(instance, seed=3),
        solve_aco(instance, seed=3, eval_budget=120, pop_size=10),
        solve_de(instance, seed=3, eval_budget=120, pop_size=10),
        solve_pso(instance, seed=3, eval_budget=120, pop_size=10),
    ]
    portfolio = select_portfolio(instance, candidates, seed=3)
    assert portfolio.valid
    assert portfolio.length_km <= min(item.length_km for item in candidates) + 1e-12
    assert portfolio.selected_method in {"Raw", "NN+2opt", "ACO", "DE", "PSO"}
