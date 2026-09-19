from pathlib import Path

import numpy as np

from aries_portfolio.external_solvers import (
    _cycle_to_target_order,
    _tour_integer_length,
    integer_distance_matrix,
    write_explicit_tsplib,
)
from aries_portfolio.types import MissionInstance


def _instance() -> MissionInstance:
    return MissionInstance(
        task_id=1,
        home_gps=(0.0, 0.0),
        targets_gps=((0.0, 0.01), (0.01, 0.01), (0.01, 0.0)),
    )


def test_external_cycle_rotates_home_and_maps_targets() -> None:
    assert _cycle_to_target_order([2, 3, 0, 1], 4) == (0, 1, 2)


def test_integer_matrix_and_tsplib_are_symmetric(tmp_path: Path) -> None:
    _, matrix = integer_distance_matrix(_instance(), 1_000_000)
    assert matrix.dtype == np.int64
    assert np.array_equal(matrix, matrix.T)
    assert np.all(np.diag(matrix) == 0)
    output = tmp_path / "square.tsp"
    write_explicit_tsplib(output, "square", matrix)
    text = output.read_text()
    assert "EDGE_WEIGHT_FORMAT: FULL_MATRIX" in text
    assert "DIMENSION: 4" in text


def test_integer_route_length_includes_return_home() -> None:
    _, matrix = integer_distance_matrix(_instance(), 1_000_000)
    observed = _tour_integer_length((0, 1, 2), matrix)
    expected = matrix[0, 1] + matrix[1, 2] + matrix[2, 3] + matrix[3, 0]
    assert observed == expected


def test_micrometre_scale_stays_in_signed_32_bit_range() -> None:
    _, matrix = integer_distance_matrix(_instance(), 1_000_000_000)
    assert matrix.max() < np.iinfo(np.int32).max
