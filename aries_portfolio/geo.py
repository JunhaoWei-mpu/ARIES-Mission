from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine_km(a: Sequence[float], b: Sequence[float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def node_distance_matrix(
    home_gps: Sequence[float], targets_gps: Iterable[Sequence[float]]
) -> np.ndarray:
    nodes = [tuple(home_gps), *[tuple(p) for p in targets_gps]]
    matrix = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            d = haversine_km(nodes[i], nodes[j])
            matrix[i, j] = matrix[j, i] = d
    return matrix


def closed_loop_length_km(order: Sequence[int], matrix: np.ndarray) -> float:
    if len(order) == 0:
        return 0.0
    nodes = [0, *[int(i) + 1 for i in order], 0]
    return float(sum(matrix[a, b] for a, b in zip(nodes[:-1], nodes[1:])))


def route_is_valid(order: Sequence[int], n_targets: int) -> bool:
    return len(order) == n_targets and sorted(map(int, order)) == list(range(n_targets))


def crossing_count(order: Sequence[int], coords_xy: np.ndarray) -> int:
    """Count proper crossings in a closed route; coords_xy includes home at row zero."""
    nodes = [0, *[int(i) + 1 for i in order], 0]

    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float(np.cross(b - a, c - a))

    count = 0
    segments = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    for i, (a, b) in enumerate(segments):
        for j, (c, d) in enumerate(segments[i + 1 :], start=i + 1):
            if len({a, b, c, d}) < 4:
                continue
            if orient(coords_xy[a], coords_xy[b], coords_xy[c]) * orient(coords_xy[a], coords_xy[b], coords_xy[d]) < 0 and orient(coords_xy[c], coords_xy[d], coords_xy[a]) * orient(coords_xy[c], coords_xy[d], coords_xy[b]) < 0:
                count += 1
    return count


def gps_to_local_xy(home_gps: Sequence[float], targets_gps: Iterable[Sequence[float]]) -> np.ndarray:
    lat0, lon0 = map(math.radians, home_gps)
    points = [tuple(home_gps), *[tuple(p) for p in targets_gps]]
    out = []
    for lat, lon in points:
        x = EARTH_RADIUS_KM * math.cos(lat0) * (math.radians(lon) - lon0)
        y = EARTH_RADIUS_KM * (math.radians(lat) - lat0)
        out.append((x, y))
    return np.asarray(out, dtype=np.float64)
