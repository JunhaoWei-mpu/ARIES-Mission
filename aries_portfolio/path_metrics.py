from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .geo import haversine_km
from .io import load_detection_instances, load_human_route


def resample_closed_path(nodes: list[tuple[float, float]], samples: int = 256) -> list[tuple[float, float]]:
    distances = [haversine_km(a, b) for a, b in zip(nodes[:-1], nodes[1:])]
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    if cumulative[-1] == 0:
        return [nodes[0]] * samples
    targets = np.linspace(0.0, cumulative[-1], samples)
    out = []
    for target in targets:
        edge = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(nodes) - 2)
        span = cumulative[edge + 1] - cumulative[edge]
        fraction = 0.0 if span == 0 else (target - cumulative[edge]) / span
        a, b = nodes[edge], nodes[edge + 1]
        out.append((a[0] + fraction * (b[0] - a[0]), a[1] + fraction * (b[1] - a[1])))
    return out


def sequence_rmse_m(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    distances_m = np.asarray([1000.0 * haversine_km(x, y) for x, y in zip(a, b)])
    return float(np.sqrt(np.mean(distances_m**2)))


def dtw_rmse_m(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf)
    steps = np.zeros((n + 1, m + 1), dtype=int)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (1000.0 * haversine_km(a[i - 1], b[j - 1])) ** 2
            choices = [(dp[i - 1, j], steps[i - 1, j]), (dp[i, j - 1], steps[i, j - 1]), (dp[i - 1, j - 1], steps[i - 1, j - 1])]
            previous_cost, previous_steps = min(choices, key=lambda item: item[0])
            dp[i, j] = cost + previous_cost
            steps[i, j] = previous_steps + 1
    return float(np.sqrt(dp[n, m] / max(1, steps[n, m])))


def one_way_knn_m(vlm: tuple[tuple[float, float], ...], human: tuple[tuple[float, float], ...]) -> float:
    if not vlm or not human:
        return float("nan")
    return float(np.mean([min(1000.0 * haversine_km(point, target) for target in human) for point in vlm]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = args.root
    instances = load_detection_instances(
        root / "results" / "detections_buildings.json",
        root / "data" / "nano30" / "img_lat_long_data.txt",
        root / "data" / "nano30" / "images",
    )
    routes = pd.read_csv(root / "results" / "routes_stochastic.csv")
    routes = routes[routes.seed == args.seed]
    rows = []
    for instance in instances:
        human_home, human_targets = load_human_route(root / "data" / "nano30" / "human_waypoints" / f"{instance.task_id}.waypoints")
        human_nodes = [human_home, *human_targets, human_home]
        human_sampled = resample_closed_path(human_nodes)
        knn = one_way_knn_m(instance.targets_gps, human_targets)
        for method in ["Raw", "ACO", "DE", "PSO", "Portfolio"]:
            record = routes[(routes.task_id == instance.task_id) & (routes.method == method)].iloc[0]
            order = json.loads(record.order)
            nodes = [instance.home_gps, *[instance.targets_gps[i] for i in order], instance.home_gps]
            sampled = resample_closed_path(nodes)
            rows.append(
                {
                    "task_id": instance.task_id,
                    "seed": args.seed,
                    "method": method,
                    "knn_vlm_to_human_m": knn,
                    "sequence_rmse_m": sequence_rmse_m(sampled, human_sampled),
                    "dtw_rmse_m": dtw_rmse_m(sampled, human_sampled),
                    "vlm_targets": len(instance.targets_gps),
                    "human_targets": len(human_targets),
                }
            )
    with (root / "results" / "path_similarity_seed0.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
