"""Portfolio subset / leave-one-out ablation.

The revised Portfolio selects the shortest of five complete candidates
(Raw, NN+2-opt, ACO, DE, PSO) for every task-seed run.  This module quantifies the
marginal contribution of each branch by re-deriving the terminal selection
over solver *subsets* from the already frozen candidate lengths in
``routes_stochastic.csv``.  No solver is re-executed: for a subset S the
per-run length is ``min_{m in S} length_km`` for that (task, seed), which is
exactly what the terminal selector would have returned had only the members
of S been generated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .statistics import bootstrap_ci

CANDIDATES = ["Raw", "NN+2opt", "ACO", "DE", "PSO"]

# Ordered ablation rows: original design, deterministic extension and leave-one-out.
SUBSETS: list[tuple[str, tuple[str, ...]]] = [
    ("ACO only", ("ACO",)),
    ("ACO + DE", ("ACO", "DE")),
    ("Original four (Raw+ACO+DE+PSO)", ("Raw", "ACO", "DE", "PSO")),
    ("Five candidates (add NN+2-opt)", ("Raw", "NN+2opt", "ACO", "DE", "PSO")),
    ("drop Raw", ("NN+2opt", "ACO", "DE", "PSO")),
    ("drop NN+2-opt", ("Raw", "ACO", "DE", "PSO")),
    ("drop ACO", ("Raw", "NN+2opt", "DE", "PSO")),
    ("drop DE", ("Raw", "NN+2opt", "ACO", "PSO")),
    ("drop PSO", ("Raw", "NN+2opt", "ACO", "DE")),
]


def _subset_task_means(stochastic: pd.DataFrame, subset: tuple[str, ...]) -> pd.Series:
    """Seed-averaged closed-loop length per task for the terminal selection over ``subset``."""
    frame = stochastic[stochastic.method.isin(subset)]
    # Shortest candidate within the subset for each (task, seed).
    per_run = frame.groupby(["task_id", "seed"], as_index=False).length_km.min()
    # Task is the inferential unit: average the per-run selection across seeds.
    return per_run.groupby("task_id").length_km.mean()


def build_ablation(root: Path) -> pd.DataFrame:
    stochastic = pd.read_csv(root / "results" / "routes_stochastic.csv")
    stochastic = stochastic[stochastic.method.isin(CANDIDATES)]
    baselines = pd.read_csv(root / "results" / "routes_baselines.csv")
    reference = (
        baselines[baselines.method == "ReferenceExact"][["task_id", "length_km"]]
        .rename(columns={"length_km": "reference_km"})
        .set_index("task_id")
        .reference_km
    )

    raw_task_means = _subset_task_means(stochastic, ("Raw",))
    raw_mean = float(raw_task_means.mean())

    rows = []
    for label, subset in SUBSETS:
        task_means = _subset_task_means(stochastic, subset)
        lo, hi = bootstrap_ci(task_means.to_numpy())
        # Exact optimality gap on all Concorde-certified tasks.
        common = task_means.index.intersection(reference.index)
        gaps = 100.0 * (task_means.loc[common] - reference.loc[common]).clip(lower=0.0) / reference.loc[common]
        rows.append(
            {
                "subset": label,
                "n_solvers": len(subset),
                "mean_km": float(task_means.mean()),
                "median_km": float(task_means.median()),
                "ci95_low_km": lo,
                "ci95_high_km": hi,
                "total_km": float(task_means.sum()),
                "improvement_vs_raw_pct": 100.0 * (raw_mean - float(task_means.mean())) / raw_mean,
                "mean_exact_gap_pct": float(gaps.mean()),
                "max_exact_gap_pct": float(gaps.max()),
                "n_exact_tasks": int(len(common)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    table = build_ablation(args.root)
    out = args.root / "results" / "ablation.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
