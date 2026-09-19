from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


METHOD_ORDER = [
    "Raw",
    "NearestNeighbor",
    "CheapestInsertion",
    "NN+2opt",
    "Insertion+2opt",
    "MultiStart2opt",
    "ILS",
    "ACO",
    "ACO×3",
    "DE",
    "PSO",
    "Portfolio",
    "MultiStart2opt-RT",
    "ILS-RT",
    "HumanPlan",
]
PRIMARY_COMPARISONS = [
    "Raw",
    "NearestNeighbor",
    "CheapestInsertion",
    "NN+2opt",
    "Insertion+2opt",
    "MultiStart2opt",
    "ILS",
    "ACO",
    "ACO×3",
    "DE",
    "PSO",
    "MultiStart2opt-RT",
    "ILS-RT",
]
TIE_TOLERANCE_KM = 1e-12


def bootstrap_ci(
    values: np.ndarray,
    seed: int = 20260711,
    reps: int = 10000,
    statistic: str = "mean",
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    samples = rng.choice(values, size=(reps, len(values)), replace=True)
    if statistic == "mean":
        estimates = samples.mean(axis=1)
    elif statistic == "median":
        estimates = np.median(samples, axis=1)
    else:
        raise ValueError(f"Unsupported bootstrap statistic: {statistic}")
    return tuple(map(float, np.percentile(estimates, [2.5, 97.5])))


def holm_adjust(pvalues: list[float]) -> list[float]:
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (n - rank) * pvalues[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def rank_biserial(differences: np.ndarray, tolerance: float = TIE_TOLERANCE_KM) -> float:
    differences = np.asarray(differences, dtype=float).copy()
    differences[np.abs(differences) <= tolerance] = 0.0
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = pd.Series(abs(nonzero)).rank(method="average").to_numpy()
    return float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())


def _task_method_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["task_id", "method"], as_index=False).agg(
        length_km=("length_km", "mean"),
        length_sd=("length_km", "std"),
        runtime_s=("runtime_s", "mean"),
        evaluations=("evaluations", "mean"),
        n_runs=("seed", "count"),
    )


def _parse_metadata(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def build_statistics(root: Path) -> None:
    config = json.loads((root / "configs" / "experiment.json").read_text())
    bootstrap_seed = int(config["bootstrap_seed"])
    bootstrap_reps = int(config["bootstrap_repetitions"])
    stochastic = pd.read_csv(root / "results" / "routes_stochastic.csv")
    baselines = pd.read_csv(root / "results" / "routes_baselines.csv")
    classical = pd.read_csv(root / "results" / "routes_classical_controls.csv")
    human = pd.read_csv(root / "results" / "human_routes.csv")
    features = pd.read_csv(root / "results" / "task_features.csv")

    stochastic_task = _task_method_summary(stochastic)
    baseline_task = _task_method_summary(baselines)
    classical_task = _task_method_summary(classical)
    task_means = pd.concat(
        [
            stochastic_task,
            baseline_task[~baseline_task.method.isin(stochastic_task.method.unique())],
            classical_task[
                ~classical_task.method.isin(
                    [*stochastic_task.method.unique(), *baseline_task.method.unique()]
                )
            ],
        ],
        ignore_index=True,
    )
    task_means["length_sd"] = task_means.length_sd.fillna(0.0)
    task_means.to_csv(root / "results" / "task_method_means.csv", index=False)

    analysis_methods = task_means[~task_means.method.isin(["ReferenceExact", "HeldKarpCheck"])].copy()
    human_rows = human.assign(
        method="HumanPlan",
        length_sd=0.0,
        runtime_s=np.nan,
        evaluations=np.nan,
        n_runs=1,
    )[["task_id", "method", "length_km", "length_sd", "runtime_s", "evaluations", "n_runs"]]
    combined = pd.concat([analysis_methods, human_rows], ignore_index=True)

    aggregate_rows = []
    for method, frame in combined.groupby("method"):
        lo, hi = bootstrap_ci(
            frame.length_km.to_numpy(), seed=bootstrap_seed, reps=bootstrap_reps, statistic="mean"
        )
        aggregate_rows.append(
            {
                "method": method,
                "n_tasks": len(frame),
                "mean_km": frame.length_km.mean(),
                "median_km": frame.length_km.median(),
                "ci95_low_km": lo,
                "ci95_high_km": hi,
                "total_km": frame.length_km.sum(),
                "mean_runtime_s": frame.runtime_s.mean(),
                "mean_evaluations": frame.evaluations.mean(),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    raw_mean = float(aggregate.loc[aggregate.method == "Raw", "mean_km"].iloc[0])
    aggregate["improvement_vs_raw_pct"] = 100.0 * (raw_mean - aggregate.mean_km) / raw_mean
    aggregate["method"] = pd.Categorical(aggregate.method, METHOD_ORDER, ordered=True)
    aggregate.sort_values("method").to_csv(root / "results" / "aggregate.csv", index=False)

    portfolio = task_means[task_means.method == "Portfolio"].set_index("task_id").length_km
    tests: list[dict[str, object]] = []
    raw_p: list[float] = []
    for method in PRIMARY_COMPARISONS:
        comparator = task_means[task_means.method == method].set_index("task_id").length_km
        common = portfolio.index.intersection(comparator.index)
        diff = comparator.loc[common].to_numpy() - portfolio.loc[common].to_numpy()
        diff[np.abs(diff) <= TIE_TOLERANCE_KM] = 0.0
        if np.all(diff == 0.0):
            statistic_value, pvalue = 0.0, 1.0
        else:
            statistic_value, pvalue = wilcoxon(diff, alternative="two-sided", zero_method="pratt")
        median_lo, median_hi = bootstrap_ci(
            diff, seed=bootstrap_seed, reps=bootstrap_reps, statistic="median"
        )
        row = {
            "comparison": f"Portfolio vs {method}",
            "n_tasks": len(common),
            "wins": int(np.sum(diff > 0)),
            "ties": int(np.sum(diff == 0)),
            "losses": int(np.sum(diff < 0)),
            "total_gain_km": float(diff.sum()),
            "mean_gain_km": float(diff.mean()),
            "median_gain_km": float(np.median(diff)),
            "median_gain_ci95_low_km": median_lo,
            "median_gain_ci95_high_km": median_hi,
            "statistic": float(statistic_value),
            "p_raw": float(pvalue),
            "rank_biserial": rank_biserial(diff),
        }
        tests.append(row)
        raw_p.append(float(pvalue))
    for row, adjusted in zip(tests, holm_adjust(raw_p)):
        row["p_holm"] = adjusted
    pd.DataFrame(tests).to_csv(root / "results" / "statistical_tests.csv", index=False)
    (root / "results" / "holm_family.json").write_text(
        json.dumps(
            {
                "family": [f"Portfolio vs {method}" for method in PRIMARY_COMPARISONS],
                "correlations_in_family": False,
                "correlations_status": "exploratory",
            },
            indent=2,
        )
    )

    wins = (
        stochastic[stochastic.method == "Portfolio"]
        .selected_method.value_counts()
        .rename_axis("selected_method")
        .reset_index(name="count")
    )
    wins["frequency"] = wins["count"] / wins["count"].sum()
    wins.to_csv(root / "results" / "selection_frequency.csv", index=False)

    reference_rows = baselines[baselines.method == "ReferenceExact"].copy()
    reference_rows["metadata_parsed"] = reference_rows.metadata.map(_parse_metadata)
    reference = reference_rows[["task_id", "length_km"]].rename(columns={"length_km": "reference_km"})
    reference["rounding_bound_km"] = reference_rows.metadata_parsed.map(
        lambda item: float(item.get("rounding_bound_km_per_tour", 0.0))
    ).to_numpy()
    gap_methods = task_means[~task_means.method.isin(["ReferenceExact", "HeldKarpCheck", "HumanPlan"])]
    gaps = gap_methods.merge(reference, on="task_id", validate="many_to_one")
    gap_delta = gaps.length_km - gaps.reference_km
    comparison_bound = 2.0 * gaps.rounding_bound_km
    if (gap_delta < -comparison_bound - TIE_TOLERANCE_KM).any():
        offending = gaps.loc[gap_delta < -comparison_bound - TIE_TOLERANCE_KM, ["task_id", "method"]]
        raise RuntimeError(f"A candidate is shorter than the rounded exact reference beyond its bound: {offending}")
    gap_delta = gap_delta.mask(gap_delta.abs() <= comparison_bound + TIE_TOLERANCE_KM, 0.0)
    gaps["gap_pct"] = 100.0 * gap_delta.clip(lower=0.0) / gaps.reference_km
    gaps.to_csv(root / "results" / "optimality_gaps.csv", index=False)
    gap_summary = gaps.groupby("method", as_index=False).agg(
        n_tasks=("task_id", "count"),
        mean_gap_pct=("gap_pct", "mean"),
        median_gap_pct=("gap_pct", "median"),
        max_gap_pct=("gap_pct", "max"),
    )
    gap_summary.to_csv(root / "results" / "exact_gap_summary.csv", index=False)

    hk = baseline_task[baseline_task.method == "HeldKarpCheck"].set_index("task_id")
    concorde = baseline_task[baseline_task.method == "ReferenceExact"].set_index("task_id")
    common_exact = hk.index.intersection(concorde.index)
    exact_crosscheck = pd.DataFrame(
        {
            "task_id": common_exact,
            "held_karp_km": hk.loc[common_exact].length_km,
            "concorde_km": concorde.loc[common_exact].length_km,
        }
    )
    exact_crosscheck["absolute_difference_km"] = (
        exact_crosscheck.held_karp_km - exact_crosscheck.concorde_km
    ).abs()
    exact_crosscheck.to_csv(root / "results" / "exact_crosscheck.csv", index=False)

    gains = features.merge(
        task_means[task_means.method == "Portfolio"][["task_id", "length_km"]].rename(
            columns={"length_km": "portfolio_km"}
        ),
        on="task_id",
    )
    gains["gain_pct"] = 100.0 * (gains.raw_length_km - gains.portfolio_km) / gains.raw_length_km
    correlations = []
    for feature in ["num_targets", "raw_crossings"]:
        rho, pvalue = spearmanr(gains[feature], gains.gain_pct)
        correlations.append(
            {
                "feature": feature,
                "spearman_rho": rho,
                "p_value": pvalue,
                "analysis_status": "exploratory; outside Holm family",
            }
        )
    gains.to_csv(root / "results" / "task_gains.csv", index=False)
    pd.DataFrame(correlations).to_csv(root / "results" / "gain_correlations.csv", index=False)

    summary = {
        "aggregate": aggregate.astype({"method": str}).to_dict(orient="records"),
        "selection_frequency": wins.to_dict(orient="records"),
        "tests": tests,
        "correlations": correlations,
        "holm_family": [f"Portfolio vs {method}" for method in PRIMARY_COMPARISONS],
    }
    (root / "results" / "summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    build_statistics(args.root)


if __name__ == "__main__":
    main()
