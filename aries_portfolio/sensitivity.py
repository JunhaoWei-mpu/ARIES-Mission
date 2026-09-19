from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import pandas as pd

from .io import load_detection_instances, load_frozen_instances
from .solvers import solve_aco


BASE = {"alpha": 1.0, "beta": 2.0, "evaporation": 0.5, "evaluation_budget": 7500}


def _settings(config: dict) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = [{"config_id": "base", "factor": "base", "value": 0, **BASE}]
    for factor in ["alpha", "beta", "evaporation", "evaluation_budget"]:
        for value in config["aco_sensitivity"][factor]:
            if float(value) == float(BASE[factor]):
                continue
            row = {"config_id": f"{factor}_{value}", "factor": factor, "value": value, **BASE}
            row[factor] = value
            rows.append(row)
    return rows


def _job(payload: tuple) -> dict:
    instance, seed, setting, population = payload
    result = solve_aco(
        instance,
        seed=int(seed),
        eval_budget=int(setting["evaluation_budget"]),
        pop_size=int(population),
        alpha=float(setting["alpha"]),
        beta=float(setting["beta"]),
        evaporation=float(setting["evaporation"]),
    )
    return {
        "config_id": setting["config_id"],
        "factor": setting["factor"],
        "value": setting["value"],
        "task_id": instance.task_id,
        "seed": int(seed),
        "alpha": setting["alpha"],
        "beta": setting["beta"],
        "evaporation": setting["evaporation"],
        "evaluation_budget": int(setting["evaluation_budget"]),
        "length_km": result.length_km,
        "runtime_s": result.runtime_s,
        "evaluations": result.evaluations,
        "order": list(result.order),
        "history_km": list(result.history_km),
    }


def run_sensitivity(root: Path, config_path: Path) -> pd.DataFrame:
    config = json.loads(config_path.read_text())
    frozen_path = root / "results" / "frozen_instances_gps.json"
    if frozen_path.exists():
        instances = load_frozen_instances(frozen_path)
    else:
        instances = load_detection_instances(
            root / "results" / "detections_buildings.json",
            root / "data" / "nano30" / "img_lat_long_data.txt",
            root / "data" / "nano30" / "images",
        )
    settings = _settings(config)
    base = pd.read_csv(root / "results" / "routes_stochastic.csv")
    base = base[base.method == "ACO"]
    rows = [
        {
            "config_id": "base",
            "factor": "base",
            "value": 0,
            "task_id": int(row.task_id),
            "seed": int(row.seed),
            **BASE,
            "length_km": float(row.length_km),
            "runtime_s": float(row.runtime_s),
            "evaluations": int(row.evaluations),
            "order": json.loads(row.order),
            "history_km": [],
        }
        for row in base.itertuples()
    ]
    jobs = [
        (instance, seed, setting, config["solver_population"])
        for setting in settings
        if setting["config_id"] != "base"
        for instance in instances
        for seed in config["solver_seeds"]
    ]
    with ProcessPoolExecutor(max_workers=int(config["sensitivity_workers"])) as executor:
        futures = [executor.submit(_job, payload) for payload in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 200 == 0 or completed == len(futures):
                print(f"completed ACO sensitivity jobs={completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: (str(row["config_id"]), int(row["task_id"]), int(row["seed"])))
    (root / "results" / "aco_sensitivity.json").write_text(json.dumps(rows, indent=2))

    scalar_rows = [{key: value for key, value in row.items() if key not in {"order", "history_km"}} for row in rows]
    frame = pd.DataFrame(scalar_rows)
    frame.to_csv(root / "results" / "aco_sensitivity.csv", index=False)
    reference = (
        pd.read_csv(root / "results" / "routes_baselines.csv")
        .query("method == 'ReferenceExact'")
        .set_index("task_id")
        .length_km
    )
    task = frame.groupby(["config_id", "factor", "value", "task_id"], as_index=False).agg(
        mean_length_km=("length_km", "mean"),
        sd_length_km=("length_km", "std"),
        mean_runtime_s=("runtime_s", "mean"),
    )
    task["reference_km"] = task.task_id.map(reference)
    task["gap_pct"] = 100.0 * (task.mean_length_km - task.reference_km).clip(lower=0) / task.reference_km
    task.to_csv(root / "results" / "aco_sensitivity_task.csv", index=False)
    summary = task.groupby(["config_id", "factor", "value"], as_index=False).agg(
        mean_route_km=("mean_length_km", "mean"),
        total_route_km=("mean_length_km", "sum"),
        mean_gap_pct=("gap_pct", "mean"),
        max_gap_pct=("gap_pct", "max"),
        mean_runtime_s=("mean_runtime_s", "mean"),
    )
    summary.to_csv(root / "results" / "aco_sensitivity_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config_path = args.config or args.root / "configs" / "experiment.json"
    print(run_sensitivity(args.root, config_path).to_string(index=False))


if __name__ == "__main__":
    main()
