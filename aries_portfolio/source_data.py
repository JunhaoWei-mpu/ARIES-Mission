from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FORMAL_STOCHASTIC_METHODS = ["Raw", "NN+2opt", "ACO", "DE", "PSO", "ACO×3", "Portfolio"]
FORMAL_BASELINE_METHODS = ["NearestNeighbor", "CheapestInsertion", "ReferenceExact", "HeldKarpCheck"]


def _parse_json(value: Any, default: Any) -> Any:
    if isinstance(value, type(default)):
        return value
    if not isinstance(value, str) or not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _route_hash(order: list[int]) -> str:
    payload = ",".join(map(str, order)).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _search_budget(method: str, evaluations: int, metadata: dict[str, Any]) -> dict[str, Any]:
    if method in {"ACO", "DE", "PSO"}:
        return {"type": "full-route objective evaluations", "count": int(evaluations)}
    if method == "ACO×3":
        return {"type": "three independent ACO streams", "count": int(evaluations)}
    if method in {"MultiStart2opt", "MultiStart2opt-RT"}:
        return {
            "type": "fixed multi-start local search",
            "initial_solutions": int(metadata["initial_solutions"]),
            "candidate_moves_evaluated": int(metadata["candidate_moves_evaluated"]),
            "accepted_moves": int(metadata["accepted_moves"]),
            "completed_local_search_passes": int(metadata["completed_local_search_passes"]),
            "full_route_objective_evaluations": int(metadata["full_route_objective_evaluations"]),
        }
    if method in {"ILS", "ILS-RT"}:
        return {
            "type": "fixed iterated local search",
            "perturbations": int(metadata["perturbations"]),
            "candidate_moves_evaluated": int(metadata["candidate_moves_evaluated"]),
            "accepted_moves": int(metadata["accepted_moves"]),
            "completed_local_search_passes": int(metadata["completed_local_search_passes"]),
            "full_route_objective_evaluations": int(metadata["full_route_objective_evaluations"]),
        }
    if method == "Insertion+2opt":
        return {
            "type": "one constructive route plus local search to convergence",
            "insertion_candidate_evaluations": int(metadata["insertion_candidate_evaluations"]),
            "candidate_moves_evaluated": int(metadata["candidate_moves_evaluated"]),
            "accepted_moves": int(metadata["accepted_moves"]),
            "completed_local_search_passes": int(metadata["completed_local_search_passes"]),
        }
    if method == "Portfolio":
        return {
            "type": "terminal validation and minimum selection over five frozen candidates",
            "candidate_evaluations": metadata.get("candidate_evaluations", {}),
        }
    if method == "ReferenceExact":
        return {"type": "Concorde exact certification; not a search candidate"}
    if method == "HeldKarpCheck":
        return {"type": "Held-Karp independent exact cross-check", "dynamic_programming_evaluations": int(evaluations)}
    return {"type": "method-specific evaluations", "count": int(evaluations)}


def build_formal_source_data(root: Path) -> pd.DataFrame:
    stochastic_path = root / "results" / "routes_stochastic.csv"
    baseline_path = root / "results" / "routes_baselines.csv"
    classical_path = root / "results" / "routes_classical_controls.csv"
    stochastic = pd.read_csv(stochastic_path)
    stochastic = stochastic[stochastic.method.isin(FORMAL_STOCHASTIC_METHODS)].copy()
    baselines = pd.read_csv(baseline_path)
    baselines = baselines[baselines.method.isin(FORMAL_BASELINE_METHODS)].copy()
    classical = pd.read_csv(classical_path)
    combined = pd.concat([stochastic, baselines, classical], ignore_index=True)

    frozen = json.loads((root / "results" / "frozen_instances_gps.json").read_text())
    n_by_task = {int(item["task_id"]): len(item["targets_gps"]) for item in frozen}
    reference_rows = baselines[baselines.method == "ReferenceExact"].copy()
    reference_rows["metadata_parsed"] = reference_rows.metadata.map(lambda value: _parse_json(value, {}))
    reference_length = reference_rows.set_index("task_id").length_km.to_dict()
    reference_bound = reference_rows.set_index("task_id").metadata_parsed.map(
        lambda item: float(item.get("rounding_bound_km_per_tour", 0.0))
    ).to_dict()
    selected_lookup = (
        stochastic[stochastic.method == "Portfolio"]
        .set_index(["task_id", "seed"])
        .selected_method.to_dict()
    )

    rows: list[dict[str, Any]] = []
    for record in combined.to_dict(orient="records"):
        task_id = int(record["task_id"])
        seed = int(record["seed"])
        method = str(record["method"])
        order = list(map(int, _parse_json(record.get("order"), [])))
        metadata = _parse_json(record.get("metadata"), {})
        reference_km = float(reference_length[task_id])
        bound = 2.0 * float(reference_bound[task_id]) + 1e-12
        delta = float(record["length_km"]) - reference_km
        if delta < -bound:
            raise RuntimeError(f"{method} task {task_id} is shorter than the rounded exact reference beyond its bound")
        exact_gap_pct = 0.0 if abs(delta) <= bound else 100.0 * max(0.0, delta) / reference_km
        selected_method = selected_lookup.get((task_id, seed), "")
        is_selected_candidate = method in {"Raw", "NN+2opt", "ACO", "DE", "PSO"} and method == selected_method
        rows.append(
            {
                "task_id": task_id,
                "N": int(n_by_task[task_id]),
                "seed": seed,
                "method": method,
                "route_length_km": float(record["length_km"]),
                "runtime_s": float(record["runtime_s"]),
                "search_budget": json.dumps(_search_budget(method, int(record["evaluations"]), metadata), sort_keys=True),
                "validity": bool(record["valid"]),
                "route_hash": _route_hash(order),
                "route_indices": json.dumps(order),
                "exact_gap_pct": exact_gap_pct,
                "selected": bool(is_selected_candidate or method == "Portfolio"),
                "selected_method": selected_method if method == "Portfolio" else "",
            }
        )
    frame = pd.DataFrame(rows).sort_values(["task_id", "seed", "method"]).reset_index(drop=True)
    frame.to_csv(root / "results" / "formal_source_data.csv", index=False)
    (root / "results" / "formal_source_data.json").write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2) + "\n"
    )

    manifest_paths = [
        root / "results" / "frozen_instances_gps.json",
        stochastic_path,
        baseline_path,
        classical_path,
        root / "results" / "formal_source_data.csv",
        root / "results" / "formal_source_data.json",
    ]
    manifest = {
        "status": "formal reported evidence",
        "excluded_evidence_root": "internal_exploratory_unreported",
        "files": {
            path.relative_to(root).as_posix(): {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
            for path in manifest_paths
        },
        "rows": len(frame),
        "methods": {str(key): int(value) for key, value in frame.method.value_counts().sort_index().items()},
    }
    (root / "results" / "formal_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    frame = build_formal_source_data(args.root.resolve())
    print(json.dumps({"rows": len(frame), "methods": frame.method.value_counts().to_dict()}, indent=2))


if __name__ == "__main__":
    main()

