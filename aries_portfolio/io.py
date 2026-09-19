from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable

from .types import MissionInstance, RouteResult


def parse_metadata_text(path: str | Path) -> dict[int, dict[str, float]]:
    text = Path(path).read_text()
    pattern = re.compile(
        r"Image:\s*(\d+)\.jpg.*?NW Corner Lat:\s*([-\d.]+),.*?"
        r"NW Corner Long:\s*([-\d.]+),.*?SE Corner Lat:\s*([-\d.]+),.*?"
        r"SE Corner Long:\s*([-\d.]+)",
        re.DOTALL,
    )
    rows = {}
    for task, nw_lat, nw_lon, se_lat, se_lon in pattern.findall(text):
        rows[int(task)] = {
            "nw_lat": float(nw_lat),
            "nw_lon": float(nw_lon),
            "se_lat": float(se_lat),
            "se_lon": float(se_lon),
        }
    if len(rows) != 30:
        raise ValueError(f"Expected 30 metadata rows, found {len(rows)}")
    for task, row in rows.items():
        if row["nw_lat"] <= row["se_lat"] or row["nw_lon"] >= row["se_lon"]:
            raise ValueError(f"Degenerate metadata for task {task}: {row}")
    return rows


def percent_to_gps(point: tuple[float, float], metadata: dict[str, float]) -> tuple[float, float]:
    x, y = point
    lat = metadata["nw_lat"] + (metadata["se_lat"] - metadata["nw_lat"]) * (y / 100.0)
    lon = metadata["nw_lon"] + (metadata["se_lon"] - metadata["nw_lon"]) * (x / 100.0)
    return lat, lon


def default_home(metadata: dict[str, float]) -> tuple[float, float]:
    return percent_to_gps((10.0, 10.0), metadata)


def load_detection_instances(
    detections_path: str | Path,
    metadata_path: str | Path,
    image_dir: str | Path,
) -> list[MissionInstance]:
    metadata = parse_metadata_text(metadata_path)
    records = json.loads(Path(detections_path).read_text())
    instances = []
    for record in records:
        task = int(record["task_id"])
        points_percent = [tuple(map(float, p)) for p in record["points_percent"]]
        points_gps = tuple(percent_to_gps(p, metadata[task]) for p in points_percent)
        instances.append(
            MissionInstance(
                task_id=task,
                home_gps=default_home(metadata[task]),
                targets_gps=points_gps,
                target_type=record.get("target_type", "buildings"),
                image_path=str(Path(image_dir) / f"{task}.jpg"),
                metadata={"points_percent": points_percent, "raw_text": record.get("raw_text", "")},
            )
        )
    return sorted(instances, key=lambda x: x.task_id)


def write_frozen_instances(path: str | Path, instances: Iterable[MissionInstance]) -> None:
    """Write the exact geographic routing instances without benchmark imagery."""
    payload = []
    for instance in sorted(instances, key=lambda item: item.task_id):
        payload.append(
            {
                "task_id": instance.task_id,
                "target_type": instance.target_type,
                "home_gps": list(instance.home_gps),
                "targets_gps": [list(point) for point in instance.targets_gps],
                "detection_order": list(instance.detection_order),
                "points_percent": instance.metadata.get("points_percent", []),
            }
        )
    Path(path).write_text(json.dumps(payload, indent=2))


def load_frozen_instances(
    path: str | Path,
    image_dir: str | Path = "",
) -> list[MissionInstance]:
    """Load release-ready geographic instances independently of raw benchmark files."""
    records = json.loads(Path(path).read_text())
    instances = []
    for record in records:
        task = int(record["task_id"])
        instances.append(
            MissionInstance(
                task_id=task,
                home_gps=tuple(map(float, record["home_gps"])),
                targets_gps=tuple(tuple(map(float, point)) for point in record["targets_gps"]),
                target_type=record.get("target_type", "buildings"),
                detection_order=tuple(map(int, record.get("detection_order", []))),
                image_path=str(Path(image_dir) / f"{task}.jpg") if image_dir else "",
                metadata={"points_percent": record.get("points_percent", [])},
            )
        )
    return sorted(instances, key=lambda item: item.task_id)


def load_human_route(path: str | Path) -> tuple[tuple[float, float], tuple[tuple[float, float], ...]]:
    rows = []
    for line in Path(path).read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        command = int(parts[3])
        if command != 16:
            continue
        rows.append((int(parts[0]), float(parts[8]), float(parts[9])))
    if not rows:
        raise ValueError(f"No waypoints in {path}")
    rows.sort()
    home = (rows[0][1], rows[0][2])
    targets = tuple((lat, lon) for _, lat, lon in rows[1:])
    return home, targets


def write_results_json(path: str | Path, results: Iterable[RouteResult]) -> None:
    Path(path).write_text(json.dumps([r.to_dict() for r in results], indent=2, default=_json_default))


def _json_default(value):
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except ModuleNotFoundError:
        pass
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def write_results_csv(path: str | Path, results: Iterable[RouteResult]) -> None:
    rows = [r.to_dict() for r in results]
    columns = [
        "task_id",
        "method",
        "seed",
        "length_km",
        "runtime_s",
        "evaluations",
        "valid",
        "selected_method",
        "order",
        "metadata",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key, "") for key in columns},
                    "order": json.dumps(row["order"]),
                    "metadata": json.dumps(row["metadata"], sort_keys=True),
                }
            )
