from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class MissionInstance:
    task_id: int
    home_gps: tuple[float, float]
    targets_gps: tuple[tuple[float, float], ...]
    target_type: str = "buildings"
    detection_order: tuple[int, ...] = ()
    image_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.targets_gps)
        order = self.detection_order or tuple(range(n))
        if sorted(order) != list(range(n)):
            raise ValueError(f"Invalid detection order for task {self.task_id}: {order}")
        object.__setattr__(self, "detection_order", tuple(order))


@dataclass
class RouteResult:
    task_id: int
    method: str
    seed: int
    order: tuple[int, ...]
    length_km: float
    runtime_s: float
    evaluations: int
    valid: bool
    history_km: tuple[float, ...] = ()
    selected_method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["order"] = list(self.order)
        payload["history_km"] = list(self.history_km)
        return payload
