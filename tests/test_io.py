from pathlib import Path

import pytest

from aries_portfolio.io import load_frozen_instances, parse_metadata_text, write_frozen_instances


def test_metadata_has_valid_task18() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "nano30" / "img_lat_long_data.txt"
    if not path.exists():
        pytest.skip("Raw benchmark metadata is intentionally absent from the public release")
    metadata = parse_metadata_text(path)
    assert len(metadata) == 30
    assert metadata[18]["se_lat"] == 43.18763333333333
    assert metadata[18]["nw_lat"] > metadata[18]["se_lat"]


def test_frozen_geographic_instance_roundtrip(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    instances = load_frozen_instances(root / "results" / "frozen_instances_gps.json")
    assert len(instances) == 30
    assert sum(len(item.targets_gps) for item in instances) == 352
    target = tmp_path / "roundtrip.json"
    write_frozen_instances(target, instances)
    restored = load_frozen_instances(target)
    assert restored == instances
