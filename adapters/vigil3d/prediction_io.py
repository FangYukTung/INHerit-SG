"""Dependency-light validation helpers for ViGiL3D prediction JSON files."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def annotation_ids(annotation_path: Path) -> tuple[set[str], dict[str, str]]:
    import pandas as pd

    frame = pd.read_parquet(annotation_path, columns=["scene_id", "ann_id"])
    sources: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        item_id = f"{row.scene_id}_{row.ann_id}"
        sources[item_id] = "scannet" if str(row.scene_id).startswith("scene") else "scannetpp"
    return set(sources), sources


def load_and_validate_predictions(
    path: Path,
    expected_ids: set[str] | None = None,
    *,
    require_complete: bool = True,
) -> tuple[dict[str, list[list[list[float]]]], dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Predictions must be a JSON object keyed by '<scene_id>_<ann_id>'.")

    normalized: dict[str, list[list[list[float]]]] = {}
    errors: list[str] = []
    for annotation_id, boxes in raw.items():
        if not isinstance(annotation_id, str) or not annotation_id:
            errors.append(f"Invalid annotation ID: {annotation_id!r}")
            continue
        if not isinstance(boxes, list):
            errors.append(f"{annotation_id}: boxes must be a list")
            continue
        normalized_boxes: list[list[list[float]]] = []
        for index, box in enumerate(boxes):
            if not (
                isinstance(box, list)
                and len(box) == 2
                and all(isinstance(corner, list) and len(corner) == 3 for corner in box)
            ):
                errors.append(f"{annotation_id}[{index}]: expected [[min xyz], [max xyz]]")
                continue
            try:
                minimum = [float(value) for value in box[0]]
                maximum = [float(value) for value in box[1]]
            except (TypeError, ValueError):
                errors.append(f"{annotation_id}[{index}]: coordinates must be numeric")
                continue
            if not all(math.isfinite(value) for value in minimum + maximum):
                errors.append(f"{annotation_id}[{index}]: coordinates must be finite")
                continue
            if any(low > high for low, high in zip(minimum, maximum)):
                errors.append(f"{annotation_id}[{index}]: min corner exceeds max corner")
                continue
            if any(low == high for low, high in zip(minimum, maximum)):
                errors.append(f"{annotation_id}[{index}]: box has zero extent")
                continue
            normalized_boxes.append([minimum, maximum])
        normalized[annotation_id] = normalized_boxes

    submitted_ids = set(normalized)
    expected_ids = expected_ids or set()
    missing = sorted(expected_ids - submitted_ids)
    unknown = sorted(submitted_ids - expected_ids) if expected_ids else []
    if unknown:
        errors.append(f"{len(unknown)} unknown annotation IDs")
    if require_complete and missing:
        errors.append(f"{len(missing)} missing annotation IDs")
    report = {
        "valid": not errors,
        "submitted": len(submitted_ids),
        "empty_predictions": sum(not boxes for boxes in normalized.values()),
        "predicted_boxes": sum(len(boxes) for boxes in normalized.values()),
        "missing_count": len(missing),
        "unknown_count": len(unknown),
        "missing_ids": missing,
        "unknown_ids": unknown,
        "errors": errors,
    }
    if errors:
        preview = "; ".join(errors[:5])
        raise ValueError(preview + ("; ..." if len(errors) > 5 else ""))
    return normalized, report
