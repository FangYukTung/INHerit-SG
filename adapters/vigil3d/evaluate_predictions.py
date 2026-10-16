#!/usr/bin/env python3
"""Evaluate ViGiL3D axis-aligned box predictions with the official metric."""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys

from prediction_io import annotation_ids, load_and_validate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, nargs="?")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/ViGiL3D/annotations/data/validation-00000-of-00001.parquet"),
    )
    parser.add_argument(
        "--metric-metadata",
        type=Path,
        default=Path("data/ViGiL3D/metric_metadata"),
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Allow predictions for only ScanNet or ScanNet++ (sets strict=False).",
    )
    parser.add_argument(
        "--check-install",
        action="store_true",
        help="Load all local ground truth and exit without evaluating predictions.",
    )
    parser.add_argument(
        "--slice-membership",
        type=Path,
        help="Also score every slice in a linguistic_slices.json file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from datasets import load_dataset
        from torchmetrics_ext.metrics.visual_grounding import vigil3d as vigil3d_module
    except ImportError as exc:
        raise SystemExit(
            "Missing evaluator dependencies. Install with: "
            "pip install -r adapters/vigil3d/requirements-eval.txt"
        ) from exc

    annotations = args.annotations.resolve()
    metric_metadata = args.metric_metadata.resolve()
    if not annotations.is_file():
        raise SystemExit(f"Missing local ViGiL3D annotations: {annotations}")
    if not metric_metadata.is_dir():
        raise SystemExit(f"Missing local metric metadata: {metric_metadata}")

    # The official metric normally downloads these files on every new machine.
    # Redirect those two I/O functions to the pinned local checkouts so the
    # actual official metric computation remains unchanged and works offline.
    original_load_dataset = load_dataset
    vigil3d_module.load_dataset = lambda *_args, **_kwargs: original_load_dataset(
        "parquet", data_files=str(annotations), split="train"
    )

    def local_metric_file(*, filename: str, **_kwargs: object) -> str:
        path = metric_metadata / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)

    vigil3d_module.hf_hub_download = local_metric_file
    ViGiL3DMetric = vigil3d_module.ViGiL3DMetric

    if args.check_install:
        with contextlib.redirect_stdout(sys.stderr):
            metric = ViGiL3DMetric(split="validation", strict=False)
        print(f"Official ViGiL3D metric ready: {len(metric.get_all_data_ids())} annotations")
        return
    if args.predictions is None:
        raise SystemExit("predictions is required unless --check-install is used")

    expected, _ = annotation_ids(annotations)
    try:
        raw_predictions, _ = load_and_validate_predictions(
            args.predictions, expected, require_complete=not args.partial
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid predictions: {exc}") from exc
    predictions = {
        annotation_id: torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 2, 3)
        for annotation_id, boxes in raw_predictions.items()
    }

    with contextlib.redirect_stdout(sys.stderr):
        metric = ViGiL3DMetric(split="validation", strict=not args.partial)
    result = metric(predictions)

    def make_printable(values: dict[str, object]) -> dict[str, object]:
        return {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in values.items()
        }

    printable: dict[str, object] = make_printable(result)
    if args.slice_membership:
        with args.slice_membership.open(encoding="utf-8") as handle:
            memberships = json.load(handle)
        slice_results: dict[str, object] = {}
        with contextlib.redirect_stdout(sys.stderr):
            slice_metric = ViGiL3DMetric(split="validation", strict=False)
        for name, member_ids in memberships.items():
            subset = {item_id: predictions[item_id] for item_id in member_ids if item_id in predictions}
            if not subset:
                continue
            slice_metric.update(subset)
            slice_results[name] = {
                "submitted": len(subset),
                "metrics": make_printable(slice_metric.compute()),
            }
            slice_metric.reset()
        printable = {"overall": printable, "slices": slice_results}
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
