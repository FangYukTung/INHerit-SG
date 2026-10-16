#!/usr/bin/env python3
"""Validate ViGiL3D prediction schema and annotation coverage without Torch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prediction_io import annotation_ids, load_and_validate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/ViGiL3D/annotations/data/validation-00000-of-00001.parquet"),
    )
    parser.add_argument("--partial", action="store_true", help="Allow missing annotation IDs.")
    args = parser.parse_args()

    expected, sources = annotation_ids(args.annotations)
    predictions, report = load_and_validate_predictions(
        args.predictions, expected, require_complete=not args.partial
    )
    report["coverage_by_source"] = {
        source: {
            "expected": sum(value == source for value in sources.values()),
            "submitted": sum(item_id in predictions and value == source for item_id, value in sources.items()),
        }
        for source in ("scannet", "scannetpp")
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
