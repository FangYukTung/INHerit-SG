#!/usr/bin/env python3
"""Convert official Mask3D text exports to ViGiL3D's proposal NPZ format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]


def convert_scene(
    scene: str,
    export_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    score_threshold: float,
) -> Path:
    scene_short = scene.removeprefix("scene")
    points = np.load(
        processed_dir / "validation" / f"{scene_short}.npy",
        mmap_mode="r",
    )[:, :6]
    with (processed_dir / "label_database.yaml").open() as handle:
        label_db = yaml.safe_load(handle)

    proposal_points: list[np.ndarray] = []
    proposal_labels: list[str] = []
    proposal_scores: list[str] = []
    manifest = export_dir / f"{scene}.txt"
    for line in manifest.read_text().splitlines():
        mask_rel, class_id, raw_score = line.split()
        score = float(raw_score)
        # The public ViGiL3D mirror was exported at 0.1; the downstream
        # adapter independently applies its fixed 0.25 protocol threshold.
        if score < score_threshold:
            continue
        mask = np.loadtxt(export_dir / mask_rel, dtype=np.uint8).astype(bool)
        if mask.shape != (len(points),):
            raise ValueError(
                f"{scene}: mask has {len(mask)} points, expected {len(points)}"
            )
        class_key = int(class_id)
        if class_key not in label_db:
            raise KeyError(f"{scene}: unknown ScanNet200 class id {class_key}")
        proposal_points.append(np.asarray(points[mask], dtype=np.float32))
        proposal_labels.append(str(label_db[class_key]["name"]))
        proposal_scores.append(raw_score)

    if not proposal_points:
        raise RuntimeError(f"{scene}: no proposals survived threshold")
    object_points = np.empty(len(proposal_points), dtype=object)
    object_points[:] = proposal_points
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{scene}.npz"
    np.savez_compressed(
        destination,
        ins_pcds=object_points,
        ins_labels=np.asarray(proposal_labels),
        # Preserve the public mirror's string score representation.
        ins_scores=np.asarray(proposal_scores),
    )
    print(
        {
            "scene": scene,
            "proposals": len(proposal_points),
            "points": len(points),
            "output": str(destination),
        }
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=ROOT
        / "third_party/Mask3D/eval_output"
        / "instance_evaluation_vigil3d_missing_scannet200_0/decoder_-1",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data/ViGiL3D/mask3d_missing/processed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/ViGiL3D/mask3d_proposals/scannet200",
    )
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument(
        "scenes", nargs="*", default=["scene0012_00", "scene0035_00"]
    )
    args = parser.parse_args()
    for scene in args.scenes:
        convert_scene(
            scene,
            args.export_dir,
            args.processed_dir,
            args.output_dir,
            args.score_threshold,
        )


if __name__ == "__main__":
    main()
