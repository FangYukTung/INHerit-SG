#!/usr/bin/env python3
"""Build resumable query indexes for every Main-864 ReasoningGraph map."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from additional_baselines_common import DEFAULT_SPLIT, atomic_json, load_split_rows
from compact_reasoninggraph_dsg import compact


ROOT = Path(__file__).resolve().parents[2]
MAP_ROOT = ROOT / "outputs/additional_baselines/reasoninggraph/main864_mapping_stride1"
INDEX_ROOT = ROOT / "outputs/additional_baselines/reasoninggraph/query_indexes_stride1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--map-root", type=Path, default=MAP_ROOT)
    parser.add_argument("--index-root", type=Path, default=INDEX_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--scene-shard-index", type=int, default=0)
    parser.add_argument("--scene-shard-count", type=int, default=1)
    args = parser.parse_args()

    if args.scene_shard_count < 1 or not 0 <= args.scene_shard_index < args.scene_shard_count:
        raise SystemExit("Invalid scene shard index/count")

    rows = load_split_rows(args.split)
    scenes = sorted({row["scene_id"] for row in rows})
    scenes = scenes[args.scene_shard_index::args.scene_shard_count]
    args.index_root.mkdir(parents=True, exist_ok=True)
    status_name = (
        "status.json" if args.scene_shard_count == 1
        else f"status_shard_{args.scene_shard_index}_of_{args.scene_shard_count}.json"
    )
    status_path = args.index_root / status_name
    status = {"expected": len(scenes), "complete": 0, "failed": {}, "scenes": {}}
    for ordinal, scene in enumerate(scenes, 1):
        source = args.map_root / scene / "hm3dsem/backend/dsg.json"
        output = args.index_root / f"{scene}.npz"
        if not source.exists():
            status["failed"][scene] = f"missing {source}"
            atomic_json(status_path, status)
            print(json.dumps({"scene": scene, "status": "missing", "ordinal": ordinal}), flush=True)
            continue
        started = time.time()
        try:
            result = compact(source, output, args.force)
            result.update({"source": str(source), "output": str(output), "ordinal": ordinal})
            status["scenes"][scene] = result
            status["complete"] = len(status["scenes"])
            print(json.dumps({"scene": scene, **result}, sort_keys=True), flush=True)
        except Exception as error:
            status["failed"][scene] = f"{type(error).__name__}: {error}"
            print(
                json.dumps({"scene": scene, "status": "failed", "error": str(error), "elapsed_s": time.time() - started}),
                flush=True,
            )
        atomic_json(status_path, status)
    status["status"] = "complete" if status["complete"] == len(scenes) and not status["failed"] else "incomplete"
    atomic_json(status_path, status)
    if status["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
