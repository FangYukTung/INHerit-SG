#!/usr/bin/env python3
"""Run native HOV-SG retrieval and emit the shared candidate JSONL schema."""
import argparse
import os
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_rows(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("queries", payload)
    if not isinstance(rows, list):
        raise ValueError("Manifest must be a list or contain a queries list")
    return rows


def candidate_record(obj, rank):
    center = np.asarray(obj.pcd.get_center(), dtype=float)
    return {
        "id": str(obj.object_id),
        "label": obj.name,
        "position": center.tolist(),
        "score": 1.0 / rank,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", help="Native HOV-SG query config; defaults to its upstream config")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-errors", type=int, default=0)
    args = parser.parse_args()
    upstream = os.environ.get("HOVSG_ROOT")
    if not upstream:
        parser.error("Set HOVSG_ROOT to the upstream HOV-SG checkout")
    hovsg_root = Path(upstream).resolve()
    if not hovsg_root.is_dir():
        parser.error(f"HOVSG_ROOT does not exist: {hovsg_root}")
    sys.path.insert(0, str(hovsg_root))
    from omegaconf import OmegaConf
    from hovsg.graph.graph import Graph

    rows = [row for row in load_rows(args.manifest) if row["scene_id"] == args.scene_id]
    if not rows:
        raise RuntimeError(f"No manifest rows for {args.scene_id}")
    graph_path = Path(args.graph_root) / "hm3dsem" / args.scene_id / "ep_000" / "graph"
    if not graph_path.is_dir():
        raise FileNotFoundError(graph_path)

    config = OmegaConf.load(args.config or hovsg_root / "config/visualize_query_graph.yaml")
    checkpoint = Path(str(config.models.clip.checkpoint))
    if not checkpoint.is_absolute():
        config.models.clip.checkpoint = str((hovsg_root / checkpoint).resolve())
    graph = Graph(config)
    graph.load_graph(str(graph_path))
    graph.generate_room_names(
        generate_method="view_embedding",
        default_room_types=["basement", "bathroom", "bedroom", "living room", "kitchen", "hallway"],
    )

    records = []
    errors = 0
    for row in rows:
        error = None
        objects = []
        try:
            _, _, objects = graph.query_hierarchy(row["query_text"], top_k=args.top_k)
        except Exception as exc:  # preserve a complete diagnostic record, then fail the run below
            errors += 1
            error = f"{type(exc).__name__}: {exc}"
        candidates = [
            candidate_record(obj, rank)
            for rank, obj in enumerate(objects[: args.top_k], start=1)
        ]
        records.append({
            "schema_version": 1,
            "method": "hovsg_native_hierarchy",
            "key": row["key"],
            "query_id": row["query_id"],
            "scene_id": row["scene_id"],
            "prediction": candidates[0] if candidates else None,
            "candidates": candidates,
            "label_source": "saved_hovsg_object_name",
            "adapter_error": error,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    print(json.dumps({"scene_id": args.scene_id, "queries": len(records), "errors": errors, "output": str(output)}))
    if errors > args.max_errors:
        raise RuntimeError(f"HOV-SG query errors {errors} exceed allowed {args.max_errors}")


if __name__ == "__main__":
    main()
