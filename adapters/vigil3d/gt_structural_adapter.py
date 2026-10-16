#!/usr/bin/env python3
"""Build an oracle-proposal structural-event map for ViGiL3D.

This is the explicitly labeled GT-box evaluation protocol used by the
ViGiL3D paper.  It loads the complete query-independent object inventory from
the official scene annotations and exact evaluator AABBs.  It never loads the
ViGiL3D language annotations, query target IDs, target counts, or target names.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import yaml

from mask3d_structural_adapter import (
    ROOT,
    caption_nodes,
    choose_area,
    geometric_relations,
    latest_native_map,
    load_native_context,
    match_native,
    object_to_area,
    projected_view,
    slug,
)


DEFAULT_NATIVE_MAPS = ROOT / "outputs/vigil3d/inherit/maps"
DEFAULT_OUTPUT = ROOT / "outputs/vigil3d/gt_box/maps"
METADATA_ROOT = ROOT / "data/ViGiL3D/metric_metadata"
SCANNET_ROOT = ROOT / "data/ViGiL3D/raw/scannet/scans"
SCANNETPP_ROOT = ROOT / "data/ViGiL3D/raw/scannetpp/data"


def object_id_from_key(scene_id: str, key: str) -> int:
    prefix = f"{scene_id}_"
    if not key.startswith(prefix):
        raise ValueError(f"Metadata key does not belong to {scene_id}: {key}")
    return int(key[len(prefix):])


def load_label_map(scene_id: str) -> tuple[dict[int, str], Path]:
    if scene_id.startswith("scene"):
        path = SCANNET_ROOT / scene_id / f"{scene_id}.aggregation.json"
    else:
        path = SCANNETPP_ROOT / scene_id / "scans/segments_anno.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing official semantic annotation: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = {
        int(group["objectId"]): str(group["label"]).strip().lower()
        for group in payload.get("segGroups", [])
    }
    if not labels:
        raise ValueError(f"No object labels in {path}")
    return labels, path


def metadata_files(scene_id: str) -> list[Path]:
    family = "scannetv2" if scene_id.startswith("scene") else "scannetpp"
    return [
        METADATA_ROOT / family / "obj_aabbs_train.npz",
        METADATA_ROOT / family / "obj_aabbs_validation.npz",
    ]


def sample_box_surface(box: np.ndarray, resolution: int = 15) -> np.ndarray:
    """Return dense geometry used only to project the exact GT AABB to RGB."""
    lo, hi = box
    axes = [np.linspace(lo[i], hi[i], resolution, dtype=np.float32) for i in range(3)]
    surfaces = []
    for fixed_axis in range(3):
        free_axes = [axis for axis in range(3) if axis != fixed_axis]
        grid_a, grid_b = np.meshgrid(axes[free_axes[0]], axes[free_axes[1]], indexing="ij")
        for fixed_value in (lo[fixed_axis], hi[fixed_axis]):
            points = np.empty((resolution * resolution, 3), dtype=np.float32)
            points[:, fixed_axis] = fixed_value
            points[:, free_axes[0]] = grid_a.reshape(-1)
            points[:, free_axes[1]] = grid_b.reshape(-1)
            surfaces.append(points)
    return np.concatenate(surfaces, axis=0)


def load_gt_proposals(scene_id: str) -> tuple[list[dict], Path]:
    labels, label_path = load_label_map(scene_id)
    boxes: dict[int, np.ndarray] = {}
    for path in metadata_files(scene_id):
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = np.load(path)
        for key in payload.files:
            if key.startswith(f"{scene_id}_"):
                boxes[object_id_from_key(scene_id, key)] = np.asarray(payload[key], dtype=np.float32)
    if not boxes:
        raise ValueError(f"No evaluator boxes found for {scene_id}")
    missing_labels = sorted(set(boxes) - set(labels))
    if missing_labels:
        raise ValueError(
            f"{scene_id}: evaluator objects missing official labels: {missing_labels[:20]}"
        )
    proposals = []
    for object_id, box in sorted(boxes.items()):
        if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[1] <= box[0]):
            raise ValueError(f"{scene_id}_{object_id}: invalid evaluator AABB")
        label = labels[object_id]
        proposals.append({
            "source_index": object_id,
            "object_id": object_id,
            "node_id": f"gt_{object_id:04d}_{slug(label)}",
            "label": label,
            "score": 1.0,
            "points": sample_box_surface(box),
            "aabb": box,
            "center": box.mean(axis=0),
        })
    return proposals, label_path


def build(args: argparse.Namespace) -> Path:
    native_dir = latest_native_map(args.native_maps, args.scene_id)
    native_memory, _native_graph, native_nodes, native_relations, config = load_native_context(native_dir)
    proposals, label_path = load_gt_proposals(args.scene_id)
    matches = match_native(proposals, native_memory)
    native_areas = object_to_area(native_relations)
    trajectory = args.trajectory_root / args.scene_id / "ep_000"
    run_dir = args.output / "run_gt_box_fixed" / args.scene_id / "ep_000"
    evidence = run_dir / "proposal_views"

    memory = []
    for proposal in proposals:
        match = matches.get(proposal["node_id"])
        view = projected_view(proposal, trajectory, evidence)
        native = match["native"] if match else {}
        transferable = bool(match and (match["label_compatible"] or match["iou"] >= 0.35))
        description = (
            native.get("llm_desc") if transferable
            else f"Ground-truth proposal labeled {proposal['label']}."
        )
        area = choose_area(proposal, match, native_memory, native_areas)
        memory.append({
            "node_id": proposal["node_id"],
            "label": proposal["label"],
            "llm_desc": description,
            "global_pos_3d": proposal["center"].tolist(),
            "aabb_3d": proposal["aabb"].tolist(),
            "gt_object_id": proposal["object_id"],
            "parent_area": area,
            "proposal_source": "ViGiL3D-official-GT-box-oracle",
            "native_context_match": match["native_id"] if match else None,
            "native_context_iou": match["iou"] if match else 0.0,
            **view,
        })

    if args.caption_api:
        caption_nodes(memory, config, args.output, args.max_cost_usd)

    graph = geometric_relations(proposals)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "memory_kb.json").write_text(
        json.dumps(memory, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    with (run_dir / "spatial_graph.pkl").open("wb") as handle:
        pickle.dump(graph, handle)
    (run_dir / "spatial_graph.json").write_text(json.dumps([{"source": str(u), "target": str(v), "relation": d.get("relation", "related_to")} for u, v, d in graph.edges(data=True)]))

    hierarchy_nodes = [node for node in native_nodes if int(node.get("node_level", 3)) < 3]
    object_nodes = [{
        "node_id": item["node_id"],
        "node_level": 3,
        "semantic_tag": item["label"],
        "llm_desc": item["llm_desc"],
        "global_pos_3d": item["global_pos_3d"],
        "proposal_source": item["proposal_source"],
    } for item in memory]
    (run_dir / "nodes.jsonl").write_text(
        "\n".join(json.dumps(node, ensure_ascii=True) for node in hierarchy_nodes + object_nodes) + "\n",
        encoding="utf-8",
    )
    hierarchy_edges = [
        row for row in native_relations[1:]
        if len(row) >= 3 and row[2] in {"1", "2"}
    ]
    object_edges = [
        [item["node_id"], item["parent_area"], "3"]
        for item in memory if item.get("parent_area")
    ]
    with (run_dir / "relations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            native_relations[0] if native_relations
            else ["child_node_id", "parent_node_id", "node_level"]
        )
        writer.writerows(hierarchy_edges + object_edges)

    config.setdefault("gt_box_adapter", {}).update({
        "protocol": "ViGiL3D official oracle proposal evaluation",
        "fixed_proposal_boxes": True,
        "uses_vigil3d_evaluator_boxes": True,
        "uses_vigil3d_query_targets": False,
        "uses_vigil3d_query_text_during_mapping": False,
        "official_scene_label_file": str(label_path.resolve()),
        "native_context_map": str(native_dir),
    })
    runtime = config["llm_config"]["model_router"].setdefault("runtime", {})
    runtime["cache_dir"] = str((args.output / "model_cache").resolve())
    runtime["usage_log"] = str((args.output / "model_usage.jsonl").resolve())
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    events = [{
        "event": "gt_oracle_proposal_added",
        "node_id": item["node_id"],
        "gt_object_id": item["gt_object_id"],
        "view_frame": item.get("view_frame"),
        "native_context_match": item.get("native_context_match"),
    } for item in memory]
    events.extend(
        {"event": "geometric_relation_added", "source": source, "target": target, **edge}
        for source, target, edge in graph.edges(data=True)
    )
    (run_dir / "maintenance_events.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=True) for event in events) + "\n",
        encoding="utf-8",
    )
    audit = {
        "scene_id": args.scene_id,
        "proposals_loaded": len(proposals),
        "labels_loaded": len(proposals),
        "native_context_matches": len(matches),
        "relations": graph.number_of_edges(),
        "visual_evidence": sum(bool(item.get("cropped_image_path")) for item in memory),
        "proposal_source": "official evaluator AABBs plus official scene semantic labels",
        "uses_vigil3d_query_annotations": False,
        "oracle_gt_box_protocol": True,
    }
    (run_dir / "adapter_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (run_dir / "run_log.log").write_text(
        "[GT-box Adapter] oracle proposals and structural context built successfully\n",
        encoding="utf-8",
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--native-maps", type=Path, default=DEFAULT_NATIVE_MAPS)
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=ROOT / "data/ViGiL3D/derived/inherit_2hz",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--caption-api", action="store_true")
    parser.add_argument("--max-cost-usd", type=float, default=1.0)
    args = parser.parse_args()
    run_dir = build(args)
    print(json.dumps({"status": "complete", "run_dir": str(run_dir)}))


if __name__ == "__main__":
    main()
