#!/usr/bin/env python3
"""Build a fixed-proposal INHerit-SG map from Mask3D ScanNet predictions.

The adapter deliberately keeps every submitted box equal to the AABB of a
Mask3D instance point set.  It never reads ViGiL3D target IDs, target boxes, or
target counts.  The existing structural-event map contributes only its
query-independent room/area hierarchy and, when an object can be matched
without annotations, its visual-language description.  Object relations are
recomputed from Mask3D geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import sys
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_NATIVE_MAPS = ROOT / "outputs/vigil3d/inherit/maps"
DEFAULT_OUTPUT = ROOT / "outputs/vigil3d/mask3d/maps"


def latest_native_map(root: Path, scene_id: str) -> Path:
    paths = sorted(root.glob(f"run_*/{scene_id}/ep_000"), key=lambda path: path.stat().st_mtime)
    paths.extend([root / scene_id / "ep_000", root / scene_id, root])
    for path in reversed(paths):
        required = ("memory_kb.json", "spatial_graph.pkl", "nodes.jsonl",
                    "relations.csv", "config_snapshot.yaml")
        if all((path / name).is_file() for name in required):
            memory = json.loads((path / "memory_kb.json").read_text(encoding="utf-8"))
            if memory:
                return path
    raise FileNotFoundError(f"No complete native structural-event map for {scene_id}")


def slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return result or "object"


def aabb_iou(first: np.ndarray, second: np.ndarray) -> float:
    lo = np.maximum(first[0], second[0])
    hi = np.minimum(first[1], second[1])
    intersection = float(np.prod(np.maximum(0.0, hi - lo)))
    first_volume = float(np.prod(np.maximum(0.0, first[1] - first[0])))
    second_volume = float(np.prod(np.maximum(0.0, second[1] - second[0])))
    union = first_volume + second_volume - intersection
    return intersection / union if union > 0 else 0.0


def label_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower())) - {"object", "thing"}


def compatible_labels(first: str, second: str) -> bool:
    a, b = label_tokens(first), label_tokens(second)
    return bool(a and b and (a <= b or b <= a or a & b))


def load_mask3d(path: Path, score_threshold: float) -> list[dict]:
    payload = np.load(path, allow_pickle=True)
    required = {"ins_pcds", "ins_labels", "ins_scores"}
    if not required <= set(payload.files):
        raise ValueError(f"{path} is missing Mask3D arrays: {sorted(required - set(payload.files))}")
    proposals = []
    for source_index, (points, label, raw_score) in enumerate(zip(
        payload["ins_pcds"], payload["ins_labels"], payload["ins_scores"]
    )):
        score = float(raw_score)
        xyzrgb = np.asarray(points, dtype=np.float32)
        if score < score_threshold or xyzrgb.ndim != 2 or xyzrgb.shape[0] < 10:
            continue
        xyz = xyzrgb[:, :3]
        if not np.isfinite(xyz).all():
            continue
        box = np.stack((xyz.min(axis=0), xyz.max(axis=0)))
        if np.any(box[1] <= box[0]):
            continue
        proposals.append({
            "source_index": source_index,
            "node_id": f"mask3d_{source_index:03d}_{slug(str(label))}",
            "label": str(label).strip().lower(),
            "score": score,
            "points": xyzrgb,
            "aabb": box,
            "center": xyz.mean(axis=0),
        })
    return proposals


def load_native_context(run_dir: Path) -> tuple[list[dict], nx.DiGraph, list[dict], list[list[str]], dict]:
    memory = json.loads((run_dir / "memory_kb.json").read_text(encoding="utf-8"))
    with (run_dir / "spatial_graph.pkl").open("rb") as handle:
        graph = pickle.load(handle)
    nodes = [
        json.loads(line) for line in (run_dir / "nodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with (run_dir / "relations.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    config = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    return memory, graph, nodes, rows, config


def match_native(proposals: list[dict], native_memory: list[dict]) -> dict[str, dict]:
    candidates = []
    for proposal in proposals:
        for native in native_memory:
            raw_box = native.get("aabb_3d")
            if not (isinstance(raw_box, list) and len(raw_box) == 2):
                continue
            native_box = np.asarray(raw_box, dtype=np.float32)
            iou = aabb_iou(proposal["aabb"], native_box)
            distance = float(np.linalg.norm(proposal["center"] - native_box.mean(axis=0)))
            label_ok = compatible_labels(proposal["label"], str(native.get("label", "")))
            # IoU dominates.  Label compatibility breaks ties without using
            # benchmark annotations.
            score = iou + (0.15 if label_ok else 0.0) - min(distance, 5.0) * 0.005
            # A weak geometric overlap is not enough to associate objects of
            # different classes: that can happen frequently for nested or
            # touching furniture.  Such a match would also leak the wrong
            # parent area into the replacement map.
            if (label_ok and (iou >= 0.05 or distance <= 0.75)) or iou >= 0.25:
                candidates.append((score, iou, label_ok, proposal["node_id"], native["node_id"], native))
    matches, used_mask, used_native = {}, set(), set()
    for score, iou, label_ok, mask_id, native_id, native in sorted(candidates, reverse=True):
        if mask_id in used_mask or native_id in used_native:
            continue
        matches[mask_id] = {
            "native": native, "native_id": native_id, "iou": iou,
            "label_compatible": label_ok, "match_score": score,
        }
        used_mask.add(mask_id)
        used_native.add(native_id)
    return matches


def object_to_area(relations: list[list[str]]) -> dict[str, str]:
    result = {}
    for row in relations[1:]:
        if len(row) >= 3:
            try:
                level = int(row[2])
            except ValueError:
                continue
            if level == 3:
                result[row[0]] = row[1]
    return result


def choose_area(proposal: dict, match: dict | None, native_memory: list[dict],
                native_areas: dict[str, str]) -> str | None:
    if match and match["native_id"] in native_areas:
        return native_areas[match["native_id"]]
    nearest = None
    for native in native_memory:
        area = native_areas.get(native["node_id"])
        position = native.get("global_pos_3d")
        if area is None or not (isinstance(position, list) and len(position) == 3):
            continue
        distance = float(np.linalg.norm(proposal["center"] - np.asarray(position)))
        if nearest is None or distance < nearest[0]:
            nearest = (distance, area)
    return nearest[1] if nearest else None


def projected_view(proposal: dict, trajectory: Path, output_dir: Path) -> dict:
    metadata = json.loads((trajectory / "trajectory_metadata.json").read_text(encoding="utf-8"))
    intrinsics = metadata["intrinsics"]
    width, height = metadata["image_size"]
    points = proposal["points"][:, :3]
    if len(points) > 6000:
        indices = np.linspace(0, len(points) - 1, 6000, dtype=np.int64)
        points = points[indices]
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float32)))
    best = None
    for pose_path in sorted((trajectory / "pose").glob("*.txt")):
        pose = np.loadtxt(pose_path)
        camera = (np.linalg.inv(pose) @ homogeneous.T).T[:, :3]
        depth = -camera[:, 2]
        valid_depth = depth > 0.05
        if not valid_depth.any():
            continue
        u = intrinsics["fx"] * camera[:, 0] / depth + intrinsics["cx"]
        v = intrinsics["cy"] - intrinsics["fy"] * camera[:, 1] / depth
        inside = valid_depth & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if inside.sum() < 12:
            continue
        stem = pose_path.stem
        depth_path = trajectory / "depth" / f"{stem}.png"
        visible = inside.copy()
        if depth_path.is_file():
            observed = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if observed is not None:
                px = np.clip(np.rint(u[inside]).astype(int), 0, width - 1)
                py = np.clip(np.rint(v[inside]).astype(int), 0, height - 1)
                measured = observed[py, px].astype(np.float32) / float(intrinsics["depth_scale"])
                consistent = (measured <= 0) | (np.abs(measured - depth[inside]) <= 0.20)
                visible[:] = False
                visible[np.flatnonzero(inside)] = consistent
        selected = visible if visible.sum() >= 12 else inside
        x1, x2 = np.percentile(u[selected], (1, 99))
        y1, y2 = np.percentile(v[selected], (1, 99))
        x1, y1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
        x2, y2 = min(width - 1, int(math.ceil(x2))), min(height - 1, int(math.ceil(y2)))
        area = max(1, x2 - x1) * max(1, y2 - y1)
        score = int(visible.sum()) + 0.02 * math.sqrt(area)
        if best is None or score > best["score"]:
            best = {"score": score, "stem": stem, "bbox": [x1, y1, x2, y2],
                    "visible_points": int(visible.sum())}
    if best is None:
        return {}
    rgb_path = trajectory / "rgb" / f"{best['stem']}.jpg"
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        return {}
    x1, y1, x2, y2 = best["bbox"]
    box_width, box_height = max(1, x2 - x1), max(1, y2 - y1)
    extra_x, extra_y = box_width // 2, box_height // 2
    cx1, cy1 = max(0, x1 - extra_x), max(0, y1 - extra_y)
    cx2, cy2 = min(image.shape[1], x2 + extra_x), min(image.shape[0], y2 + extra_y)
    crop = image[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 0, 255), 4)
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_path = output_dir / f"{proposal['node_id']}.jpg"
    cv2.imwrite(str(crop_path), crop)
    return {
        "full_image_path": str(rgb_path.resolve()), "bbox": best["bbox"],
        "cropped_image_path": str(crop_path.resolve()),
        "visible_points": best["visible_points"], "view_frame": best["stem"],
    }


def horizontal_overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    intersection = np.maximum(0.0, np.minimum(a[1, :2], b[1, :2]) - np.maximum(a[0, :2], b[0, :2]))
    area = float(np.prod(intersection))
    area_a = float(np.prod(np.maximum(1e-6, a[1, :2] - a[0, :2])))
    area_b = float(np.prod(np.maximum(1e-6, b[1, :2] - b[0, :2])))
    return area / area_a, area / area_b


def box_distance(a: np.ndarray, b: np.ndarray) -> float:
    separation = np.maximum(0.0, np.maximum(a[0] - b[1], b[0] - a[1]))
    return float(np.linalg.norm(separation))


def geometric_relations(proposals: list[dict]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(item["node_id"] for item in proposals)
    for index, first in enumerate(proposals):
        a = first["aabb"]
        for second in proposals[index + 1:]:
            b = second["aabb"]
            overlap_a, overlap_b = horizontal_overlap(a, b)
            relation = None
            source, target = first["node_id"], second["node_id"]
            weight = 0.0
            if np.all(a[0] >= b[0] - 0.05) and np.all(a[1] <= b[1] + 0.05):
                relation, weight = "inside", 0.95
            elif np.all(b[0] >= a[0] - 0.05) and np.all(b[1] <= a[1] + 0.05):
                source, target, relation, weight = target, source, "inside", 0.95
            else:
                gap_ab = float(a[0, 2] - b[1, 2])
                gap_ba = float(b[0, 2] - a[1, 2])
                if max(overlap_a, overlap_b) >= 0.25 and -0.08 <= gap_ab <= 0.18:
                    relation, weight = "on", 0.9
                elif max(overlap_a, overlap_b) >= 0.25 and -0.08 <= gap_ba <= 0.18:
                    source, target, relation, weight = target, source, "on", 0.9
                elif box_distance(a, b) <= 0.35:
                    relation, weight = "next_to", 0.65
            if relation:
                graph.add_edge(source, target, relation=relation, weight=weight)
                inverse = {"inside": "contains", "on": "under", "next_to": "next_to"}[relation]
                graph.add_edge(target, source, relation=inverse, weight=weight)
    return graph


def caption_nodes(memory: list[dict], config: dict, output_root: Path, max_cost_usd: float) -> None:
    from inherit_sg.common import LLMHandler

    router = config["llm_config"]["model_router"]
    runtime = router.setdefault("runtime", {})
    runtime["cache_dir"] = str((output_root / "model_cache").resolve())
    runtime["usage_log"] = str((output_root / "model_usage.jsonl").resolve())
    runtime["max_cost_usd"] = max_cost_usd
    handler = LLMHandler(config["llm_config"])
    for item in memory:
        image = item.get("cropped_image_path")
        if image and Path(image).is_file():
            item["llm_desc"] = handler.generate_leaf_node_semantics(
                image, item["label"], f"3D center: {item['global_pos_3d']}"
            )


def build(args: argparse.Namespace) -> Path:
    proposal_path = args.proposals / f"{args.scene_id}.npz"
    if not proposal_path.is_file():
        raise FileNotFoundError(proposal_path)
    native_dir = latest_native_map(args.native_maps, args.scene_id)
    native_memory, native_graph, native_nodes, native_relations, config = load_native_context(native_dir)
    proposals = load_mask3d(proposal_path, args.score_threshold)
    if not proposals:
        raise ValueError(f"No Mask3D proposals survived for {args.scene_id}")
    matches = match_native(proposals, native_memory)
    native_areas = object_to_area(native_relations)
    trajectory = args.trajectory_root / args.scene_id / "ep_000"
    run_dir = args.output / "run_mask3d_fixed" / args.scene_id / "ep_000"
    evidence = run_dir / "proposal_views"

    memory = []
    for proposal in proposals:
        match = matches.get(proposal["node_id"])
        view = projected_view(proposal, trajectory, evidence)
        native = match["native"] if match else {}
        transferable = bool(match and (match["label_compatible"] or match["iou"] >= 0.35))
        description = (
            native.get("llm_desc") if transferable
            else f"Mask3D proposal labeled {proposal['label']}."
        )
        area = choose_area(proposal, match, native_memory, native_areas)
        item = {
            "node_id": proposal["node_id"], "label": proposal["label"],
            "llm_desc": description, "global_pos_3d": proposal["center"].tolist(),
            "aabb_3d": proposal["aabb"].tolist(), "mask3d_score": proposal["score"],
            "mask3d_source_index": proposal["source_index"], "parent_area": area,
            "proposal_source": "Mask3D-ScanNet200-fixed",
            "native_context_match": match["native_id"] if match else None,
            "native_context_iou": match["iou"] if match else 0.0,
            **view,
        }
        memory.append(item)

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

    # Preserve only query-independent hierarchy levels; replace all object
    # nodes and object-to-area edges with Mask3D nodes.
    hierarchy_nodes = [node for node in native_nodes if int(node.get("node_level", 3)) < 3]
    object_nodes = [{
        "node_id": item["node_id"], "node_level": 3, "semantic_tag": item["label"],
        "llm_desc": item["llm_desc"], "global_pos_3d": item["global_pos_3d"],
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
        writer.writerow(native_relations[0] if native_relations else
                        ["child_node_id", "parent_node_id", "node_level"])
        writer.writerows(hierarchy_edges + object_edges)

    config.setdefault("mask3d_adapter", {}).update({
        "proposal_file": str(proposal_path.resolve()),
        "score_threshold": args.score_threshold,
        "fixed_proposal_boxes": True,
        "uses_vigil3d_targets": False,
        "native_context_map": str(native_dir),
    })
    runtime = config["llm_config"]["model_router"].setdefault("runtime", {})
    runtime["cache_dir"] = str((args.output / "model_cache").resolve())
    runtime["usage_log"] = str((args.output / "model_usage.jsonl").resolve())
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    events = []
    for item in memory:
        events.append({
            "event": "mask3d_fixed_proposal_added", "node_id": item["node_id"],
            "view_frame": item.get("view_frame"), "proposal_score": item["mask3d_score"],
            "native_context_match": item.get("native_context_match"),
        })
    for source, target, edge in graph.edges(data=True):
        events.append({"event": "geometric_relation_added", "source": source, "target": target, **edge})
    (run_dir / "maintenance_events.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=True) for event in events) + "\n",
        encoding="utf-8",
    )
    audit = {
        "scene_id": args.scene_id, "proposals_loaded": len(proposals),
        "native_context_matches": len(matches),
        "relations": graph.number_of_edges(),
        "visual_evidence": sum(bool(item.get("cropped_image_path")) for item in memory),
        "fixed_box_sha256_input": str(proposal_path.resolve()),
        "uses_vigil3d_targets": False,
    }
    (run_dir / "adapter_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    # Keep the artifact compatible with the existing map locator, which
    # rejects native mapping runs containing an explicit mapping error.
    (run_dir / "run_log.log").write_text(
        "[Mask3D Adapter] fixed proposals and structural context built successfully\n",
        encoding="utf-8",
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--proposals", type=Path,
                        default=ROOT / "data/ViGiL3D/mask3d_proposals/scannet200")
    parser.add_argument("--native-maps", type=Path, default=DEFAULT_NATIVE_MAPS)
    parser.add_argument("--trajectory-root", type=Path,
                        default=ROOT / "data/ViGiL3D/derived/inherit_2hz")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--caption-api", action="store_true")
    parser.add_argument("--max-cost-usd", type=float, default=0.6)
    args = parser.parse_args()
    run_dir = build(args)
    print(json.dumps({"status": "complete", "run_dir": str(run_dir)}))


if __name__ == "__main__":
    main()
