"""Native CLIP retrieval over ConceptGraphs or DualMap objects.

Only query text and method-produced map contents enter ranking. Coordinate
transforms must be query-independent; GT is read only by the shared evaluator.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import pickle
import sys

import numpy as np


def transform_centers(centers, payload):
    if payload.get("uses_gt_objects") is not False:
        raise ValueError("Transform must explicitly declare uses_gt_objects=false")
    rotation = np.asarray(payload["rotation"], dtype=float)
    translation = np.asarray(payload["translation"], dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,) or not np.allclose(rotation @ rotation.T, np.eye(3)):
        raise ValueError("Invalid rigid coordinate transform")
    return np.asarray(centers) @ rotation.T + translation


def load_objects(path, method, class_names=None):
    if method == "conceptgraphs":
        with gzip.open(path, "rb") as handle:
            native = pickle.load(handle)["objects"]
    else:
        # Native pickles require the installed DualMap class definitions.
        files = sorted(Path(path).glob("*.pkl"))
        native = []
        for file in files:
            with file.open("rb") as handle:
                native.append(pickle.load(handle))
    objects = []
    for index, obj in enumerate(native):
        get = obj.get if isinstance(obj, dict) else lambda name, default=None: getattr(obj, name, default)
        feature = get("clip_ft")
        if feature is None:
            continue
        if method == "conceptgraphs":
            points = get("bbox_np")
            if points is None:
                points = get("pcd_np")
            if points is None or not len(points):
                continue
            center = np.asarray(points).mean(axis=0)
            names = get("class_name", [])
            label = names if isinstance(names, str) else (Counter(map(str, names)).most_common(1)[0][0] if names else "unknown")
        else:
            cid = get("class_id")
            label = class_names[int(cid)] if cid is not None and class_names and 0 <= int(cid) < len(class_names) else "unknown"
            pcd = get("pcd")
            points = np.asarray(pcd.points) if pcd is not None else get("pcd_points")
            center = None
            if points is not None and len(points):
                import open3d as o3d
                cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points)))
                try:
                    center = np.asarray(cloud.get_oriented_bounding_box().center)
                except RuntimeError:
                    center = np.asarray(points).mean(axis=0)
            if center is None:
                center = get("center")
            if center is None:
                bbox = get("bbox")
                if bbox is not None and np.asarray(bbox).shape == (2, 3):
                    center = np.asarray(bbox).mean(axis=0)
            if center is None:
                continue
        if hasattr(feature, "detach"):
            feature = feature.detach().cpu().numpy()
        objects.append({"id": str(get("id", index)), "label": label,
                        "position": np.asarray(center).tolist(), "feature": np.asarray(feature).reshape(-1)})
    if not objects:
        raise ValueError("No native objects with geometry and CLIP features")
    return objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("conceptgraphs", "dualmap"), required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--transform", required=True)
    parser.add_argument("--class-list", help="Exact map-time DualMap vocabulary")
    parser.add_argument("--upstream-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pretrained", help="Override native checkpoint with its local path")
    args = parser.parse_args()
    if args.upstream_root:
        sys.path.insert(0, args.upstream_root)
    if args.method == "dualmap" and not args.class_list:
        parser.error("DualMap requires --class-list matching its map-time vocabulary")
    names = Path(args.class_list).read_text().splitlines() if args.class_list else None
    objects = load_objects(args.map, args.method, names)
    positions = transform_centers([obj["position"] for obj in objects], json.loads(Path(args.transform).read_text()))
    for obj, pos in zip(objects, positions):
        obj["position"] = pos.tolist()
    import open_clip
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name, pretrained = (("ViT-H-14", "laion2b_s32b_b79k") if args.method == "conceptgraphs" else ("MobileCLIP-S2", "datacompdr"))
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=args.pretrained or pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    matrix = np.stack([obj["feature"] for obj in objects])
    matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    rows = json.loads(Path(args.manifest).read_text())["queries"]
    rows = [row for row in rows if row["scene_id"] == args.scene_id]
    if not rows:
        raise ValueError("Scene is absent from manifest")
    with Path(args.output).open("x") as handle, torch.no_grad():
        for row in rows:
            feature = model.encode_text(tokenizer([row["query_text"]]).to(device)).float().cpu().numpy()[0]
            feature /= max(np.linalg.norm(feature), 1e-12)
            scores = matrix @ feature
            order = np.argsort(-scores, kind="stable")[:5]
            candidates = [{k: objects[int(i)][k] for k in ("id", "label", "position")} for i in order]
            record = {k: row[k] for k in ("key", "scene_id", "query_id")}
            record.update(prediction=candidates[0], candidates=candidates)
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
