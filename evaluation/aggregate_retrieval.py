#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dataset import load_source_query, load_split


ORDER = ("Basic", "Attribute", "Relation", "Negation", "Compositional", "Ambiguous")
DEFAULT_SOFT_ONTOLOGY = Path(__file__).with_name("category_ontology_soft_v1.json")


def lexical_category(value):
    return " ".join(str(value or "").lower().replace("_", " ").split())


def load_category_aliases(path):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    aliases = {}
    for canonical, values in data.get("categories", {}).items():
        canonical = lexical_category(canonical)
        for value in (canonical, *values):
            alias = lexical_category(value)
            previous = aliases.setdefault(alias, canonical)
            if previous != canonical:
                raise ValueError(f"Category alias {alias!r} maps to both {previous!r} and {canonical!r}")
    return aliases


def load_soft_category_pairs(path, aliases=None):
    if path is None:
        return set()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = set()
    for item in data.get("soft_pairs", []):
        map_label = normalize_category(item["map_label"], aliases)
        simulator_category = normalize_category(item["simulator_category"], aliases)
        pairs.add((map_label, simulator_category))
    return pairs


def normalize_category(value, aliases=None):
    category = lexical_category(value)
    return (aliases or {}).get(category, category)


def categories_compatible(map_label, simulator_category, aliases=None, soft_pairs=None):
    left = normalize_category(map_label, aliases)
    right = normalize_category(simulator_category, aliases)
    return bool(left and right and (left == right or (left, right) in (soft_pairs or set())))


def scene_objects(dataset_root: Path, row, aliases):
    path = dataset_root / row["source_file"]
    scene_path = next(path.parent.glob("scene_*_ep_000.json"))
    data = json.loads(scene_path.read_text(encoding="utf-8"))
    objects = {}
    for item in data.get("objects_registry", []):
        if item.get("id") is None or not (item.get("obb_center") or item.get("position")):
            continue
        record = {
            "position": np.asarray(item.get("obb_center") or item.get("position"), dtype=float),
            "category": normalize_category(item.get("category"), aliases),
        }
        transform = item.get("obb_local_to_world")
        if transform is not None:
            axes = np.asarray(transform, dtype=float)[:3, :3]
            half_extents = np.linalg.norm(axes, axis=0)
            if np.all(half_extents > 1e-9):
                record["obb_axes"] = axes / half_extents
                record["obb_half_extents"] = half_extents
        objects[int(item["id"])] = record
    return objects


def point_to_object_distance(position, item):
    position = np.asarray(position, dtype=float)
    if "obb_axes" not in item or "obb_half_extents" not in item:
        return float(np.linalg.norm(position - item["position"]))
    local = item["obb_axes"].T @ (position - item["position"])
    outside = np.maximum(np.abs(local) - item["obb_half_extents"], 0.0)
    return float(np.linalg.norm(outside))


def align_candidate(candidate, objects, identity_threshold, aliases=None):
    if not candidate or candidate.get("position") is None or not candidate.get("label"):
        return None
    position = np.asarray(candidate["position"], dtype=float)
    category = normalize_category(candidate["label"], aliases)
    matches = [
        (float(np.linalg.norm(position - item["position"])), object_id)
        for object_id, item in objects.items() if item["category"] == category
    ]
    if not matches:
        return None
    distance, object_id = min(matches)
    return object_id if distance <= identity_threshold else None


def align_soft_candidate(candidate, objects, identity_threshold, aliases=None, soft_pairs=None):
    if not candidate or candidate.get("position") is None or not candidate.get("label"):
        return None
    position = np.asarray(candidate["position"], dtype=float)
    map_category = normalize_category(candidate["label"], aliases)
    nearby = [
        (float(np.linalg.norm(position - item["position"])), object_id, item["category"])
        for object_id, item in objects.items()
    ]
    nearby = [item for item in nearby if item[0] <= identity_threshold]
    exact = [
        (distance, object_id)
        for distance, object_id, category in nearby
        if map_category == normalize_category(category, aliases)
    ]
    if exact:
        return min(exact)[1]
    compatible = [
        (distance, object_id)
        for distance, object_id, category in nearby
        if (map_category, normalize_category(category, aliases)) in (soft_pairs or set())
    ]
    return min(compatible)[1] if compatible else None


def align_obb_soft_candidate(
    candidate, objects, identity_threshold, aliases=None, soft_pairs=None, tie_margin=0.05,
):
    if not candidate or candidate.get("position") is None or not objects:
        return None
    distances = [
        (point_to_object_distance(candidate["position"], item), object_id, item)
        for object_id, item in objects.items()
    ]
    eligible = [item for item in distances if item[0] <= identity_threshold]
    if not eligible:
        return None
    minimum = min(item[0] for item in eligible)
    geometric_ties = [item for item in eligible if item[0] <= minimum + tie_margin]
    compatible = [
        item for item in geometric_ties
        if categories_compatible(candidate.get("label"), item[2]["category"], aliases, soft_pairs)
    ]
    pool = compatible or geometric_ties
    return min(pool, key=lambda item: (item[0], item[1]))[1]


def align_nearest_candidate(candidate, objects, identity_threshold):
    if not candidate or candidate.get("position") is None or not objects:
        return None
    position = np.asarray(candidate["position"], dtype=float)
    distance, object_id = min(
        (float(np.linalg.norm(position - item["position"])), object_id)
        for object_id, item in objects.items()
    )
    return object_id if distance <= identity_threshold else None


def evaluate_record(
    record, row, query, objects, threshold, identity_threshold, aliases=None,
    identity_alignment="category", soft_pairs=None, obb_tie_margin=0.05,
):
    gt_ids = {int(value) for value in query.get("ground_truth", {}).get("object_ids", [])}
    candidates = record.get("candidates") or ([record["prediction"]] if record.get("prediction") else [])
    if identity_alignment == "nearest":
        aligned = [align_nearest_candidate(candidate, objects, identity_threshold) for candidate in candidates[:5]]
    elif identity_alignment == "category":
        aligned = [align_candidate(candidate, objects, identity_threshold, aliases) for candidate in candidates[:5]]
    elif identity_alignment == "soft-category":
        aligned = [
            align_soft_candidate(candidate, objects, identity_threshold, aliases, soft_pairs)
            for candidate in candidates[:5]
        ]
    elif identity_alignment == "obb-soft":
        aligned = [
            align_obb_soft_candidate(
                candidate, objects, identity_threshold, aliases, soft_pairs, obb_tie_margin
            )
            for candidate in candidates[:5]
        ]
    else:
        raise ValueError(f"Unknown identity alignment: {identity_alignment}")
    prediction = record.get("prediction") or (candidates[0] if candidates else None)
    distances = []
    if prediction and prediction.get("position") is not None:
        predicted = np.asarray(prediction["position"], dtype=float)
        distances = [float(np.linalg.norm(predicted - objects[object_id]["position"])) for object_id in gt_ids if object_id in objects]
    target_category = normalize_category(
        query.get("query", {}).get("target_category") or row.get("target_category"), aliases
    )
    predicted_category = normalize_category((prediction or {}).get("label"), aliases)
    if identity_alignment in {"soft-category", "obb-soft"}:
        category_consistent = categories_compatible(
            predicted_category, target_category, aliases, soft_pairs
        )
    else:
        category_consistent = bool(predicted_category and predicted_category == target_category)
    geometric_success = bool(distances and min(distances) <= threshold)
    spatial_success = geometric_success if identity_alignment == "obb-soft" else bool(
        category_consistent and geometric_success
    )
    return {
        "key": row["key"], "group": row["coarse_group"], "query_type": row["query_type"],
        "scene_id": row["scene_id"],
        "strict_top1": bool(aligned and aligned[0] in gt_ids),
        "recall_at_5": any(object_id in gt_ids for object_id in aligned if object_id is not None),
        "sr_at_1m": spatial_success,
        "geometric_sr_at_1m": geometric_success,
        "category_consistent": category_consistent,
        "distance": min(distances) if distances else float("inf"),
        "aligned_top1_id": aligned[0] if aligned else None,
    }


def evaluate_file(
    path: Path, split_rows, dataset_root: Path, threshold: float,
    identity_threshold: float, aliases, identity_alignment="category", soft_pairs=None,
    obb_tie_margin=0.05,
):
    rows_by_key = {row["key"]: row for row in split_rows}
    center_cache = {}
    outcomes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        row = rows_by_key[record["key"]]
        query = load_source_query(str(dataset_root), row)
        if row["scene_id"] not in center_cache:
            center_cache[row["scene_id"]] = scene_objects(dataset_root, row, aliases)
        objects = center_cache[row["scene_id"]]
        outcomes.append(evaluate_record(
            record, row, query, objects, threshold, identity_threshold, aliases,
            identity_alignment=identity_alignment, soft_pairs=soft_pairs,
            obb_tie_margin=obb_tie_margin,
        ))
    return outcomes


def validate_complete(paths, split_rows):
    expected = {row["key"] for row in split_rows}
    observed = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                observed.append(json.loads(line).get("key"))
    duplicates = sorted(key for key, count in Counter(observed).items() if count > 1)
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if duplicates or missing or extra:
        raise ValueError(
            f"Incomplete predictions: expected={len(expected)} observed={len(observed)} "
            f"missing={len(missing)} duplicate={len(duplicates)} extra={len(extra)}; "
            f"examples missing={missing[:3]} duplicate={duplicates[:3]} extra={extra[:3]}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root")
    parser.add_argument("--split", default="data/splits/main_864.json")
    parser.add_argument("--dataset-root", default="data/reviewed")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--identity-threshold", type=float)
    parser.add_argument(
        "--identity-alignment", choices=("obb-soft",),
        default="obb-soft",
    )
    parser.add_argument("--obb-tie-margin", type=float, default=0.05)
    parser.add_argument("--category-ontology", default=str(DEFAULT_SOFT_ONTOLOGY))
    parser.add_argument("--output", default="outputs/evaluation")
    parser.add_argument("--require-complete", action="store_true", default=True)
    args = parser.parse_args()
    if args.identity_threshold is None:
        args.identity_threshold = 0.25 if args.identity_alignment == "obb-soft" else 1.0
    aliases = load_category_aliases(args.category_ontology)
    soft_pairs = load_soft_category_pairs(args.category_ontology, aliases)
    split_rows = load_split(args.split)["queries"]
    result_root = Path(args.result_root)
    summaries = []
    for variant_dir in sorted(path for path in result_root.iterdir() if path.is_dir() and path.name != "parser_cache"):
        outcomes = []
        prediction_paths = sorted(variant_dir.glob("*.jsonl"))
        if not prediction_paths:
            continue
        if args.require_complete:
            validate_complete(prediction_paths, split_rows)
        for path in prediction_paths:
            outcomes.extend(evaluate_file(
                path, split_rows, Path(args.dataset_root), args.threshold, args.identity_threshold,
                aliases, args.identity_alignment, soft_pairs, args.obb_tie_margin,
            ))
        if not outcomes:
            continue
        grouped = defaultdict(list)
        for item in outcomes:
            grouped[item["group"]].append(item["strict_top1"])
        row = {"Variant": variant_dir.name}
        for group in ORDER:
            values = grouped[group]
            row[group] = 100 * sum(values) / len(values) if values else None
        row["StrictTop1"] = 100 * sum(item["strict_top1"] for item in outcomes) / len(outcomes)
        row["Recall5"] = 100 * sum(item["recall_at_5"] for item in outcomes) / len(outcomes)
        row["SR1m"] = 100 * sum(item["sr_at_1m"] for item in outcomes) / len(outcomes)
        row["GeoSR1m"] = 100 * sum(item["geometric_sr_at_1m"] for item in outcomes) / len(outcomes)
        row["Category"] = 100 * sum(item["category_consistent"] for item in outcomes) / len(outcomes)
        row["Avg"] = row["StrictTop1"]
        row["Queries"] = len(outcomes)
        summaries.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ("Variant", *ORDER, "StrictTop1", "Recall5", "SR1m", "GeoSR1m", "Category", "Avg", "Queries")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)
    latex = [" & ".join(columns) + r" \\", r"\midrule"]
    for row in summaries:
        metric_columns = (*ORDER, "StrictTop1", "Recall5", "SR1m", "GeoSR1m", "Category", "Avg")
        values = [row["Variant"]] + ["--" if row[key] is None else f"{row[key]:.1f}" for key in metric_columns] + [str(row["Queries"])]
        latex.append(" & ".join(values) + r" \\")
    output.with_suffix(".tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(summaries, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
