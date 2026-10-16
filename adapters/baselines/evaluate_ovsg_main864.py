#!/usr/bin/env python3
"""Evaluate OVSG with its native ViT-B/32 + probabilistic graph kernel."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import open3d as o3d

from additional_baselines_common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SPLIT,
    OpenAICompatibleClient,
    load_env_file,
    load_split_rows,
    prediction_record,
    rows_for_scene,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OVSG_ROOT = Path(os.environ["OVSG_ROOT"])
OVSG_INPUT = Path(os.environ.get("OVSG_INPUT", REPO_ROOT / "outputs/additional_baselines/ovsg/input"))
_HYDRA_CONTEXT = None
sys.path.insert(0, str(OVSG_ROOT))

from ovsg.env.algo.notion import Domain, Feature, FeatureType, Space  # noqa: E402
from ovsg.env.notionovidb import NotionOVIDB  # noqa: E402
from ovsg.env.ovimap.vocabs import vocabs as OVSG_VOCABS  # noqa: E402
from ovsg.utils.spatial.spatial import Points9 as p9  # noqa: E402

SCANNET200_FEATURES = np.load(
    OVSG_ROOT / "external/OVIR-3D/src/scannet200.npy"
).astype(np.float32)
SCANNET200_FEATURES /= np.maximum(
    np.linalg.norm(SCANNET200_FEATURES, axis=1, keepdims=True), 1e-12
)
SCANNET200_LABELS = OVSG_VOCABS["scannet"]


OVSG_SYSTEM = """You translate a free-form localization request into OVSG query syntax.
Return JSON only: {"query_syntax":"..."}.
The first line must be: target @ NAME {object}.
Optional relation lines have exactly this form:
NAME1 {object|region|tag} -- RELATION [spatial|tag] -- NAME2 {object|region|tag}
Use [spatial] for near, on, above, under, left/right, front/behind, and in-room
relations. Use [tag] for an explicit attribute. Keep the target object category
as the target NAME. Use only facts in the request; never invent a relation.

Examples:
Request: Find a cushion on the sofa chair.
{"query_syntax":"target @ cushion {object}\\ncushion {object} -- on [spatial] -- sofa chair {object}"}
Request: Find a curtain in the bedroom.
{"query_syntax":"target @ curtain {object}\\ncurtain {object} -- in [spatial] -- bedroom {region}"}
Request: Find a hard object in the living room.
{"query_syntax":"target @ SOMETHING {object}\\nhard {tag} -- tag [tag] -- SOMETHING {object}\\nSOMETHING {object} -- in [spatial] -- living room {region}"}
"""

OVSG_REPAIR_SYSTEM = """You repair formatting of OVSG query syntax. Return JSON only as
{"query_syntax":"..."}. Preserve the request and every constraint in the proposed
syntax; do not invent or remove facts. The first line must match:
target @ NAME {object|region|tag}
Every remaining line must match:
NAME1 {object|region|tag} -- RELATION [spatial|tag] -- NAME2 {object|region|tag}
Put attributes such as fabric or brown in a {tag} notion, with `tag [tag]` as
the middle relation. Put spatial predicates in the middle with [spatial]."""

TARGET_LINE = re.compile(r"^target\s*@\s*.+\s+\{(?:object|region|tag)\}$")
RELATION_LINE = re.compile(
    r"^.+\s+\{(?:object|region|tag)\}\s*--\s*.+\s+\[(?:spatial|tag)\]\s*--\s*.+\s+\{(?:object|region|tag)\}$"
)
NATIVE_RELATION = re.compile(r"^\s*(\w.*\w)\s*\[\s*(\w.*\w)\s*\]\s*$")


def validate_query_syntax(syntax: str) -> None:
    lines = [line.strip() for line in syntax.strip().splitlines() if line.strip()]
    if not lines or not TARGET_LINE.fullmatch(lines[0]):
        raise ValueError(f"Invalid OVSG target line: {lines[0] if lines else '<empty>'}")
    for line in lines[1:]:
        if line.count("--") != 2 or not RELATION_LINE.fullmatch(line):
            raise ValueError(f"Invalid OVSG relation line: {line}")
        relation = line.split("--")[1].strip()
        if not NATIVE_RELATION.fullmatch(relation):
            raise ValueError(f"Relation rejected by native OVSG parser: {relation}")


def normalize_missing_relation_types(syntax: str) -> tuple[str, list[dict[str, str]]]:
    """Add an omitted relation type only when endpoint types determine it."""
    normalized = []
    audit = []
    for line in syntax.splitlines():
        parts = [part.strip() for part in line.split("--")]
        if len(parts) != 3 or re.search(r"\[(?:spatial|tag)\]", parts[1]):
            normalized.append(line.strip())
            continue
        endpoint_types = re.findall(r"\{(object|region|tag)\}", parts[0] + " " + parts[2])
        if len(endpoint_types) != 2:
            normalized.append(line.strip())
            continue
        relation_type = "tag" if "tag" in endpoint_types else "spatial"
        relation = parts[1]
        if relation_type == "tag" and relation.lower() in {"not", "negation"}:
            relation = "not tag"
        fixed = f"{parts[0]} -- {relation} [{relation_type}] -- {parts[2]}"
        audit.append({"original": line.strip(), "normalized": fixed})
        normalized.append(fixed)
    return "\n".join(normalized), audit


def normalize_missing_endpoint_types(syntax: str) -> tuple[str, list[dict[str, str]]]:
    """Restore an omitted endpoint type when a tag relation makes it unique."""
    normalized = []
    audit = []
    for line in syntax.splitlines():
        parts = [part.strip() for part in line.split("--")]
        if len(parts) != 3 or not re.search(r"\[tag\]", parts[1]):
            normalized.append(line.strip())
            continue
        left_type = re.search(r"\{(object|region|tag)\}\s*$", parts[0])
        right_type = re.search(r"\{(object|region|tag)\}\s*$", parts[2])
        fixed = None
        if left_type and not right_type:
            missing_type = "object" if left_type.group(1) == "tag" else "tag"
            fixed = f"{parts[0]} -- {parts[1]} -- {parts[2]} {{{missing_type}}}"
        elif right_type and not left_type:
            missing_type = "object" if right_type.group(1) == "tag" else "tag"
            fixed = f"{parts[0]} {{{missing_type}}} -- {parts[1]} -- {parts[2]}"
        if fixed is None:
            normalized.append(line.strip())
            continue
        audit.append({"original": line.strip(), "normalized": fixed})
        normalized.append(fixed)
    return "\n".join(normalized), audit


def compose_config():
    global _HYDRA_CONTEXT
    device = os.environ.get("OVSG_QUERY_DEVICE", "cuda")
    config_dir = str((OVSG_ROOT / "ovsg/config").resolve())
    if _HYDRA_CONTEXT is None:
        _HYDRA_CONTEXT = hydra.initialize_config_dir(version_base=None, config_dir=config_dir)
        _HYDRA_CONTEXT.__enter__()
    return hydra.compose(
        config_name="config",
        overrides=[
            "api=openai_demo",
            f"work_dir={OVSG_ROOT}",
            f"data_dir={OVSG_ROOT / 'ovsg/data'}",
            f"notion_dir={OVSG_ROOT / 'ovsg/data/notion'}",
            f"+ovi_exp_path={DEFAULT_OUTPUT_ROOT / 'ovsg_native/internal'}",
            f"notion_device={device}",
            f"prompt_device={device}",
            "gnn_padding=0",
        ],
    )


def build_native_graph(scene_id: str):
    prediction_path = (
        OVSG_INPUT / scene_id / "detic_output/imagenet21k-0.3/predictions/proposed_fusion_detic_native.pkl"
    )
    scan_path = OVSG_INPUT / scene_id / "scan-0.020.pcd"
    if not prediction_path.exists() or not scan_path.exists():
        raise FileNotFoundError(f"Incomplete OVSG map for {scene_id}")
    with prediction_path.open("rb") as handle:
        graph = pickle.load(handle)
    points = np.asarray(o3d.io.read_point_cloud(str(scan_path)).points)
    database = NotionOVIDB(compose_config())
    for instance_id, attrs in graph.nodes(data=True):
        indices = np.asarray(attrs["pt_indices"], dtype=np.int64)
        valid = indices[(indices >= 0) & (indices < len(points))]
        if not len(valid):
            continue
        instance_points = points[valid]
        minimum, maximum = instance_points.min(axis=0), instance_points.max(axis=0)
        address = p9.from_bbox(pos=(minimum + maximum) / 2.0, size=np.maximum(maximum - minimum, 1e-3))
        content = o3d.geometry.PointCloud()
        content.points = o3d.utility.Vector3dVector(
            np.asarray(
                [
                    [x, y, z]
                    for x in (minimum[0], maximum[0])
                    for y in (minimum[1], maximum[1])
                    for z in (minimum[2], maximum[2])
                ],
                dtype=np.float64,
            )
        )
        feature = np.asarray(attrs["feature"], dtype=np.float32)
        norm = np.linalg.norm(feature)
        if norm:
            feature = feature / norm
        # Existing custom-data maps were fused with scannet200.npy but their
        # indices were accidentally decoded with the LVIS label list.  The
        # fused feature is valid, so recover its name using the matching text
        # feature matrix instead of the corrupted stored top5_vocabs field.
        label_index = int(np.argmax(SCANNET200_FEATURES @ feature))
        name = str(SCANNET200_LABELS[label_index]).replace("_", " ")
        database.add_notion(
            keys=[Feature(feature, FeatureType.DETICIMG)],
            content=content,
            pos=address,
            domain=Domain.INS,
            name=name,
            space=Space.STATIC,
            instance_id=int(instance_id),
        )
    database.notion_graph.update()
    return database


def parse_query(client: OpenAICompatibleClient, text: str):
    model = os.environ.get("OVSG_LLM_MODEL", "gpt-3.5-turbo")
    parsed, metadata = client.chat_json(
        model=model,
        system=OVSG_SYSTEM,
        prompt=f"Request: {text}",
        max_tokens=500,
        cache_namespace="ovsg_query_parser",
    )
    syntax = str(parsed.get("query_syntax", "")).strip()
    repair_audit = []
    dropped_orphan_lines = []
    normalization_audit = []
    for repair_round in range(4):
        syntax, normalized = normalize_missing_endpoint_types(syntax)
        normalization_audit.extend(normalized)
        syntax, normalized = normalize_missing_relation_types(syntax)
        normalization_audit.extend(normalized)
        try:
            validate_query_syntax(syntax)
            break
        except ValueError as syntax_error:
            if repair_round == 3:
                lines = [line.strip() for line in syntax.splitlines() if line.strip()]
                kept = [lines[0], *[line for line in lines[1:] if "--" in line]] if lines else []
                dropped_orphan_lines = [line for line in lines[1:] if "--" not in line]
                syntax = "\n".join(kept)
                validate_query_syntax(syntax)
                break
            repaired, repair_metadata = client.chat_json(
                model=model,
                system=OVSG_REPAIR_SYSTEM,
                prompt=(
                    f"Request: {text}\nProposed syntax:\n{syntax}\n"
                    f"Formatting error: {syntax_error}\n"
                    "Rewrite every compound chain as separate lines. Each relation line must "
                    "contain exactly two `--` delimiters."
                ),
                max_tokens=500,
                cache_namespace=f"ovsg_query_syntax_repair_round_{repair_round + 1}",
            )
            repair_audit.append(repair_metadata)
            syntax = str(repaired.get("query_syntax", "")).strip()
    if repair_audit:
        metadata = {**metadata, "syntax_repairs": repair_audit}
    if dropped_orphan_lines:
        metadata = {**metadata, "dropped_orphan_lines": dropped_orphan_lines}
    if normalization_audit:
        metadata = {**metadata, "deterministic_syntax_normalizations": normalization_audit}
    return syntax, metadata


def evaluate_scene(
    scene_id: str, rows, output_root: Path, client: OpenAICompatibleClient,
    top_k: int, method_name: str,
):
    output = output_root / method_name / f"{scene_id}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    observed = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                observed.add(json.loads(line)["key"])
    pending = [row for row in rows_for_scene(rows, scene_id) if row["key"] not in observed]
    if not pending:
        return 0
    database = build_native_graph(scene_id)
    with output.open("a", encoding="utf-8") as handle:
        for row in pending:
            started = time.perf_counter()
            syntax, api = parse_query(client, row["query_text"])
            result = database.query(query_str=syntax, top_k=top_k, method="prob", verbose=False)
            candidates = []
            for rank, candidate in enumerate(result["candidates"][:top_k]):
                candidates.append(
                    {
                        "id": int(candidate.attributes.get("instance_id", candidate.id)),
                        "label": candidate.name,
                        "position": np.asarray(candidate.address[:3], dtype=float).tolist(),
                        "score": float(result["sprobs"][rank]) if rank < len(result["sprobs"]) else None,
                    }
                )
            record = prediction_record(
                row,
                method_name,
                candidates,
                query_syntax=syntax,
                parser_api=api,
                latency_s=round(time.perf_counter() - started, 3),
                method_metadata={
                    "query_model": os.environ.get("OVSG_LLM_MODEL", "gpt-3.5-turbo"),
                    "object_encoder": "OpenAI CLIP ViT-B/32 (native OVSG/OVIR map feature)",
                    "candidate_label_decoder": "ScanNet200 labels matched to scannet200.npy",
                    "graph_kernel": "prob",
                    "top_k": top_k,
                },
            )
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            handle.flush()
    return len(pending)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id")
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--method-name", default="ovsg_native")
    parser.add_argument("--scene-shard-index", type=int, default=0)
    parser.add_argument("--scene-shard-count", type=int, default=1)
    parser.add_argument("--confirm-formal-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.confirm_formal_evaluation:
        raise SystemExit("Refusing to evaluate without --confirm-formal-evaluation")
    load_env_file()
    rows = load_split_rows(args.split)
    scenes = [args.scene_id] if args.scene_id else sorted({row["scene_id"] for row in rows})
    if args.scene_shard_count < 1 or not 0 <= args.scene_shard_index < args.scene_shard_count:
        raise SystemExit("Invalid scene shard index/count")
    if not args.scene_id:
        scenes = scenes[args.scene_shard_index::args.scene_shard_count]
    client = OpenAICompatibleClient(args.output_root / "api_cache")
    total = 0
    for scene_id in scenes:
        count = evaluate_scene(
            scene_id, rows, args.output_root, client, args.top_k, args.method_name
        )
        total += count
        print(json.dumps({"method": args.method_name, "scene": scene_id, "new_records": count}), flush=True)
    print(json.dumps({"method": args.method_name, "new_records": total, "status": "complete"}))


if __name__ == "__main__":
    main()
