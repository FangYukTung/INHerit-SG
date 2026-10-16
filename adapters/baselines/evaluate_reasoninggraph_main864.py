#!/usr/bin/env python3
"""Offline Main-864 adapter for the native ReasoningGraph query stages."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import clip
import numpy as np
import torch

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
RG_MAP_ROOT = Path(os.environ.get(
    "REASONINGGRAPH_MAP_ROOT",
    REPO_ROOT / "outputs/additional_baselines/reasoninggraph/main864_mapping_stride1",
))
RG_INDEX_ROOT = Path(os.environ.get(
    "REASONINGGRAPH_INDEX_ROOT",
    REPO_ROOT / "outputs/additional_baselines/reasoninggraph/query_indexes_stride1",
))
CLIP_CHECKPOINT = Path(os.environ.get("REASONINGGRAPH_CLIP_CHECKPOINT", "weights/ViT-L-14.pt"))


TASK_SYSTEM = """You will be given a natural language prompt describing a visual or spatial task.

Your goal is to reason about the task in two steps:
1. Identify relevant objects required to fulfill the task. Only select object names from the provided list.
2. Determine whether the task requires reasoning about how objects physically interact — such as moving, placing, storing, supporting, or using one object with another. Be realistic about these interactions and which objects can be moved and which cannot.

Your output must be in **JSON format** and include:
- "objects": a list of relevant object names (only from the provided list)
- "interactions": a dictionary of reasoning prompts about object interactions, if applicable

Include "interactions" **if the task involves or implies**:
- moving, placing, lifting, or removing objects,
- storing items or determining suitable locations for them,
- determining whether one object can support, contain, or obstruct another,
- preparing a space by reasoning about spatial arrangements (e.g., clearing a surface or floor)

When generating an interaction, only include object pairs where the second object enables or supports a meaningful physical interaction (e.g., placing one object on another, moving it out of the way, or using it to store/hide/support something). Avoid pairing objects arbitrarily. Do not generate an interaction if the second object plays no actual role in the scenario.

Leave "interactions" empty only for tasks that **purely** ask to:
- find, detect, or locate objects,
- identify where objects might typically be found (without concern for feasibility),
- classify or list relevant items

Only use object names from this list:"""

TASK_EXAMPLES = """Here are a few examples:
Example 1:
Prompt: "Prepare the office room to clean its floor."
Response:
'{"objects": ["trashcan", "chair", "bag", "desk"],
 "interactions": {"1": {"objects": ["trashcan", "desk"], "prompt": "I would like to mop the room's floor of the image. Consider the objects in the image and their physical properties. Is the chair obstructing the floor? If yes, is it physically possible to momentarily put the trashcan on the desk so that I can mop the floor? Explain your reasoning."}, "2": {"objects": ["chair", "desk"], "prompt": "I would like to mop the room's floor in the image. Consider the objects in the image and their physical properties. Is the chair obstructing the floor? If yes, is it physically possible to momentarily put the chair on the desk so that I can mop the floor? Explain your reasoning."}, "3": {"objects": ["bag", "desk"], "prompt": "I would like to mop the room's floor in the image. Consider the objects in the image and their physical properties.  Is the bag obstructing the floor? If yes, is it physically possible to momentarily put the bag on the desk so that I can mop the floor? Explain your reasoning."}, "4": {"objects": ["bag", "chair"], "prompt": "I would like to mop the room's floor in the image. Consider the objects in the image and their physical properties. Is the bag obstructing the floor? If yes, is it physically possible to momentarily put the bag on the chair so that I can mop the floor? Explain your reasoning."}}}'
Example 2:
Prompt: "Find a place to sit down."
Response:
'{"objects": ["sofa", "chair", "bed"],
 "interactions": {}}'
Example 3:
Prompt: "Save the food for later."
Response:
'{"objects": ["food", "fridge", "freezer"],
 "interactions": {"1": {"objects": ["food", "fridge"], "prompt": "I would like to save the food to eat it later. Consider the objects in the image and their physical properties. Is the food in a good storage place? If not, is it possible to store the food in the fridge? Explain your reasoning."}, "2": {"objects": ["food", "freezer"], "prompt": "I would like to save the food to eat it later. Consider the objects in the image and their physical properties. Is the food in a good storage place? If not, is it possible to store the food in the freezer? Explain your reasoning."}}}'
Example 4:
Prompt: "Your task is to locate places where there could be water"
'{"objects": ["water", "sink", "bottle", "refrigerator", "shower", "lake", "pier", "waterfall", "swimming pool", "fountain", "river", "bathtub", "sea"],
"interactions": {}}'"""

# This is the exact object-label subset appended by the official
# navigation_prompt_parser.launch through anymal_label_space.yaml.
ANYMAL_OBJECT_LABELS = (
    "tree, bed, windowpane, door, plant, curtain, chair, water, painting, sofa, "
    "mirror, fence, desk, rock, bathtub, base, sink, fireplace, refrigerator, case, "
    "pillow, toilet, book, stove, computer, towel, light, chandelier, streetlight, "
    "booth, television receiver, pole, ottoman, bottle, poster, stool, barrel, basket, "
    "oven, ball, food, microwave, pot, dishwasher, screen, trashcan, fan, plate, "
    "monitor, shower, radiator, glass, clock, flag, blanket, backpack"
)


def task_system(vocabulary_mode: str) -> str:
    if vocabulary_mode == "native_anymal":
        # Exact concatenation performed by upstream OpenAIPrompter: system
        # prompt, AnyMal label list, then examples.txt.
        return TASK_SYSTEM + "\n[" + ANYMAL_OBJECT_LABELS + "]\n\n" + TASK_EXAMPLES
    raise ValueError("The formal native ReasoningGraph protocol only supports native_anymal")

DECISOR_SYSTEM = """You are a decision-making assistant.
Your role is to determine whether a task should be executed based on the output of a Vision-Language Model (VLM). You will be given two inputs:

VLM's prompt – a question or instruction given to the VLM, often asking whether a task is necessary or feasible.

VLM's response – the VLM's answer to that prompt, possibly including reasoning or evaluation.

Your job:
Carefully analyze the VLM's response to decide if the task should be performed. Consider whether:

The task is clearly needed or justified.

It is feasible according to the VLM.

There is sufficient context or certainty to proceed.

Respond only with:

"Yes" — if the task should be executed, based on a reasonable interpretation of the VLM’s response.

"No" — if the task should not be executed, due to lack of necessity, feasibility, or clarity."""


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def reasoninggraph_to_hm3d(position: np.ndarray) -> np.ndarray:
    """Convert the official Hydra ROS world frame back to HM3D/Habitat.

    The upstream ``publish_hm3dsem.py`` applies two fixed left-handed basis
    changes to every Habitat camera pose.  Their combined translation mapping
    is ``(x_h, y_h, z_h) -> (z_h, x_h, y_h)``.  Hydra's DSG object positions
    therefore use that same ROS world frame, while the benchmark annotations
    retain Habitat coordinates.  This inverse permutation is fixed entirely by
    the official publisher and is not fitted against benchmark objects.
    """
    position = np.asarray(position, dtype=float)
    if position.shape != (3,):
        raise ValueError(f"Expected a 3D ReasoningGraph position, got {position.shape}")
    return position[[1, 2, 0]]


class TextEncoder:
    def __init__(self, device: str):
        self.device = torch.device(device)
        self.model, _ = clip.load(str(CLIP_CHECKPOINT), device=self.device, jit=False)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> np.ndarray:
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        features = self.model.encode_text(tokens).float()
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.cpu().numpy()


def parse_task(client: OpenAICompatibleClient, query_text: str, vocabulary_mode: str):
    model = os.environ.get("REASONINGGRAPH_TASK_MODEL", "gpt-4o")
    parsed, metadata = client.chat_json(
        model=model,
        system=task_system(vocabulary_mode),
        prompt=f"Task: {query_text}",
        max_tokens=900,
        cache_namespace="reasoninggraph_official_task_parser_gpt4o_v1",
    )
    allowed = {value.strip().lower() for value in ANYMAL_OBJECT_LABELS.split(",")}
    # Upstream trusts the model's `objects` array after instructing it to use
    # the provided AnyMal labels. Preserve order and do not invent a target.
    objects = [str(value).strip().lower() for value in parsed.get("objects", []) if str(value).strip()]
    interactions = parsed.get("interactions") or {}
    if isinstance(interactions, dict):
        interactions = list(interactions.values())
    elif not isinstance(interactions, list):
        interactions = []
    # Exact upstream validation in OpenAIPrompter.generate().
    accepted_interactions = []
    rejected_interactions = []
    for interaction in interactions:
        pair = interaction.get("objects") or []
        prompt = str(interaction.get("prompt") or "")
        if len(pair) != 2:
            rejected_interactions.append({"interaction": interaction, "reason": "pair_length"})
            continue
        pair = [str(value).strip().lower() for value in pair]
        if pair[0] not in prompt.lower() or pair[1] not in prompt.lower():
            rejected_interactions.append({"interaction": interaction, "reason": "names_absent_from_prompt"})
            continue
        pair = ["desk" if value == "table" else value for value in pair]
        if pair[0] not in allowed or pair[1] not in allowed:
            rejected_interactions.append({"interaction": interaction, "reason": "outside_anymal_labels"})
            continue
        accepted_interactions.append({"objects": pair, "prompt": prompt})
    result = {"objects": objects, "interactions": accepted_interactions}
    if rejected_interactions:
        result["official_parser_rejections"] = rejected_interactions
    return result, metadata


def request_vlm(index_path: Path, edge_index: int, direction: str, prompt: str) -> str:
    endpoint = os.environ.get("REASONINGGRAPH_VLM_URL")
    if not endpoint:
        raise RuntimeError(
            "ReasoningGraph produced a physical interaction but REASONINGGRAPH_VLM_URL is not set"
        )
    body = json.dumps(
        {"index_path": str(index_path.resolve()), "edge_index": edge_index, "direction": direction, "prompt": prompt}
    ).encode("utf-8")
    request = urllib.request.Request(endpoint.rstrip("/") + "/generate", data=body, headers={"Content-Type": "application/json"})
    last_error = None
    for _ in range(120):
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                return str(json.loads(response.read().decode("utf-8"))["response"])
        except (urllib.error.URLError, ConnectionError) as error:
            last_error = error
            time.sleep(15)
    raise RuntimeError(f"DeepSeek-VL2 endpoint did not become ready: {last_error}")


def index_is_current(scene_id: str) -> bool:
    index_path = RG_INDEX_ROOT / f"{scene_id}.npz"
    source = RG_MAP_ROOT / scene_id / "hm3dsem/backend/dsg.json"
    if not index_path.exists() or not source.exists():
        return False
    try:
        with np.load(index_path, allow_pickle=False) as data:
            return (
                len(data["object_ids"]) > 0
                and "object_mesh_vertices" in data.files
                and "agent_position" in data.files
                and int(data["source_size"][0]) == source.stat().st_size
                and int(data["source_mtime_ns"][0]) == source.stat().st_mtime_ns
            )
    except Exception:
        return False


def wait_for_index(scene_id: str, poll_seconds: int) -> None:
    while not index_is_current(scene_id):
        print(json.dumps({"scene": scene_id, "status": "waiting_for_index"}), flush=True)
        time.sleep(poll_seconds)


def native_normalized_scores(scores: np.ndarray) -> np.ndarray:
    """Match CosSimSearch::normalize from the upstream C++ implementation."""
    scores = np.asarray(scores, dtype=np.float32)
    if not len(scores):
        return scores
    minimum, maximum = float(scores.min()), float(scores.max())
    if maximum == minimum:
        return np.full_like(scores, np.nan)
    return (scores - minimum) / (maximum - minimum)


def native_object_matches(
    object_features: np.ndarray,
    text_feature: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the HM3DSem default native object search threshold."""
    valid_indices = np.flatnonzero(valid_mask)
    raw = object_features[valid_indices] @ text_feature
    normalized = native_normalized_scores(raw)
    return valid_indices[normalized > threshold], raw


def decide_interaction(client: OpenAICompatibleClient, prompt: str, response: str):
    model = os.environ.get("REASONINGGRAPH_DECISOR_MODEL", "gpt-4o")
    decision, metadata = client.chat_text(
        model=model,
        system=DECISOR_SYSTEM,
        prompt=f"\n VLM's prompt: {prompt}\n VLM's response: {response}\n",
        max_tokens=80,
        cache_namespace="reasoninggraph_official_subtask_decisor_gpt4o_v1",
    )
    # This is the literal upstream decision rule.
    return "yes" in decision.lower(), metadata


def interaction_filter(index_path: Path, data, parsed, object_features, valid_mask, encoder, client):
    interactions = parsed["interactions"]
    if not interactions:
        return None, []
    ids = data["object_ids"]
    edge_lookup = {}
    for edge_index, (source, target) in enumerate(zip(data["edge_source"], data["edge_target"])):
        edge_lookup[(int(source), int(target))] = (edge_index, "source_target")
        edge_lookup[(int(target), int(source))] = (edge_index, "target_source")
    accepted = set()
    audit = []
    for interaction in interactions:
        pair = interaction.get("objects") or []
        if len(pair) != 2:
            continue
        target_text, subject_text = map(str, pair)
        if target_text not in parsed["objects"] or subject_text not in parsed["objects"]:
            audit.append({"prompt": interaction.get("prompt"), "accepted": False,
                          "reason": "official_pair_label_missing_from_objects"})
            continue
        text_features = encoder.encode([target_text, subject_text])
        target_order, _ = native_object_matches(object_features, text_features[0], valid_mask)
        subject_order, _ = native_object_matches(object_features, text_features[1], valid_mask)
        for target_index in target_order:
            for subject_index in subject_order:
                key = (int(ids[target_index]), int(ids[subject_index]))
                if key not in edge_lookup:
                    continue
                edge_index, direction = edge_lookup[key]
                prompt = str(interaction.get("prompt") or f"Is the {target_text} related to the {subject_text} as requested?")
                vlm_response = request_vlm(index_path, edge_index, direction, prompt)
                accepted_pair, api = decide_interaction(client, prompt, vlm_response)
                audit.append(
                    {"edge_index": edge_index, "direction": direction, "prompt": prompt, "response": vlm_response, "accepted": accepted_pair, "decisor_api": api}
                )
                if accepted_pair:
                    accepted.add(int(target_index))
    if not accepted:
        return np.asarray([], dtype=np.int64), audit
    agent_position = np.asarray(data["agent_position"], dtype=np.float32)
    positions = np.asarray(data["object_positions"], dtype=np.float32)
    return np.asarray(
        sorted(accepted, key=lambda index: float(np.linalg.norm(positions[index] - agent_position))),
        dtype=np.int64,
    ), audit


def evaluate_scene(
    scene_id: str, rows, output_root: Path, client, encoder, top_k: int,
    vocabulary_mode: str, method: str, frame_stride: int, room_mode: str,
):
    index_path = RG_INDEX_ROOT / f"{scene_id}.npz"
    if not index_path.exists():
        raise FileNotFoundError(f"ReasoningGraph query index missing: {index_path}")
    output = output_root / method / f"{scene_id}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    observed = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                observed.add(json.loads(line)["key"])
    pending = [row for row in rows_for_scene(rows, scene_id) if row["key"] not in observed]
    if not pending:
        return 0
    with np.load(index_path, allow_pickle=False) as data, output.open("a", encoding="utf-8") as handle:
        object_features = normalize_rows(data["object_features"])
        valid_mask = np.asarray(data["object_mesh_vertices"] >= 10, dtype=bool)
        agent_position = np.asarray(data["agent_position"], dtype=np.float32)
        positions = np.asarray(data["object_positions"], dtype=np.float32)
        for row in pending:
            started = time.perf_counter()
            parsed, task_api = parse_task(client, row["query_text"], vocabulary_mode)
            text_features = encoder.encode(parsed["objects"]) if parsed["objects"] else np.empty((0, object_features.shape[1]))
            raw_score_by_index = np.full(len(object_features), -np.inf, dtype=np.float32)
            native_order = []
            for text_feature in text_features:
                matches, _ = native_object_matches(object_features, text_feature, valid_mask)
                full_scores = object_features @ text_feature
                raw_score_by_index = np.maximum(raw_score_by_index, full_scores)
                native_order.extend(int(index) for index in matches)
            # Upstream uses std::sort by distance to the final agent pose after
            # concatenating every threshold match in parser-object order.
            native_order.sort(key=lambda index: float(np.linalg.norm(positions[index] - agent_position)))
            order = np.asarray(native_order, dtype=np.int64)
            filtered, relation_audit = interaction_filter(
                index_path, data, parsed, object_features, valid_mask, encoder, client
            )
            if filtered is not None:
                order = filtered
            candidates = [
                {
                    "id": int(data["object_ids"][index]),
                    "label": str(data["object_names"][index]),
                    "position": reasoninggraph_to_hm3d(data["object_positions"][index]).tolist(),
                    "score": float(raw_score_by_index[index]),
                }
                for index in order[:top_k]
            ]
            record = prediction_record(
                row,
                method,
                candidates,
                parsed_task=parsed,
                task_api=task_api,
                relationship_audit=relation_audit,
                latency_s=round(time.perf_counter() - started, 3),
                method_metadata={
                    "task_model": os.environ.get("REASONINGGRAPH_TASK_MODEL", "gpt-4o"),
                    "decisor_model": os.environ.get("REASONINGGRAPH_DECISOR_MODEL", "gpt-4o"),
                    "object_encoder": "OpenAI CLIP ViT-L/14",
                    "relationship_encoder": "DeepSeek-VL2 Visual",
                    "frame_stride": frame_stride,
                    "room_mode": room_mode,
                    "vocabulary_mode": vocabulary_mode,
                    "task_parser": "upstream gpt_system_prompt.txt + AnyMal labels + examples.txt",
                    "object_search": "upstream CosSimSearch min-max threshold=0.5, then agent-distance order",
                    "interaction_policy": "upstream parser output is executed without benchmark-specific suppression",
                    "coordinate_transform": (
                        "inverse of official publish_hm3dsem.py translation basis: "
                        "Hydra ROS (z_h,x_h,y_h) -> HM3D/Habitat (x_h,y_h,z_h)"
                    ),
                    "coordinate_transform_fitted_to_gt": False,
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vocabulary-mode", choices=("native_anymal",), default="native_anymal")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--room-mode", choices=("all",), default="all")
    parser.add_argument("--method-name")
    parser.add_argument("--scene-shard-index", type=int, default=0)
    parser.add_argument("--scene-shard-count", type=int, default=1)
    parser.add_argument("--wait-for-index", action="store_true")
    parser.add_argument("--index-poll-seconds", type=int, default=60)
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
    encoder = TextEncoder(args.device)
    method = args.method_name or f"reasoninggraph_{args.vocabulary_mode}_vocab"
    total = 0
    for scene_id in scenes:
        if args.wait_for_index:
            wait_for_index(scene_id, args.index_poll_seconds)
        count = evaluate_scene(
            scene_id, rows, args.output_root, client, encoder, args.top_k,
            args.vocabulary_mode, method, args.frame_stride, args.room_mode,
        )
        total += count
        print(json.dumps({"method": method, "scene": scene_id, "new_records": count}), flush=True)
    print(json.dumps({"method": method, "new_records": total, "status": "complete"}))


if __name__ == "__main__":
    main()
