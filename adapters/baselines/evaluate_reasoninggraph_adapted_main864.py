#!/usr/bin/env python3
"""Language-only benchmark adapter for ReasoningGraph.

The adapter never reads split annotations other than query_text and identifiers.
Candidate generation uses the saved ReasoningGraph CLIP features; room membership,
hierarchy, and visual relationship tokens all come from its frozen stride-1 DSG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np

from additional_baselines_common import (
    DEFAULT_OUTPUT_ROOT, DEFAULT_SPLIT, OpenAICompatibleClient, atomic_json,
    load_env_file, load_json, load_split_rows, prediction_record, rows_for_scene,
)
from evaluate_reasoninggraph_main864 import (
    RG_INDEX_ROOT, TextEncoder, normalize_rows, reasoninggraph_to_hm3d,
)


ROOT = Path(__file__).resolve().parents[2]
PARSER_SYSTEM = """Parse a free-form object-localization request. Return JSON only.
Use this schema:
{"target":{"name":"object noun phrase","attributes":[],"not_attributes":[]},
 "room":{"name":"room phrase or empty","negated":false},
 "floor":null,
 "anchors":{"a1":{"name":"object noun phrase","attributes":[],"not_attributes":[]}},
 "relations":[{"subject":"target","predicate":"near|on|under|above|inside|left of|right of|in front of|behind","object":"a1","negated":false}]}

Rules:
- Use only information stated in the request. Do not use outside scene knowledge.
- The target is the object the request asks to find, never an anchor.
- Keep attributes (color, material, descriptive modifiers) separate from the noun.
- Represent relation chains with a1, a2, ... and multiple relation records.
- "not near any X" is a negated relation. "X not made of wood" is a
  not_attributes entry on X, not a negated relation.
- room.name excludes words such as "in", "not", and "floor".
- floor is a zero-based integer only when explicitly stated; otherwise null.
- Empty anchors and relations are valid.
"""

DECISOR_SYSTEM = """Judge whether a visual-language response supports a proposed
spatial relationship. Return only Yes or No. Answer Yes only when the response
clearly supports the proposed directed relation; uncertainty or contradiction is No."""

ALLOWED_RELATIONS = {
    "near", "on", "under", "above", "inside", "left of", "right of", "in front of", "behind"
}
RELATION_ALIASES = {
    "close to": "near", "next to": "near", "beside": "near",
    "below": "under", "over": "above", "atop": "on",
    "in": "inside", "inside of": "inside",
    "left": "left of", "right": "right of", "front of": "in front of",
}
PREFILTER_K = 20
VLM_MAX_NEW_TOKENS = 256
FLOOR_GAP_M = 1.5


def clean_entity(value, default=""):
    if not isinstance(value, dict):
        value = {}
    return {
        "name": str(value.get("name") or default).strip().lower(),
        "attributes": [str(x).strip().lower() for x in value.get("attributes", []) if str(x).strip()],
        "not_attributes": [str(x).strip().lower() for x in value.get("not_attributes", []) if str(x).strip()],
    }


def validate_parse(value):
    target = clean_entity(value.get("target"))
    if not target["name"]:
        raise ValueError("adapted parser returned an empty target")
    anchors = {
        str(key): clean_entity(item)
        for key, item in (value.get("anchors") or {}).items()
        if str(key) != "target" and clean_entity(item)["name"]
    }
    relations = []
    nodes = {"target", *anchors}
    for item in value.get("relations") or []:
        subject, obj = str(item.get("subject")), str(item.get("object"))
        predicate = " ".join(str(item.get("predicate") or "").lower().split())
        predicate = RELATION_ALIASES.get(predicate, predicate)
        if subject not in nodes or obj not in nodes or subject == obj:
            raise ValueError(f"invalid adapted relation endpoints: {subject}, {obj}")
        if predicate not in ALLOWED_RELATIONS:
            raise ValueError(f"unsupported adapted relation: {predicate}")
        relations.append({"subject": subject, "predicate": predicate, "object": obj,
                          "negated": bool(item.get("negated", False))})
    room_value = value.get("room") or {}
    room = {"name": str(room_value.get("name") or "").strip().lower(),
            "negated": bool(room_value.get("negated", False))}
    floor = value.get("floor")
    floor = None if floor is None or floor == "" else int(floor)
    if floor is not None and floor < 0:
        raise ValueError("floor must be nonnegative")
    return {"target": target, "room": room, "floor": floor,
            "anchors": anchors, "relations": relations}


def normalize_relation_endpoints(value):
    """Materialize relation endpoint names that the parser omitted from anchors."""
    normalized = dict(value)
    target = clean_entity(normalized.get("target"))
    anchors = {
        str(key): clean_entity(item)
        for key, item in (normalized.get("anchors") or {}).items()
        if str(key) != "target" and clean_entity(item)["name"]
    }
    name_to_key = {item["name"]: key for key, item in anchors.items()}
    relations = []
    audit = []
    for raw in normalized.get("relations") or []:
        item = dict(raw)
        for field in ("subject", "object"):
            endpoint = str(item.get(field) or "").strip().lower()
            if endpoint in {"target", *anchors}:
                continue
            if endpoint and endpoint == target["name"]:
                item[field] = "target"
                audit.append({"endpoint": endpoint, "normalized_to": "target"})
                continue
            if endpoint:
                key = name_to_key.get(endpoint)
                if key is None:
                    key = f"a{len(anchors) + 1}"
                    anchors[key] = clean_entity({"name": endpoint})
                    name_to_key[endpoint] = key
                item[field] = key
                audit.append({"endpoint": endpoint, "normalized_to": key})
        relations.append(item)
    normalized["anchors"] = anchors
    normalized["relations"] = relations
    return normalized, audit


def parse_query(client, query_text):
    model = os.environ.get("REASONINGGRAPH_TASK_MODEL", "gpt-4o")
    parsed, api = client.chat_json(
        model=model, system=PARSER_SYSTEM, prompt=f"Request: {query_text}", max_tokens=700,
        cache_namespace="reasoninggraph_benchmark_adapted_parser_v1",
    )
    candidate, normalization_audit = normalize_relation_endpoints(parsed)
    metadata = dict(api)
    if normalization_audit:
        metadata["deterministic_endpoint_normalization"] = normalization_audit
    for repair_index in range(3):
        try:
            return validate_parse(candidate), metadata
        except (TypeError, ValueError) as error:
            repaired, repair_api = client.chat_json(
                model=model, system=PARSER_SYSTEM,
                prompt=(f"Request: {query_text}\nInvalid proposed JSON: {json.dumps(candidate)}\n"
                        f"Validation error: {error}\nReturn a corrected JSON object in the exact schema."),
                max_tokens=700,
                cache_namespace="reasoninggraph_benchmark_adapted_parser_repair_v1",
            )
            metadata["repair" if repair_index == 0 else f"repair_{repair_index + 1}"] = repair_api
            candidate, repair_normalization = normalize_relation_endpoints(repaired)
            if repair_normalization:
                metadata.setdefault("deterministic_endpoint_normalization", []).extend(
                    repair_normalization
                )
    return validate_parse(candidate), metadata


def entity_phrase(entity):
    return " ".join([*entity["attributes"], entity["name"]]).strip()


def entity_scores(entity, encoder, object_features):
    texts = [entity_phrase(entity), *[f"{x} {entity['name']}" for x in entity["not_attributes"]]]
    encoded = encoder.encode(texts)
    score = object_features @ encoded[0]
    if len(encoded) > 1:
        score = score - np.max(object_features @ encoded[1:].T, axis=1)
    return score


def native_pool(scores, valid_mask, limit=PREFILTER_K):
    valid = np.flatnonzero(valid_mask)
    if not len(valid):
        return []
    raw = scores[valid]
    lo, hi = float(raw.min()), float(raw.max())
    if hi == lo:
        return []
    keep = valid[(raw - lo) / (hi - lo) > 0.5]
    return sorted(map(int, keep), key=lambda i: (-float(scores[i]), i))[:limit]


def object_room_lookup(data):
    room_ids = set(map(int, data["room_ids"]))
    adjacency = {}
    for a, b in zip(data["hierarchy_source"], data["hierarchy_target"]):
        a, b = int(a), int(b)
        adjacency.setdefault(a, set()).add(b); adjacency.setdefault(b, set()).add(a)
    result = {}
    for object_id in map(int, data["object_ids"]):
        frontier, seen = [object_id], {object_id}
        found = None
        for _ in range(4):
            following = []
            for node in frontier:
                if node in room_ids: found = node; break
                for nxt in adjacency.get(node, ()):
                    if nxt not in seen: seen.add(nxt); following.append(nxt)
            if found is not None: break
            frontier = following
        result[object_id] = found
    return result


def select_room(room_text, encoder, data):
    if not room_text or not len(data["room_ids"]):
        return None, []
    query = encoder.encode([room_text])[0]
    scores = []
    for i, count in enumerate(data["room_feature_counts"]):
        features = normalize_rows(np.asarray(data["room_features"][i, :int(count)], dtype=np.float32))
        scores.append(float(np.max(features @ query)) if len(features) else -np.inf)
    best = int(np.argmax(scores))
    return int(data["room_ids"][best]), scores


def floor_by_object(data, room_by_object):
    positions = np.asarray([reasoninggraph_to_hm3d(x) for x in data["object_positions"]])
    ids = list(map(int, data["object_ids"])); by_room = {}
    for i, oid in enumerate(ids):
        if room_by_object.get(oid) is not None:
            by_room.setdefault(room_by_object[oid], []).append(float(positions[i, 1]))
    heights = sorted((float(np.median(v)), room) for room, v in by_room.items())
    room_floor = {}; floor = 0; previous = None
    for height, room in heights:
        if previous is not None and height - previous > FLOOR_GAP_M: floor += 1
        room_floor[room] = floor; previous = height
    return {oid: room_floor.get(room_by_object.get(oid)) for oid in ids}


class RelationshipJudge:
    def __init__(self, index_path, data, client, cache_root):
        self.index_path, self.client = index_path, client
        self.cache_root = Path(cache_root) / "reasoninggraph_benchmark_adapted_vlm_v1"
        self.lookup = {}
        for i, (a, b) in enumerate(zip(data["edge_source"], data["edge_target"])):
            self.lookup[(int(a), int(b))] = (i, "source_target")
            self.lookup[(int(b), int(a))] = (i, "target_source")
        self.calls = []

    def supports(self, source_id, target_id, source_name, predicate, target_name):
        edge = self.lookup.get((source_id, target_id))
        if edge is None:
            return False, "no_graph_edge"
        edge_index, direction = edge
        prompt = (f"Does the {source_name} have the spatial relation '{predicate}' to the "
                  f"{target_name}? Explain briefly using only the depicted objects.")
        digest = hashlib.sha256(json.dumps({"index": str(self.index_path), "edge": edge_index,
            "direction": direction, "prompt": prompt, "max": VLM_MAX_NEW_TOKENS},
            sort_keys=True).encode()).hexdigest()
        path = self.cache_root / f"{digest}.json"; cached = load_json(path)
        if cached is None:
            body = json.dumps({"index_path": str(self.index_path.resolve()), "edge_index": edge_index,
                "direction": direction, "prompt": prompt,
                "max_new_tokens": VLM_MAX_NEW_TOKENS}).encode()
            endpoint = os.environ.get("REASONINGGRAPH_VLM_URL", "http://127.0.0.1:8765")
            request = urllib.request.Request(endpoint.rstrip("/") + "/generate",
                data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=3600) as response:
                text = str(json.loads(response.read().decode())["response"])
            atomic_json(path, {"response": text})
        else:
            text = str(cached["response"])
        decision, api = self.client.chat_text(model=os.environ.get("REASONINGGRAPH_DECISOR_MODEL", "gpt-4o"),
            system=DECISOR_SYSTEM,
            prompt=f"Proposed relation: {source_name} {predicate} {target_name}\nVLM response: {text}",
            max_tokens=20, cache_namespace="reasoninggraph_benchmark_adapted_decisor_v1")
        accepted = decision.strip().lower().startswith("yes")
        self.calls.append({"edge_index": edge_index, "direction": direction, "predicate": predicate,
                           "accepted": accepted, "decisor_api": api, "vlm_cache_key": digest})
        return accepted, "vlm"


def relation_filter(parsed, pools, ids, names, judge):
    relations = parsed["relations"]
    if not relations:
        return pools["target"]
    survivors = []
    def search(assignments, remaining):
        if not remaining: return True
        for ri, rel in enumerate(remaining):
            s, o = rel["subject"], rel["object"]
            if s in assignments or o in assignments:
                break
        else: return False
        options = [(s, assignments[s], o, pools[o])] if s in assignments else [(o, assignments[o], s, pools[s])]
        known_name, known_index, new_name, candidates = options[0]
        for candidate in candidates:
            if candidate in assignments.values(): continue
            trial = dict(assignments); trial[new_name] = candidate
            si, oi = trial[s], trial[o]
            ok, _ = judge.supports(ids[si], ids[oi], names[si], rel["predicate"], names[oi])
            if rel["negated"]: ok = not ok
            if ok and search(trial, remaining[:ri] + remaining[ri + 1:]): return True
        return False
    for target in pools["target"]:
        if search({"target": target}, relations): survivors.append(target)
    return survivors


def evaluate_scene(scene_id, rows, output_root, client, encoder, method, max_queries=None):
    index_path = RG_INDEX_ROOT / f"{scene_id}.npz"; output = Path(output_root) / method / f"{scene_id}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True); observed = set()
    if output.exists(): observed = {json.loads(x)["key"] for x in output.read_text().splitlines() if x.strip()}
    pending = [r for r in rows_for_scene(rows, scene_id) if r["key"] not in observed]
    if max_queries is not None: pending = pending[:max_queries]
    if not pending: return 0
    with np.load(index_path, allow_pickle=False) as data, output.open("a") as handle:
        features = normalize_rows(np.asarray(data["object_features"], dtype=np.float32)); valid = np.asarray(data["object_mesh_vertices"] >= 10)
        ids = list(map(int, data["object_ids"])); names = list(map(str, data["object_names"])); room_of = object_room_lookup(data); floor_of = floor_by_object(data, room_of)
        for row in pending:
            started = time.perf_counter(); parsed, api = parse_query(client, row["query_text"])
            entities = {"target": parsed["target"], **parsed["anchors"]}; scores = {k: entity_scores(v, encoder, features) for k, v in entities.items()}; pools = {k: native_pool(scores[k], valid) for k in entities}
            chosen_room, room_scores = select_room(parsed["room"]["name"], encoder, data)
            if chosen_room is not None:
                neg = parsed["room"]["negated"]
                pools["target"] = [i for i in pools["target"] if (room_of.get(ids[i]) != chosen_room) if neg] if neg else [i for i in pools["target"] if room_of.get(ids[i]) == chosen_room]
            if parsed["floor"] is not None:
                pools["target"] = [i for i in pools["target"] if floor_of.get(ids[i]) == parsed["floor"]]
            judge = RelationshipJudge(index_path, data, client, Path(output_root) / "api_cache")
            order = relation_filter(parsed, pools, ids, names, judge)
            candidates = [{"id": ids[i], "label": names[i], "position": reasoninggraph_to_hm3d(data["object_positions"][i]).tolist(), "score": float(scores["target"][i])} for i in order[:5]]
            record = prediction_record(row, method, candidates, parsed_query=parsed, parser_api=api,
                relationship_audit=judge.calls, latency_s=round(time.perf_counter()-started,3),
                method_metadata={"input_policy":"query_text_only", "object_encoder":"OpenAI CLIP ViT-L/14", "candidate_ranking":"target CLIP similarity", "prefilter_k":PREFILTER_K, "vlm":"DeepSeek-VL2 Visual", "vlm_max_new_tokens":VLM_MAX_NEW_TOKENS, "room_filter":"native room CLIP features", "floor_filter":f"unsupervised height gaps > {FLOOR_GAP_M}m"})
            handle.write(json.dumps(record, ensure_ascii=True)+"\n"); handle.flush()
    return len(pending)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--scene-id"); p.add_argument("--query-key"); p.add_argument("--split",type=Path,default=DEFAULT_SPLIT); p.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT_ROOT); p.add_argument("--method-name",default="reasoninggraph_benchmark_adapted_stride1_area_free"); p.add_argument("--device",default="cuda"); p.add_argument("--scene-shard-index",type=int,default=0); p.add_argument("--scene-shard-count",type=int,default=1); p.add_argument("--max-queries-per-scene",type=int); p.add_argument("--confirm-formal-evaluation",action="store_true"); args=p.parse_args()
    if not args.confirm_formal_evaluation: raise SystemExit("Refusing to evaluate without --confirm-formal-evaluation")
    load_env_file(); rows=load_split_rows(args.split)
    if args.query_key:
        rows=[row for row in rows if row["key"] == args.query_key]
        if len(rows) != 1: raise SystemExit(f"query key not found or ambiguous: {args.query_key}")
        args.scene_id=rows[0]["scene_id"]
    scenes=[args.scene_id] if args.scene_id else sorted({x["scene_id"] for x in rows})[args.scene_shard_index::args.scene_shard_count]; client=OpenAICompatibleClient(args.output_root/"api_cache"); encoder=TextEncoder(args.device); total=0
    for scene in scenes:
        n=evaluate_scene(scene,rows,args.output_root,client,encoder,args.method_name,args.max_queries_per_scene); total+=n; print(json.dumps({"method":args.method_name,"scene":scene,"new_records":n}),flush=True)
    print(json.dumps({"method":args.method_name,"new_records":total,"status":"complete"}),flush=True)


if __name__ == "__main__": main()
