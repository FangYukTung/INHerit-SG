#!/usr/bin/env python3
"""Stream a multi-GB ReasoningGraph DSG into a query-efficient NPZ index."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np


OBJECT_PREFIX = ord("O")


def has_prefix(node_id: int, prefix: int) -> bool:
    return ((int(node_id) >> 56) & 0xFF) == prefix


def find_key_offset(path: Path, key: str, chunk_size: int = 8 << 20) -> int:
    needle = f'"{key}":['.encode("ascii")
    with path.open("rb") as handle:
        tail = b""
        position = 0
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise KeyError(f"Array {key!r} not found in {path}")
            merged = tail + chunk
            index = merged.find(needle)
            if index >= 0:
                return position - len(tail) + index + len(needle)
            tail = merged[-len(needle) :]
            position += len(chunk)


def iter_array(path: Path, key: str, read_size: int = 1 << 20):
    offset = find_key_offset(path, key)
    decoder = json.JSONDecoder()
    with path.open("rb") as binary:
        binary.seek(offset)
        text = io.TextIOWrapper(binary, encoding="utf-8")
        buffer = ""
        index = 0
        while True:
            while index < len(buffer) and (buffer[index].isspace() or buffer[index] == ","):
                index += 1
            if index < len(buffer) and buffer[index] == "]":
                return
            try:
                value, end = decoder.raw_decode(buffer, index)
            except json.JSONDecodeError:
                if index:
                    buffer = buffer[index:]
                    index = 0
                chunk = text.read(read_size)
                if not chunk:
                    raise ValueError(f"Truncated JSON array {key!r} in {path}")
                buffer += chunk
                continue
            yield value
            index = end
            if index > read_size:
                buffer = buffer[index:]
                index = 0


def compact(source: Path, output: Path, force: bool = False) -> dict:
    if output.exists() and not force:
        with np.load(output, allow_pickle=False) as data:
            source_matches = (
                int(data["source_size"][0]) == source.stat().st_size
                and int(data["source_mtime_ns"][0]) == source.stat().st_mtime_ns
            )
            native_fields_present = (
                "object_mesh_vertices" in data.files and "agent_position" in data.files
            )
            if source_matches and native_fields_present:
                return {
                    "status": "skipped",
                    "objects": int(len(data["object_ids"])),
                    "edges": int(len(data["edge_source"])),
                }

    started = time.time()
    edge_source, edge_target, edge_st, edge_ts, edge_rows, edge_cols = [], [], [], [], [], []
    hierarchy_source, hierarchy_target = [], []
    for edge in iter_array(source, "edges"):
        source_id, target_id = int(edge["source"]), int(edge["target"])
        source_object = has_prefix(source_id, OBJECT_PREFIX)
        target_object = has_prefix(target_id, OBJECT_PREFIX)
        if source_object and target_object:
            info = edge.get("info", {})
            st_info = info.get("relationship_source_target.feature", {})
            ts_info = info.get("relationship_target_source.feature", {})
            st = st_info.get("data")
            ts = ts_info.get("data")
            if st and ts:
                edge_source.append(source_id)
                edge_target.append(target_id)
                edge_st.append(np.asarray(st, dtype=np.float16))
                edge_ts.append(np.asarray(ts, dtype=np.float16))
                edge_rows.append(int(st_info.get("rows", 1)))
                edge_cols.append(int(st_info.get("cols", len(st))))
        elif source_object or target_object:
            hierarchy_source.append(source_id)
            hierarchy_target.append(target_id)
        else:
            # Keep place-room/building hierarchy, but no feature payload.
            hierarchy_source.append(source_id)
            hierarchy_target.append(target_id)

    object_ids, positions, dimensions, features, observations, mesh_vertices, names = [], [], [], [], [], [], []
    room_ids, room_names, room_features = [], [], []
    latest_agent_time = -1
    latest_agent_position = None
    for node in iter_array(source, "nodes"):
        attrs = node.get("attributes", {})
        node_type = attrs.get("type")
        if node_type == "ObjectNodeAttributes":
            feature = attrs.get("semantic_feature")
            position = attrs.get("position")
            if not feature or position is None:
                continue
            object_ids.append(int(node["id"]))
            positions.append(position)
            dimensions.append(attrs.get("bounding_box", {}).get("dimensions", [0.0, 0.0, 0.0]))
            features.append(np.asarray(feature, dtype=np.float16))
            observations.append(int(attrs.get("num_observations", 0)))
            mesh_vertices.append(len(attrs.get("mesh_connections") or []))
            names.append(str(attrs.get("name", "")))
        elif node_type == "RoomNodeAttributes":
            vectors = attrs.get("feature_vectors") or []
            room_ids.append(int(node["id"]))
            room_names.append(str(attrs.get("name", "")))
            room_features.append(np.asarray(vectors, dtype=np.float16))
        elif node_type == "AgentNodeAttributes" and attrs.get("position") is not None:
            update_time = int(attrs.get("last_update_time_ns", 0))
            if update_time >= latest_agent_time:
                latest_agent_time = update_time
                latest_agent_position = attrs["position"]

    if not object_ids:
        raise ValueError(f"No queryable objects found in {source}")
    relation_dim = len(edge_st[0]) if edge_st else 0
    room_width = max((len(value) for value in room_features), default=0)
    room_dim = max((value.shape[-1] if value.ndim == 2 and value.size else 0 for value in room_features), default=0)
    padded_rooms = np.zeros((len(room_features), room_width, room_dim), dtype=np.float16)
    room_counts = np.zeros(len(room_features), dtype=np.int32)
    for index, value in enumerate(room_features):
        if value.ndim == 2 and value.size:
            padded_rooms[index, : len(value), : value.shape[-1]] = value
            room_counts[index] = len(value)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp.npz")
    # The visual-token edge arrays are already float16 and poorly compressible.
    # A stored NPZ is several times faster to prepare and supports mmap-like
    # sequential reads; saving ~25% disk is not worth delaying evaluation.
    if relation_dim:
        edge_source_target = np.asarray(edge_st, dtype=np.float16).reshape(-1, relation_dim)
        edge_target_source = np.asarray(edge_ts, dtype=np.float16).reshape(-1, relation_dim)
    else:
        # A valid DSG may contain no object-object relationship embeddings.
        # NumPy cannot infer the -1 dimension when the other dimension is zero.
        edge_source_target = np.empty((0, 0), dtype=np.float16)
        edge_target_source = np.empty((0, 0), dtype=np.float16)

    np.savez(
        temporary,
        object_ids=np.asarray(object_ids, dtype=np.uint64),
        object_positions=np.asarray(positions, dtype=np.float32),
        object_dimensions=np.asarray(dimensions, dtype=np.float32),
        object_features=np.asarray(features, dtype=np.float16),
        object_observations=np.asarray(observations, dtype=np.int32),
        object_mesh_vertices=np.asarray(mesh_vertices, dtype=np.int32),
        object_names=np.asarray(names, dtype=np.str_),
        edge_source=np.asarray(edge_source, dtype=np.uint64),
        edge_target=np.asarray(edge_target, dtype=np.uint64),
        edge_source_target=edge_source_target,
        edge_target_source=edge_target_source,
        edge_rows=np.asarray(edge_rows, dtype=np.int32),
        edge_cols=np.asarray(edge_cols, dtype=np.int32),
        hierarchy_source=np.asarray(hierarchy_source, dtype=np.uint64),
        hierarchy_target=np.asarray(hierarchy_target, dtype=np.uint64),
        room_ids=np.asarray(room_ids, dtype=np.uint64),
        room_names=np.asarray(room_names, dtype=np.str_),
        room_features=padded_rooms,
        room_feature_counts=room_counts,
        agent_position=np.asarray(
            latest_agent_position if latest_agent_position is not None else [0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        source_size=np.asarray([source.stat().st_size], dtype=np.uint64),
        source_mtime_ns=np.asarray([source.stat().st_mtime_ns], dtype=np.uint64),
    )
    temporary.replace(output)
    return {
        "status": "complete",
        "objects": len(object_ids),
        "object_edges": len(edge_source),
        "hierarchy_edges": len(hierarchy_source),
        "rooms": len(room_ids),
        "output_bytes": output.stat().st_size,
        "elapsed_s": round(time.time() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(compact(args.source, args.output, args.force), sort_keys=True))


if __name__ == "__main__":
    main()
