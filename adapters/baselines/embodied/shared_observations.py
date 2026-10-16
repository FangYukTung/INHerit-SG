#!/usr/bin/env python3
"""Export and validate the minimal object observations shared with Embodied-RAG."""

import argparse
import json
from pathlib import Path


FIELDS = ("schema_version", "scene_id", "observation_id", "category", "caption", "position", "timestamp")
FORBIDDEN_SOURCE_FIELDS = ("parent_area", "parent_room", "floor_id", "relations", "edges")


def read_rows(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        return json.load(handle) if first == "[" else [json.loads(line) for line in handle if line.strip()]


def convert(row, scene_id):
    level = row.get("node_level", row.get("level", 3))
    if int(level) != 3:
        return None
    position = row.get("global_pos_3d", row.get("position"))
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        raise ValueError(f"Object {row.get('node_id')} has no valid 3D position")
    result = {
        "schema_version": 1,
        "scene_id": scene_id,
        "observation_id": str(row.get("node_id", row.get("id"))),
        "category": str(row.get("semantic_tag", row.get("label", row.get("category", "unknown")))),
        "caption": str(row.get("llm_desc", row.get("caption", ""))),
        "position": [float(value) for value in position[:3]],
        "timestamp": row.get("timestamp"),
    }
    if set(result) != set(FIELDS):
        raise AssertionError("Shared observation schema changed unexpectedly")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="INHerit-SG object nodes JSON/JSONL")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = read_rows(args.input)
    output_rows = [item for row in source if (item := convert(row, args.scene_id)) is not None]
    if not output_rows:
        raise ValueError("No object observations were exported")
    if len({row["observation_id"] for row in output_rows}) != len(output_rows):
        raise ValueError("Duplicate observation IDs")
    for row in output_rows:
        if any(field in row for field in FORBIDDEN_SOURCE_FIELDS):
            raise AssertionError("Structural INHerit-SG field leaked into shared observations")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"Wrote {len(output_rows)} minimal observations to {output}")


if __name__ == "__main__":
    main()
