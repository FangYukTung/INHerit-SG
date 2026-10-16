#!/usr/bin/env python3
"""Build Embodied-RAG's flat object graph from the shared observation contract."""

import argparse
import json
import os
from pathlib import Path

import networkx as nx
EXPECTED_FIELDS = {
    "schema_version", "scene_id", "observation_id", "category", "caption", "position", "timestamp"
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--encoder",
        default=os.getenv("EMBODIED_RAG_ENCODER", "all-MiniLM-L6-v2"),
        help="SentenceTransformer name or, preferably, a frozen local snapshot path",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Do not materialize embeddings; official LLM forest/traversal does not consume them",
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("Observation file is empty")
    embeddings = [None] * len(rows)
    if not args.skip_embeddings:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.encoder, device=args.device)
        texts = [f"{row['category']}. {row['caption']}" for row in rows]
        embeddings = model.encode(texts, show_progress_bar=False)

    graph = nx.Graph()
    for row, embedding in zip(rows, embeddings):
        if set(row) != EXPECTED_FIELDS:
            raise ValueError(f"Unexpected shared fields: {sorted(set(row) - EXPECTED_FIELDS)}")
        x, y, z = row["position"]
        attributes = {
            "type": "object",
            "level": 0,
            "name": row["category"],
            "unique_id": row["observation_id"],
            "summary": row["caption"],
            "x": float(x), "y": float(y), "z": float(z),
            "timestamp": str(row["timestamp"]),
        }
        if embedding is not None:
            attributes["embedding"] = embedding.tolist()
        graph.add_node(row["observation_id"], **attributes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(graph, output)
    print(f"Wrote {graph.number_of_nodes()} object nodes to {output}")


if __name__ == "__main__":
    main()
