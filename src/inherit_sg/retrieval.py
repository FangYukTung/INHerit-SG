"""Textual hierarchy, recursive relations, three-path fusion and optional reranking."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import csv
import json
import os
from pathlib import Path
import re

import numpy as np

from .query import ObjectQuery, evidence_route


ALIASES = {"next_to": "near", "beside": "near", "close_to": "near", "on_top_of": "on", "beneath": "under", "in": "inside"}
INVERSE = {"on": "under", "under": "on", "above": "below", "below": "above", "left_of": "right_of", "right_of": "left_of", "inside": "contains", "contains": "inside"}


def relation_name(text):
    name = re.sub(r"[^a-z0-9]+", "_", text.lower().strip()).strip("_")
    return ALIASES.get(name, name)


@dataclass
class Memory:
    objects: dict
    nodes: dict
    parents: dict
    edges: tuple

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        objects = json.loads((directory / "memory_kb.json").read_text())
        nodes = {}
        for line in (directory / "nodes.jsonl").read_text().splitlines():
            if line.strip():
                node = json.loads(line)
                nodes[node["node_id"]] = node
        with (directory / "relations.csv").open() as handle:
            parents = {r["child_node_id"]: r["parent_node_id"] for r in csv.DictReader(handle)}
        edges = json.loads((directory / "spatial_graph.json").read_text())
        return cls({o["node_id"]: o for o in objects}, nodes, parents, tuple(edges))

    def room(self, node_id):
        seen = set()
        while node_id in self.parents:
            if node_id in seen:
                raise ValueError("cycle in hierarchy")
            seen.add(node_id)
            node_id = self.parents[node_id]
            node = self.nodes.get(node_id, {})
            if node.get("node_level") == 1:
                return node
        return {}

    def floor(self, node_id):
        obj = self.objects[node_id]
        room = self.room(node_id)
        return obj.get("floor_id", room.get("floor_id"))

    def text(self):
        """Complete textual hierarchy and relations; no image or feature payloads."""
        forbidden = {"dinov3_feature", "feature", "features", "image_path", "full_image_path", "cropped_image_path", "bbox"}
        def clean(value):
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items() if k not in forbidden and "image" not in k and "feature" not in k}
            if isinstance(value, list):
                return [clean(v) for v in value]
            return value
        return json.dumps(clean({"nodes": self.nodes, "objects": self.objects, "parents": self.parents, "relations": self.edges}), ensure_ascii=False)


@dataclass(frozen=True)
class RetrievalSettings:
    target_threshold: float = 0.55
    near_distance: float = 1.5
    route_depth: int = 20
    pool_size: int = 5
    rrf_constant: float = 10.0
    structured_weight: float = 1.0
    target_weight: float = 0.5
    full_query_weight: float = 0.35
    name_points: float = 8.0
    room_points: float = 1.5
    attribute_points: float = 2.0
    referenced_attribute_points: float = 1.2
    relation_points: float = 3.0


def fuse_pool(routes, settings):
    weights = {"structured": settings.structured_weight, "target": settings.target_weight, "full_query": settings.full_query_weight}
    scores = {}
    for route, ids in routes.items():
        unique = list(dict.fromkeys(ids))[:settings.route_depth]
        for rank, node_id in enumerate(unique, 1):
            scores[node_id] = scores.get(node_id, 0.0) + weights[route] / (settings.rrf_constant + rank)
    order = sorted(scores, key=lambda node_id: (-scores[node_id], node_id))
    if routes["structured"]:
        first = routes["structured"][0]
        order = [first] + [node_id for node_id in order if node_id != first]
    return order[:settings.pool_size]


class StructuredRetriever:
    def __init__(self, memory, encoder=None, settings=None):
        self.memory = memory
        self.settings = settings or RetrievalSettings()
        if self.settings.route_depth < 1 or self.settings.pool_size < 1 or self.settings.rrf_constant < 0:
            raise ValueError("invalid candidate budget or RRF constant")
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(os.environ.get("INHERIT_SG_TEXT_ENCODER", "sentence-transformers/all-MiniLM-L6-v2"))
        self.encoder = encoder
        self._vectors = {}
        self.neighbors = {node_id: [] for node_id in memory.objects}
        for edge in memory.edges:
            source, target = edge["source"], edge["target"]
            relation = relation_name(edge["relation"])
            if source in self.neighbors and target in self.neighbors:
                self.neighbors[source].append((target, relation))
                # Never reverse asymmetric predicates without a defined inverse.
                inverse = INVERSE.get(relation, relation if relation == "near" else None)
                if inverse:
                    self.neighbors[target].append((source, inverse))

    def vector(self, text):
        if text not in self._vectors:
            value = np.asarray(self.encoder.encode([text], normalize_embeddings=True)[0], dtype=float)
            self._vectors[text] = value
        return self._vectors[text]

    def sim(self, first, second):
        return float(self.vector(first) @ self.vector(second)) if first and second else 0.0

    def related(self, node_id, predicate):
        predicate = relation_name(predicate)
        result = {other for other, rel in self.neighbors[node_id] if rel == predicate}
        if predicate == "near":
            pos = self.memory.objects[node_id].get("global_pos_3d")
            if pos is not None:
                for other, obj in self.memory.objects.items():
                    value = obj.get("global_pos_3d")
                    if other != node_id and value is not None and np.linalg.norm(np.asarray(pos) - value) <= self.settings.near_distance:
                        result.add(other)
        return sorted(result)

    def retrieve(self, instruction, query, reranker=None):
        route = evidence_route(query)
        settings = self.settings

        @lru_cache(maxsize=None)
        def score(node_id, constraint, use_relations=True, ignore_positive_floor=False, nested=False):
            obj = self.memory.objects[node_id]
            for index, polarity in constraint.floors:
                if polarity > 0 and not ignore_positive_floor and self.memory.floor(node_id) != index:
                    return None
            name_score = self.sim(constraint.target, obj.get("label", ""))
            if name_score < settings.target_threshold:
                return None
            pieces = [(settings.name_points, name_score)]
            attrs = constraint.attributes
            if attrs:
                points = settings.referenced_attribute_points if constraint.relations else settings.attribute_points
                pieces.extend((points / len(attrs) * a.polarity, self.sim(a.text, obj.get("llm_desc", ""))) for a in attrs)
            room = self.memory.room(node_id)
            room_text = f"{room.get('room_label', room.get('semantic_tag', ''))}. {room.get('llm_summary', '')}"
            if constraint.rooms:
                pieces.extend((settings.room_points / len(constraint.rooms) * cue.polarity, self.sim(cue.text, room_text)) for cue in constraint.rooms)
            for index, polarity in constraint.floors:
                if polarity < 0:
                    pieces.append((-settings.room_points, float(self.memory.floor(node_id) == index)))
            if use_relations and constraint.relations:
                for relation in constraint.relations:
                    compatible = []
                    for other in self.related(node_id, relation.predicate):
                        child = score(other, relation.reference, True, False, True)
                        if child is not None:
                            compatible.append(child)
                    # Inside-out resolution: a reference must satisfy its positive chain.
                    if nested and relation.polarity > 0 and not compatible:
                        return None
                    evidence = max(compatible) if compatible else 0.0
                    pieces.append((settings.relation_points / len(constraint.relations) * relation.polarity, evidence))
            return sum(weight * value for weight, value in pieces) / sum(abs(weight) for weight, _ in pieces)

        ranked = [(node_id, score(node_id, query, route == "relations")) for node_id in self.memory.objects]
        ranked = [(node_id, value) for node_id, value in ranked if value is not None]
        floor_fallback = False
        if not ranked and any(polarity > 0 for _, polarity in query.floors):
            floor_fallback = True
            ranked = [(node_id, score(node_id, query, route == "relations", True)) for node_id in self.memory.objects]
            ranked = [(node_id, value) for node_id, value in ranked if value is not None]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        target_text = " ".join([query.target] + [a.text for a in query.attributes if a.polarity > 0])
        def dense(text):
            return sorted(self.memory.objects, key=lambda node_id: (
                -self.sim(text, f"{self.memory.objects[node_id].get('label', '')}. {self.memory.objects[node_id].get('llm_desc', '')}"), node_id
            ))[:settings.route_depth]
        routes = {"structured": [node_id for node_id, _ in ranked[:settings.route_depth]],
                  "target": dense(target_text), "full_query": dense(instruction)}
        pool = fuse_pool(routes, settings)
        rerank_trace = {"eligible": False, "promoted": False}
        if reranker is not None:
            pool, rerank_trace = reranker.rerank(instruction, query, pool, self.memory)
        candidates = [{"id": node_id, "node_id": node_id,
                       "label": self.memory.objects[node_id].get("label"),
                       "position": self.memory.objects[node_id].get("global_pos_3d")} for node_id in pool]
        return {"prediction": candidates[0] if candidates else None, "candidates": candidates,
                "route": route, "floor_fallback": floor_fallback, "routes": routes, "reranking": rerank_trace}
