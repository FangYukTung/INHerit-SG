"""Typed constraints with independent polarities and nested references."""
from __future__ import annotations

from dataclasses import dataclass, field
import json


def decode_model_json(content):
    """Accept JSON or a complete Markdown JSON code block from a model."""
    if not isinstance(content, str):
        raise ValueError("model response must be text")
    value = content.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in ("```", "```json") or lines[-1].strip() != "```":
            raise ValueError("invalid JSON code block")
        value = "\n".join(lines[1:-1]).strip()
    return json.loads(value)


@dataclass(frozen=True)
class Attribute:
    text: str
    polarity: int = 1
    visual: bool = False


@dataclass(frozen=True)
class Relation:
    predicate: str
    reference: "ObjectQuery"
    polarity: int = 1


@dataclass(frozen=True)
class ObjectQuery:
    target: str
    attributes: tuple[Attribute, ...] = ()
    rooms: tuple[Attribute, ...] = ()
    floors: tuple[tuple[int, int], ...] = ()
    relations: tuple[Relation, ...] = ()


def _polarity(value):
    if type(value) is not int or value not in (-1, 1):
        raise ValueError("polarity must be the integer +1 or -1")
    return value


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("constraint text must be nonempty")
    return value.strip()


def parse_object(value, *, max_depth=16, _depth=0):
    """Validate recursively. Invalid output is an error, never a partial query.

    max_depth is an input resource limit, not a fixed two-hop parser schema.
    """
    if _depth > max_depth:
        raise ValueError(f"query exceeds configured nesting limit {max_depth}")
    if not isinstance(value, dict):
        raise ValueError("object constraint must be a JSON object")
    allowed = {"target", "attributes", "rooms", "floors", "relations"}
    if set(value) - allowed:
        raise ValueError(f"unknown constraint fields: {sorted(set(value) - allowed)}")
    def attrs(name):
        result = []
        for item in value.get(name, []):
            if set(item) - {"text", "polarity", "visual"}:
                raise ValueError("unknown attribute fields")
            visual = item.get("visual", False)
            if type(visual) is not bool:
                raise ValueError("visual must be boolean")
            result.append(Attribute(_text(item["text"]), _polarity(item.get("polarity", 1)), visual))
        return tuple(result)
    floors = []
    for item in value.get("floors", []):
        if set(item) - {"index", "polarity"} or type(item["index"]) is not int:
            raise ValueError("invalid floor constraint")
        floors.append((item["index"], _polarity(item.get("polarity", 1))))
    relations = []
    for item in value.get("relations", []):
        if set(item) - {"predicate", "reference", "polarity"}:
            raise ValueError("unknown relation fields")
        relations.append(Relation(_text(item["predicate"]), parse_object(
            item["reference"], max_depth=max_depth, _depth=_depth + 1
        ), _polarity(item.get("polarity", 1))))
    return ObjectQuery(_text(value["target"]), attrs("attributes"), attrs("rooms"), tuple(floors), tuple(relations))


def evidence_route(query: ObjectQuery):
    """Paper routing: retain hierarchy when bypassing unreliable relation evidence."""
    def negated_reference(node):
        return any(r.polarity < 0 or negated_reference(r.reference) for r in node.relations)
    if negated_reference(query):
        return "hierarchy"
    if query.rooms and any(relation.reference.relations for relation in query.relations):
        return "hierarchy"
    return "relations"


def visual_attributes(query):
    if query.relations:
        return ()
    return tuple(a.text for a in query.attributes if a.visual and a.polarity > 0)


PARSER_PROMPT = '''Parse the instruction into a JSON object with this recursive schema:
{"target": "object category", "attributes": [{"text": "one attribute", "polarity": 1, "visual": true}],
 "rooms": [{"text": "room cue", "polarity": 1}],
 "floors": [{"index": 0, "polarity": 1}],
 "relations": [{"predicate": "near", "polarity": 1, "reference": {"target": "table", "attributes": [], "rooms": [], "floors": [], "relations": []}}]}
Use +1 for required constraints, -1 for excluded constraints. Split mixed-polarity
attributes: "red but not plastic" is positive red and negative plastic. Mark
visible appearance (including color, material, shape, texture, pattern, visible
state) as visual=true; location, purpose and invisible properties are not visual.
"Not near a table" negates that relation, not the table's color. An attribute on
the reference belongs inside its reference object. Recursively nest reference
relations; preserve parallel relations as separate list entries. Do not invent
constraints. Use empty lists when absent. Use signed floor indices:
"floor N" maps to index N; "first floor" maps to 1, "second floor" to 2,
and "ground floor" to 0. Do not subtract one from ordinal floor numbers.
"Basement" or "B1" maps to floor -1; "B2" maps to -2, and so on.
Keep floor polarity separate from its signed index: "not in the basement"
is {"index": -1, "polarity": -1}. Do not add a basement room constraint
unless the instruction separately specifies a room.
Output JSON only, with no extra fields.'''


class QueryParser:
    def __init__(self, client, model, max_depth=16):
        self.client, self.model, self.max_depth = client, model, max_depth

    def parse(self, instruction):
        response = self.client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "system", "content": PARSER_PROMPT},
                      {"role": "user", "content": instruction}],
            response_format={"type": "json_object"},
        )
        return parse_object(decode_model_json(response.choices[0].message.content), max_depth=self.max_depth)
