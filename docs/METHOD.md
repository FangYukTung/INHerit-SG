# Method and parameters

| Component | Implementation | Behavior |
| --- | --- | --- |
| Producer–consumer mapping | `mapping.py`, `streaming.py` | Separate geometric/semantic state; owned context snapshots; FIFO queue; blocking backpressure; worker errors propagate. |
| Two-stage association | `mapping_backend.py`, `reid.py` | Global greedy high-similarity association; unmatched observations select a visually compatible candidate after spatial and semantic filtering. |
| Structural events | `mapping.py` | Refresh pending/changed rooms at room transitions; finalize the old floor before switching; flush at sequence end. |
| Geometric cleanup | `mapping_backend.py` | Prune small or empty room masks, close gaps within retained masks, regularize boundaries and resolve overlaps. |
| Four-level hierarchy | `mapping_backend.py` | Best-view room evidence; expansion of category vocabulary; spatial area clustering and semantic summaries. |
| Typed constraints | `query.py` | Independent signed attributes, room/floor constraints, parallel relations and recursively nested references. |
| Evidence routing | `query.py`, `retrieval.py` | Negated reference relations and nested relations with a room cue bypass relation scoring while retaining type and hierarchy; other queries use relations. |
| Three-path pool | `retrieval.py` | Structured, target text and raw-query text; weighted RRF; structured Top-1 retained; ID deduplication. |
| Selective visual reranking | `visual.py` | Positive visible attributes, no reference objects, usable observations; one listwise selection, confidence-gated promotion. |

## Configuration

`configs/default.yaml` configures mapping; `configs/retrieval.yaml` configures queries. Set credentials through environment variables. The model router writes response caches and usage records under `outputs/`.

| Setting | Default | Meaning |
| --- | ---: | --- |
| Queue capacity | 50 | Maximum queued contexts, excluding work in progress. |
| Keyframe cosine threshold | 0.85 | Compare current frame with last accepted keyframe. |
| Occupancy resolution | 0.05 m | Shared coordinate grid. |
| Initial floor | 0 | `floor_tracking.initial_floor`; configurable signed integer. |
| Floor height quantum | 3.5 m | Height interval used for floor tracking. |
| Floor confirmation | 20 frames | Consecutive consistent floor IDs; code constant. |
| Stage-1 visual threshold | 0.8 | Greedy visual association. |
| Stage-2 spatial radius | 1.5 m | `force_merge_distance_threshold` gates candidate association by distance. |
| Stage-2 label similarity | 0.6 | Specific-label semantic compatibility, with exact/substr compatibility also accepted. |
| Stage-2 visual threshold | 0.6 | Best eligible visual candidate must exceed threshold. |
| Functional-area proximity | 1.2 m | Horizontal connected-component clustering; code constant. |
| Target similarity gate | 0.55 | Structured candidate category filtering. |
| Near fallback | 1.5 m | Geometric near evidence from textual 3D locations. |
| Route depth / final pool | 20 / 5 | Candidates retained per path / after fusion. |
| RRF constant | 10 | Score proportional to `1/(10 + rank)`. |
| Structured / target / full query weights | 1 / 0.5 / 0.35 | Fusion weights. |
| Target / room / relation points | 8 / 1.5 / 3 | Structured evidence weights, normalized over active constraints. |
| Attribute points | 2 (1.2 with references) | Divided across attributes, each retaining its polarity. |
| Visual confidence / image budget | 0.8 / 5 | Promotion gate and maximum candidates inspected. |
| Visual context padding | 2× each box dimension per side | Stored best view with red box, JPEG quality 90. |
| Maximum parsed nesting | 16 | Resource guard for recursively nested constraints. |

The room cleaner removes masks below 50 grid cells and empty masks below 2,000 cells, then regularizes the retained masks and resolves overlaps.

## Query schema

Floor indices are signed: `floor N` maps to N, `first floor` to 1, `second floor` to 2, `ground floor` to 0, `basement` / `B1` to -1, and `B2` to -2. Constraint polarity is independent of the floor index: “not in the basement” is `{"index": -1, "polarity": -1}`.

`floor_tracking.initial_floor` defaults to 0 and labels the floor of the first observation. Set it to -1 when a trajectory begins in the basement, or 1 when it begins on floor 1. Tracking uses a 3.5 m height quantum and 20-frame transition confirmation: `floor = initial_floor + round((height - initial_height) / 3.5)`. Both streams use this setting, and export preserves signed indices. Set the starting floor from the recording's known location; geometry alone cannot identify the physical ground floor.

```json
{
  "target": "cup",
  "attributes": [
    {"text": "red", "polarity": 1, "visual": true},
    {"text": "plastic", "polarity": -1, "visual": true}
  ],
  "rooms": [{"text": "kitchen", "polarity": 1}],
  "floors": [{"index": 0, "polarity": 1}],
  "relations": []
}
```

A relation is `{"predicate":"on","polarity":1,"reference":{...object constraint...}}`. Reference attributes and relations remain inside that reference. Positive floor constraints filter structured candidates; if none survive, that path retries globally. Dense paths widen recall and do not replace a nonempty structured Top-1. Negative constraints penalize compatible evidence; relation evidence is bypassed for the routed cases above.

Visual attributes include color, material, shape, texture, pattern and visible state. The reranker requires at least two usable images including the initial top candidate. Missing images, low confidence, invalid model choices or API failures preserve the candidate order.
