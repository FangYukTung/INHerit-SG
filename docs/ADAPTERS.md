# Baseline and ViGiL3D adapters

Adapters serialize predictions into a common format while preserving each baseline's retrieval interface. Install the upstream baselines and generate their native maps before running them.

Run the scripts from the repository root. Generate native pickle/NPZ exports with trusted upstream installations and load only trusted files.

## HM3DSem-SQR baselines

| Method | Entry and required inputs |
| --- | --- |
| ConceptGraphs | `adapters/baselines/dense_native.py --method conceptgraphs`; native `.pkl.gz` object map, matching CLIP encoder, query-independent world transform. |
| DualMap | `dense_native.py --method dualmap`; native object-pickle directory, exact map-time class vocabulary, MobileCLIP-S2 and world transform. |
| HOV-SG | `hovsg_adapter.py`; `HOVSG_ROOT`, native saved graph and query configuration. Uses native hierarchy retrieval and saved object labels. |
| KeySG | `keysg_adapter.py`; `KEYSG_ROOT` and native saved map. `run_keysg_pipeline.py` supplies SAM2 config visibility and compatibility patches when constructing maps. |
| Embodied-RAG | `run_embodied_scene.py`; `EMBODIED_RAG_ROOT`, INHerit-SG observation map, text encoder and model endpoint. Rebuilds its own forest from object observations. |
| OVSG | `evaluate_ovsg_main864.py`; `OVSG_ROOT`, `OVSG_INPUT` with native Detic/OVIR fusion and scan PCDs; model endpoint. |
| ReasoningGraph | `prepare_reasoninggraph_indexes.py` and `evaluate_reasoninggraph_adapted_main864.py`; native DSGs, CLIP model and model endpoint. |

Install each baseline's dependencies in its own environment and set the upstream-root variables before running its adapter.

Dense baseline example:

```sh
python adapters/baselines/dense_native.py --method conceptgraphs \
  --map /path/to/native-map.pkl.gz --transform transform.json \
  --manifest data/splits/main_864.json --scene-id SCENE \
  --output outputs/predictions/conceptgraphs/SCENE.jsonl
```

`transform.json` contains a 3×3 `rotation`, a three-vector `translation`, and `uses_gt_objects: false`. Derive the transform from input poses and coordinate conventions without matching evaluation objects. For an already aligned map, supply the identity transform. DualMap additionally needs `--class-list` and its upstream source/model dependencies.

HOV-SG expects `<graph-root>/hm3dsem/<scene>/ep_000/graph`. KeySG serialization preserves the selected native Top-1 and fills the remaining candidates in native retrieval order.

Embodied-RAG uses `embodied_nav` and `embodied_rag_core` to build its forest from object observations. Configure `OPENAI_API_KEY`, `INHERIT_API_BASE` and `EMBODIED_RAG_MODEL` for its LLM interface.

OVSG input layout is `<OVSG_INPUT>/<scene>/scan-0.020.pcd` plus `detic_output/imagenet21k-0.3/predictions/proposed_fusion_detic_native.pkl`. Generate native maps and Detic/OVIR inputs with the upstream pipeline, including its `external/OVIR-3D/src/scannet200.npy` class embeddings.

ReasoningGraph DSGs live at `<map-root>/<scene>/hm3dsem/backend/dsg.json`. Set `REASONINGGRAPH_MAP_ROOT`, `REASONINGGRAPH_INDEX_ROOT`, and `REASONINGGRAPH_CLIP_CHECKPOINT` for retrieval. `prepare_reasoninggraph_indexes.py` accepts the corresponding paths as arguments. OVSG and ReasoningGraph use `OPENAI_API_KEY` / `OPENAI_BASE_URL` and the configurable baseline endpoint options.

Score all results with `evaluation/aggregate_retrieval.py`. Dataset target counts and target IDs serve the evaluator; ranking uses the method's predictions.

## ViGiL3D: fixed Mask3D and GT-box proposals

Both protocols use the INHerit-SG retrieval pipeline. Mask3D uses predicted instances. GT-box uses the complete query-independent official object inventory and exact evaluator boxes; report it as an oracle proposal condition. Query target IDs and counts serve evaluation and do not guide proposal selection or predictions.

1. Obtain licensed ScanNet/ScanNet++ raw data, official ViGiL3D annotation parquet and evaluator metadata. Put them under `data/ViGiL3D/{raw,annotations,metric_metadata}` or pass explicit paths where supported. GT-box defaults use this layout.
2. `prepare_inherit_trajectory.py --scene SCENE` converts raw frames/poses. It needs `lz4`, Pillow, NumPy and the relevant video decoding tools. Use its saved `trajectory_metadata.json` calibration when configuring the mapper; do not reuse HM3D intrinsics.
3. Run `inherit-sg map` and save each scene under `<native-maps>/<scene>/ep_000`. Build native context without any query annotations.
4. For Mask3D, provide `<proposals>/<scene>.npz` with `ins_pcds`, `ins_labels`, `ins_scores`; run `mask3d_structural_adapter.py --scene-id SCENE --native-maps ... --proposals ...`. `convert_mask3d_export.py` converts native exports. Optional captioning uses mapping vision endpoints.
5. For GT-box, run `gt_structural_adapter.py --scene-id SCENE --native-maps ...`. It loads the complete official scene labels and metadata AABBs.

Proposal adapters reuse query-independent room/area context and recompute object relations. Submitted boxes retain the fixed proposal AABBs.

Export query-only inputs and run:

```sh
python adapters/vigil3d/export_queries.py --annotations /path/to/validation.parquet \
  --output outputs/vigil_queries.json
python adapters/vigil3d/run_queries.py --queries outputs/vigil_queries.json \
  --maps-root outputs/vigil3d/mask3d/maps --output outputs/vigil_predictions.json
```

The set selector uses candidate observations and textual memory to output multiple IDs or an empty set. An invalid or failed response falls back to the initial Top-1 and records the fallback. Boxes are exported as `{ "<scene_id>_<ann_id>": [[[min_x,min_y,min_z],[max_x,max_y,max_z]], ...] }`; a no-target prediction is `[]`.

The evaluator integration uses the official ViGiL3D metric from `eamonn-zh/torchmetrics_ext` at commit `d89bc18ff1ccd8de2c66abef51ffbde1f3107093`. Install that checkout and run `pip install -r adapters/vigil3d/requirements-eval.txt`, then evaluate:

```sh
python adapters/vigil3d/validate_predictions.py outputs/vigil_predictions.json \
  --annotations /path/to/validation.parquet
python adapters/vigil3d/evaluate_predictions.py outputs/vigil_predictions.json \
  --annotations /path/to/validation.parquet --metric-metadata /path/to/metric_metadata
```

`--partial` explicitly enables a partial official evaluation. Full official validation requires the complete annotation set.
