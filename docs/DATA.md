# HM3DSem-SQR data and evaluation

## Included files

| Location | Contents |
| --- | --- |
| `data/full/` | 6,084 generated queries over 36 scenes, plus scene object/geometry metadata. |
| `data/reviewed/` | 864 reviewed evaluation queries and their scene metadata. Use these annotations with the canonical splits. |
| `data/splits/main_864.json` | Main-864: 864 unique queries. |
| `data/splits/core_432.json` | Core-432 subset of Main-864. |
| `data/splits/map_ablation_96.json` | Mapping-96 subset of Main-864. |
| `data/manifest.json` | Query counts and SHA-256 hashes of the annotations and splits. |
| `data/poses/` | Camera poses and frame order for 39,598 selected frames across 36 scenes. |
| `data/poses/manifest.json` | Camera/render configuration and SHA-256 hashes of the pose/sequence payloads. |

The 864 reviewed queries form the evaluation set; their original generated versions remain in `data/full/`. Query IDs are scoped by scene and episode; `key` is `<scene_id>/<episode_id>/<query_id>`. Relative evidence paths point to the corresponding raw observations.

The [Matterport academic-use agreement](https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data) governs the HM3D-derived queries, poses, and scene annotations. Use them for academic, non-commercial research and retain the upstream attribution and agreement link. The MIT license covers the original code.

The repository supplies annotations and camera poses. Obtain HM3D assets separately, render RGB-D observations locally, and build maps with INHerit-SG. Obtain ScanNet, ScanNet++ and ViGiL3D source data under their upstream terms.

## Obtain HM3DSem assets and render observations

Obtain these three HM3D v0.2 **validation** archives from the [official Matterport download page](https://github.com/matterport/habitat-matterport-3dresearch#-downloading-hm3d-v02), subject to the [Matterport academic-use agreement](https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data):

- `hm3d-val-habitat-v0.2.tar`
- `hm3d-val-semantic-annots-v0.2.tar`
- `hm3d-val-semantic-configs-v0.2.tar`

Arrange their contents under one asset root containing `hm3d_annotated_basis.scene_dataset_config.json` and `val/<scene>/<scene-hash>.{basis.glb,semantic.glb,semantic.txt}`. Render the selected camera poses:

```sh
python scripts/render_hm3dsqr.py --verify-only
python scripts/render_hm3dsqr.py --assets /path/to/hm3d-v0.2 \
  --output data/trajectories
```

The renderer writes `data/trajectories/val/<scene>/ep_000/{rgb,depth,pose}/` in the selected frame order with the specified camera calibration. Add `--semantic` to save semantic arrays. Evaluate with `data/reviewed/` and `data/splits/`; `data/full/` contains the generated query set.

For mapping, set `INHERIT_TRAJECTORY` to the rendered episode, use that scene's filename prefix (for example `TEEsavR23oF_`), and retain the documented camera intrinsics. Keep the generated frames available for image evidence during retrieval.

## Run the evaluator

Prediction files are JSONL under `outputs/predictions/<method>/<scene_id>.jsonl`. Each line has:

```json
{
  "key": "scene/ep_000/query-id",
  "scene_id": "scene",
  "query_id": "query-id",
  "prediction": {"id": "map-object-id", "label": "chair", "position": [0, 0, 0]},
  "candidates": [{"id": "map-object-id", "label": "chair", "position": [0, 0, 0]}]
}
```

Positions are in the source scene's world coordinates. `prediction` agrees with the first candidate. No prediction is represented by `null` and an empty candidate list. Every split key must appear exactly once, including failures.

```sh
python evaluation/aggregate_retrieval.py outputs/predictions \
  --split data/splits/main_864.json --dataset-root data/reviewed \
  --output outputs/evaluation/main864
```

Identity alignment uses point-to-OBB distance with a 0.25 m eligibility threshold. Among candidates within 0.05 m of the closest eligible object, category compatibility breaks ties; when none are compatible, geometry determines the match. The evaluator uses its soft category ontology for tie-breaking and reports Strict Top-1, Recall@5, SR@1m and diagnostics. Query-time ranking uses the map rather than evaluator categories or target IDs.

`evaluation/run_retrieval.py` uses only split keys, scene IDs and query text for inference. Optional `--constraints` accepts a JSON mapping from split keys to the recursive query schema, allowing a local parser to provide results. `--no-visual` disables visual reranking. A subset run needs its matching subset manifest when evaluated.

## Reconstruct the mapping inputs

Use the supplied poses and locally obtained HM3D assets to render the RGB-D observations.

Each scene has `poses.txt` (one row-major 4×4 camera-to-world matrix per line) and `frames.json` (ordered filename stems and RGB/depth/pose checksums for the source observations). Coordinates are Habitat/OpenGL: meters, world +Y up, camera +X right, +Y up and -Z forward. Camera poses include camera height.

The capture rig uses 1080×720 pinhole sensors, 90° horizontal field of view, and fx=fy=540, cx=540, cy=360. Depth is saved as `(depth_m * 1000).astype(uint16)`.

For rendering, use a separate environment with Habitat-Sim 0.3.3, NumPy, numpy-quaternion, SciPy and OpenCV. The standalone script supports Python 3.9; the INHerit-SG package requires Python 3.10+. Habitat-Sim needs a headless GPU renderer; see the [official installation instructions](https://github.com/facebookresearch/habitat-sim#installation).

```sh
# Full sequence for one scene; omit --scene to render all 36 scenes.
python scripts/render_hm3dsqr.py --assets /path/to/hm3d-v0.2 \
  --output data/trajectories --scene 00800-TEEsavR23oF
```

Output is `<output>/val/<scene>/ep_000/{rgb,depth,pose}/`. Add `--semantic` to save semantic arrays. Existing episode directories are rejected; use a fresh output root for each run. A complete run sets `complete: true` in the per-episode report.

To map this example, copy `configs/default.yaml`, set `trajectory_data.filename_prefix: TEEsavR23oF_`, and set `INHERIT_TRAJECTORY` to the generated episode directory. Retain the documented intrinsics. The default starting floor is 0; set `floor_tracking.initial_floor` if another reference is needed.
