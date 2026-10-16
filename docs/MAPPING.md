# Mapping setup

Mapping uses a geometric producer and a semantic consumer. The producer integrates RGB-D and poses, tracks floors/rooms, and selects keyframes using DINOv3. The consumer uses SAM 3, visual association, descriptions, and event-triggered hierarchy maintenance. The queue holds 50 contexts by default and makes the producer wait when it is full.

## Dependencies and models

1. Install a matched PyTorch/torchvision build for your GPU, then `pip install -e '.[mapping]'`.
2. Install SAM 3 and its dependencies separately. Set `INHERIT_SG_SAM3_ROOT` to the SAM 3 source checkout if it is not installed on the Python path.
3. Set `INHERIT_SAM3_CHECKPOINT` to the checkpoint, `INHERIT_SAM3_BPE` to SAM 3's BPE vocabulary, and `INHERIT_DINO_MODEL` to a local Hugging Face format DINOv3 model directory containing `config.json`. The loader uses `trust_remote_code=True`, so use a model directory you trust.
4. Set `INHERIT_API_BASE` and `INHERIT_API_KEY`. Edit the three endpoints in `configs/default.yaml` to your chosen models. Mapping requires image-capable vision endpoints and a text endpoint.

The configured endpoints receive prompts and, for vision tasks, scene frames or object crops. Use endpoints authorized to process your input data. Query parsing sends query text; `--no-visual` disables query-time visual reranking, not mapping-time vision calls.

## Input convention

The default layout is:

```text
episode/
  rgb/000000.png
  depth/000000.png
  pose/000000.txt
  rgb/000001.png
  depth/000001.png
  pose/000001.txt
```

Poses contain 16 whitespace-separated values in a 4×4 camera-to-world matrix. The mapper uses the Habitat/OpenGL convention: world Y is vertical, camera forward is -Z, and camera image Y is inverted when projecting depth. Depth images store integer millimeters with default `depth_scale: 1000`. RGB and depth must be synchronized, registered, and match the configured intrinsics. The example intrinsics are fx=fy=540, cx=540, cy=360; replace them with your camera calibration.

The sequence must be contiguous. A missing RGB after the first frame ends the sequence; missing depth/pose for an existing RGB is an error. Prefix, zero padding, starting index, RGB extension and subdirectory names are configurable. Prepare ViGiL3D trajectories with their dedicated adapter.

```sh
export INHERIT_TRAJECTORY=/absolute/path/to/episode
inherit-sg map --config configs/default.yaml
```

Edit `main_output_directory` before each run. The output directory must be empty. Multi-GPU device assignment is optional in `multi_gpu`.

## Floor numbering

The first observation is on floor 0 by default. If a recording starts elsewhere, set:

```yaml
floor_tracking:
  initial_floor: -1  # Start in the basement; use 1 to start on floor 1.
```

The geometry and semantic streams share this reference. Floor indices remain signed through export and retrieval, with `basement` mapped to -1. The height step is 3.5 m, and transitions require 20 consistent frames.

## Output

- `memory_kb.json`: objects, 3D geometry, association features and observation references.
- `nodes.jsonl`, `relations.csv`: hierarchy nodes and containment edges.
- `spatial_graph.json`: textual spatial edges consumed by retrieval.
- `spatial_graph.pkl`, `map_data.pkl`: serialized mapping state.
- Best-view images, room masks and `config_snapshot.yaml`.
- `queue_stats.json`: submitted/completed tasks, sampled queue occupancy, observed full-queue waits and waiting time.

Keep the RGB trajectory available when using stored frame evidence paths. Text retrieval and exported textual memory use text without image or feature payloads.
