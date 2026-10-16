"""Asynchronous geometric tracking and event-driven semantic mapping.

Both streams use the same Habitat/OpenGL camera-to-world convention and grid.
The semantic stream receives owned snapshots, never a live-map callback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import copy

import numpy as np

from .streaming import consume_async


GRID_FIELDS = ("logodds", "door_map", "occ_bin", "free_bin", "room_id_map", "door_hits")


@dataclass
class FrameContext:
    index: int
    pose: np.ndarray
    floor: int
    room: int
    base_height: float
    grids: dict
    observations: tuple
    keyframe: bool


def room_signature(mask, objects, observations):
    digest = hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes())
    fields = ("node_id", "label", "llm_desc", "global_pos_3d", "full_image_path", "bbox")
    content = [{k: item.get(k) for k in fields} for item in sorted(objects, key=lambda item: item["node_id"])]
    digest.update(json.dumps(content, sort_keys=True, default=lambda x: np.asarray(x).tolist()).encode())
    digest.update(json.dumps(observations).encode())
    return digest.hexdigest()


def run_mapping(config):
    # Heavy mapping dependencies are optional for retrieval and evaluation.
    import cv2
    from PIL import Image
    from .mapping_backend import TrajectoryPipeline, MapManager, GeometryRoomTracker, FloorTracker, extract_feature
    from .common import read_pose_matrix

    output = Path(config["main_output_directory"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Mapping output must be empty: {output}")
    pipeline = TrajectoryPipeline(config)
    signatures = {}
    valid_rooms = {}
    last_context = None

    def refresh(context):
        """Clean current geometry, then refresh only changed room semantics."""
        ids = pipeline.cleanup_and_optimize_rooms()
        pipeline._reassign_objects_to_rooms()
        floor = pipeline.floor_tracker.current_floor_id
        valid_rooms[floor] = ids
        for rid in ids:
            objects = [obj for obj in pipeline.memory_kb if obj.get("floor_id") == floor and obj.get("parent_room") == rid]
            frames = pipeline.room_frame_history.get(rid, [])
            signature = room_signature(pipeline.map_manager.room_id_map == rid, objects, frames)
            key = (floor, int(rid))
            if signatures.get(key) == signature:
                continue
            node = pipeline.finalize_room_semantics(rid, force_vlm_expansion=True)
            if node:
                pipeline.floor_tracker.add_room(node)
                # Finalization can expand objects; capture its resulting state.
                updated = [obj for obj in pipeline.memory_kb if obj.get("floor_id") == floor and obj.get("parent_room") == rid]
                signatures[key] = room_signature(pipeline.map_manager.room_id_map == rid, updated, frames)

    def semantic(context):
        nonlocal last_context
        previous_floor = pipeline.floor_tracker.current_floor_id
        floor_changed = last_context is not None and context.floor != previous_floor
        room_changed = last_context is not None and context.room != last_context.room
        if floor_changed:
            # Settle old-floor state BEFORE replacing its grid and node context.
            refresh(last_context)
            pipeline.floor_tracker.finalize_floor()
            pipeline.map_manager.switch_floor(context.floor)
            pipeline.room_segmenter.switch_floor(context.floor)
        pipeline.floor_tracker.current_floor_id = context.floor
        pipeline.floor_tracker.base_height = context.base_height
        pipeline.floor_tracker.floors_data[context.floor]["heights"].append(float(context.pose[1, 3]))
        for name, value in context.grids.items():
            setattr(pipeline.map_manager, name, value.copy())
        pipeline.room_segmenter.global_id_grid = context.grids["room_id_map"].copy()
        pipeline.current_room_id = context.room
        pipeline.room_segmenter.current_room_id = context.room
        for index, room, pose in context.observations:
            pipeline.pose_cache[index] = pose
            if room > 0:
                pipeline.room_frame_history.setdefault(room, []).append(index)
        # Room transitions update all pending rooms on this floor. Incoming
        # keyframe observations are accumulated for the next maintenance pass.
        if room_changed and not floor_changed:
            refresh(context)
        if context.keyframe:
            rgb_path, depth_path = pipeline._get_image_path(context.index)
            segmentation = pipeline._run_segmentation(context.index, rgb_path, depth_path)
            pipeline._process_frame_results(context.index, segmentation, rgb_path, depth_path, context.pose)
        pipeline.last_room_id = context.room
        last_context = context

    def geometry():
        mapper = MapManager(config.get("map_config", {}), pipeline.intrinsics, pipeline.run_dir)
        tracker = GeometryRoomTracker(mapper, config.get("room_segmentation", {}))
        initial_floor = config.get("floor_tracking", {}).get("initial_floor", 0)
        floors = FloorTracker(str(output / "geometry_floors.jsonl"), initial_floor=initial_floor)
        mapper.current_floor_idx = initial_floor
        tracker.current_floor_idx = initial_floor
        last_feature = None
        previous_room = 0
        pending = []
        last = None
        limit = config["trajectory_data"].get("num_images_to_process", -1)
        index = 0
        next_room_id = tracker.next_global_room_id
        while limit < 0 or index < limit:
            rgb_path, depth_path = pipeline._get_image_path(index)
            if not Path(rgb_path).is_file():
                if index == 0:
                    raise FileNotFoundError(rgb_path)
                break
            pose_path = pipeline._get_pose_path(index)
            if not Path(depth_path).is_file() or not Path(pose_path).is_file():
                raise FileNotFoundError(f"Incomplete RGB-D/pose observation at frame {index}")
            pose = read_pose_matrix(pose_path)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"Invalid pose at frame {index}")
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise ValueError(f"Unreadable depth: {depth_path}")
            changed_floor = floors.update(pose, None)
            if changed_floor:
                # Flush old-floor observations even when no visual keyframe was accepted.
                if last is not None and pending:
                    last.observations = tuple(pending)
                    last.keyframe = False
                    yield last
                    pending = []
                floors.current_floor_id = floors.next_floor_id
                del floors.next_floor_id
                floors.id_buffer.clear()
                floors.height_buffer.clear()
                floors.pending_data_cache.clear()
                mapper.switch_floor(floors.current_floor_id)
                tracker.switch_floor(floors.current_floor_id)
                tracker.next_global_room_id = max(next_room_id, tracker.next_global_room_id)
            mapper.integrate_depth_raycast(depth, pose)
            result = tracker.step(pose)
            next_room_id = max(next_room_id, tracker.next_global_room_id)
            room = int(result["room_id"])
            mask = result.get("room_mask_local")
            if mask is not None:
                mapper.room_id_map = mask.astype(np.int32)
            with Image.open(rgb_path) as image:
                feature = extract_feature(image.convert("RGB"), pipeline.dinov3_model, pipeline.dinov3_device)
            similarity = -1 if last_feature is None else float(np.dot(feature.reshape(-1), last_feature.reshape(-1)) / max(np.linalg.norm(feature) * np.linalg.norm(last_feature), 1e-12))
            keyframe = not getattr(pipeline, "enable_kf", True) or last_feature is None or similarity < pipeline.kf_threshold
            if keyframe:
                last_feature = feature
            pending.append((index, room, pose.copy()))
            # Create current context before the enqueue operation in every branch.
            context = FrameContext(index, pose.copy(), floors.current_floor_id, room, floors.base_height,
                                   {name: getattr(mapper, name).copy() for name in GRID_FIELDS}, (), keyframe)
            if keyframe or changed_floor or room != previous_room:
                context.observations = tuple(pending)
                pending = []
                yield context
            last = context
            previous_room = room
            index += 1
        if last is not None and pending:
            last.observations = tuple(pending)
            last.keyframe = False
            yield last

    stats = consume_async(geometry(), semantic, int(config.get("streaming", {}).get("queue_capacity", 50)))
    if last_context is None:
        raise ValueError("No observations processed")
    refresh(last_context)
    pipeline.floor_tracker.finalize_floor()
    pipeline.save_all_data()
    from .hierarchy import normalize_hierarchy
    normalize_hierarchy(output, {rid for ids in valid_rooms.values() for rid in ids})
    (output / "queue_stats.json").write_text(json.dumps(asdict(stats), indent=2) + "\n")
    return stats
