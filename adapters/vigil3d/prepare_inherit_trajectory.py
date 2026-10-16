#!/usr/bin/env python3
"""Convert licensed ScanNet/ScanNet++ streams to INHerit-SG trajectories.

Only raw RGB-D, camera calibration, and poses are read. ViGiL3D target IDs and
evaluator boxes are intentionally outside this preprocessing interface.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import struct
import subprocess
import zlib

import lz4.block
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SCANNET_ROOT = ROOT / "data/ViGiL3D/raw/scannet/scans"
SCANNETPP_ROOT = ROOT / "data/ViGiL3D/raw/scannetpp/data"
OUTPUT_ROOT = ROOT / "data/ViGiL3D/derived/inherit_2hz"
TARGET_WIDTH, TARGET_HEIGHT = 640, 480
CV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", help="Scene ID; repeat as needed.")
    parser.add_argument("--source", choices=("all", "scannet", "scannetpp"), default="all")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--scannet-root", type=Path, default=SCANNET_ROOT)
    parser.add_argument("--scannetpp-root", type=Path, default=SCANNETPP_ROOT)
    parser.add_argument("--max-sampled-frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_scene_ids(args: argparse.Namespace) -> list[str]:
    if args.scene:
        return args.scene
    manifests = ROOT / "data/ViGiL3D/manifests"
    ids: list[str] = []
    if args.source in ("all", "scannet"):
        ids += (manifests / "scannet_scenes.txt").read_text().split()
    if args.source in ("all", "scannetpp"):
        ids += (manifests / "scannetpp_scenes.txt").read_text().split()
    return ids


def axis_alignment(path: Path) -> np.ndarray:
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("axisAlignment"):
            values = [float(value) for value in line.split("=", 1)[1].split()]
            if len(values) == 16:
                return np.asarray(values, dtype=np.float64).reshape(4, 4)
    raise ValueError(f"Missing axisAlignment in {path}")


def write_pose(path: Path, pose: np.ndarray) -> None:
    np.savetxt(path, pose, fmt="%.9f")


def prepare_dirs(root: Path, overwrite: bool) -> tuple[Path, Path, Path]:
    if root.exists() and not overwrite and (root / "trajectory_metadata.json").is_file():
        raise FileExistsError(f"Prepared trajectory already exists: {root}")
    rgb, depth, pose = root / "rgb", root / "depth", root / "pose"
    for path in (rgb, depth, pose):
        path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in (*rgb.glob("*.jpg"), *depth.glob("*.png"), *pose.glob("*.txt")):
            path.unlink()
    return rgb, depth, pose


def skip_matrix(handle: io.BufferedReader) -> np.ndarray:
    return np.asarray(struct.unpack("<16f", handle.read(64)), dtype=np.float64).reshape(4, 4)


def prepare_scannet(scene_id: str, source: Path, output: Path, limit: int | None, overwrite: bool) -> dict:
    rgb_dir, depth_dir, pose_dir = prepare_dirs(output, overwrite)
    sens = source / scene_id / f"{scene_id}.sens"
    alignment = axis_alignment(source / scene_id / f"{scene_id}.txt")
    stride = 15  # ScanNet streams are nominally 30 Hz; geometry input is fixed at 2 Hz.
    selected = 0
    invalid_poses = 0
    with sens.open("rb") as handle:
        version = struct.unpack("<I", handle.read(4))[0]
        if version != 4:
            raise ValueError(f"Unexpected .sens version {version}")
        name_length = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(name_length, 1)
        intrinsic_color = skip_matrix(handle)
        skip_matrix(handle)  # extrinsic_color
        intrinsic_depth = skip_matrix(handle)
        skip_matrix(handle)  # extrinsic_depth
        color_codec, depth_codec = struct.unpack("<ii", handle.read(8))
        color_width, color_height, depth_width, depth_height = struct.unpack("<IIII", handle.read(16))
        depth_scale = struct.unpack("<f", handle.read(4))[0]
        frame_count = struct.unpack("<Q", handle.read(8))[0]
        if color_codec != 2 or depth_codec != 1:
            raise ValueError(f"Unsupported ScanNet codecs color={color_codec}, depth={depth_codec}")
        for source_index in range(frame_count):
            camera_to_world = skip_matrix(handle)
            handle.seek(16, 1)  # invalid/unused color and depth timestamps
            color_size, depth_size = struct.unpack("<QQ", handle.read(16))
            take = source_index % stride == 0 and (limit is None or selected < limit)
            if not take:
                handle.seek(color_size + depth_size, 1)
                continue
            color_data = handle.read(color_size)
            depth_data = handle.read(depth_size)
            if not np.isfinite(camera_to_world).all():
                invalid_poses += 1
                continue
            color = Image.open(io.BytesIO(color_data)).convert("RGB").resize(
                (TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS
            )
            depth = np.frombuffer(zlib.decompress(depth_data), dtype=np.uint16).reshape(
                depth_height, depth_width
            )
            if (depth_width, depth_height) != (TARGET_WIDTH, TARGET_HEIGHT):
                depth = np.asarray(
                    Image.fromarray(depth).resize(
                        (TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.NEAREST
                    )
                )
            name = f"{scene_id}_{selected:06d}"
            color.save(rgb_dir / f"{name}.jpg", quality=95)
            Image.fromarray(depth).save(depth_dir / f"{name}.png")
            # ScanNet SensorData poses consume OpenCV camera points. INHerit-SG
            # internally converts OpenCV points to OpenGL before applying the
            # supplied pose, so compensate once at this interface.
            write_pose(
                pose_dir / f"{name}.txt",
                alignment @ camera_to_world @ CV_TO_OPENGL,
            )
            selected += 1

    # ScanNet RGB and depth are both 4:3. The standard 640x480 export uses the
    # calibrated depth camera for backprojection and resizes RGB to match it.
    sx, sy = TARGET_WIDTH / depth_width, TARGET_HEIGHT / depth_height
    intrinsics = {
        "fx": float(intrinsic_depth[0, 0] * sx), "fy": float(intrinsic_depth[1, 1] * sy),
        "cx": float(intrinsic_depth[0, 2] * sx), "cy": float(intrinsic_depth[1, 2] * sy),
        "depth_scale": float(depth_scale),
    }
    return {
        "scene_id": scene_id, "source": "scannet", "source_frames": int(frame_count),
        "sample_stride": stride, "nominal_input_hz": 30.0, "output_hz": 2.0,
        "sampled_frames": selected, "image_size": [TARGET_WIDTH, TARGET_HEIGHT],
        "invalid_sampled_poses_skipped": invalid_poses,
        "intrinsics": intrinsics, "pose_space": "axis-aligned ScanNet scene coordinates",
        "input_pose_convention": "OpenGL-compatible pose for INHerit-SG",
        "axis_alignment": alignment.tolist(), "uses_vigil3d_ground_truth": False,
    }


def iter_scannetpp_depth(path: Path):
    height, width = 192, 256
    with path.open("rb") as handle:
        frame_index = 0
        while size_bytes := handle.read(4):
            size = int.from_bytes(size_bytes, "little")
            payload = handle.read(size)
            try:
                raw = lz4.block.decompress(payload, uncompressed_size=height * width * 2)
                depth = np.frombuffer(raw, dtype=np.uint16).reshape(height, width)
            except Exception:
                raw = zlib.decompress(payload, wbits=-zlib.MAX_WBITS)
                depth = (np.frombuffer(raw, dtype=np.float32).reshape(height, width) * 1000).astype(np.uint16)
            yield frame_index, depth
            frame_index += 1


def prepare_scannetpp(scene_id: str, source: Path, output: Path, limit: int | None, overwrite: bool) -> dict:
    rgb_dir, depth_dir, pose_dir = prepare_dirs(output, overwrite)
    scene = source / scene_id / "iphone"
    metadata = json.loads((scene / "pose_intrinsic_imu.json").read_text())
    names = sorted(metadata, key=lambda name: int(name.rsplit("_", 1)[1]))
    stride = 30  # Native iPhone stream is 60 Hz; geometry input is fixed at 2 Hz.
    selected_source_indices = list(range(0, len(names), stride))
    if limit is not None:
        selected_source_indices = selected_source_indices[:limit]
    selected_set = set(selected_source_indices)

    # Decode only selected RGB frames. Output numbering matches the prepared trajectory,
    # while source indices remain recorded in metadata below.
    filter_graph = f"select=not(mod(n\\,{stride})),scale={TARGET_WIDTH}:{TARGET_HEIGHT}"
    command = [
        "ffmpeg", "-v", "error", "-y", "-i", str(scene / "rgb.mkv"),
        "-vf", filter_graph, "-vsync", "vfr", "-start_number", "0",
    ]
    if limit is not None:
        command += ["-frames:v", str(limit)]
    command += [str(rgb_dir / f"{scene_id}_%06d.jpg")]
    subprocess.run(command, check=True)

    sampled = 0
    selected_intrinsics = []
    for source_index, depth in iter_scannetpp_depth(scene / "depth.bin"):
        if source_index not in selected_set:
            continue
        frame = metadata[names[source_index]]
        name = f"{scene_id}_{sampled:06d}"
        resized_depth = np.asarray(
            Image.fromarray(depth).resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.NEAREST)
        )
        Image.fromarray(resized_depth).save(depth_dir / f"{name}.png")
        write_pose(pose_dir / f"{name}.txt", np.asarray(frame["aligned_pose"], dtype=np.float64))
        intrinsic = np.asarray(frame["intrinsic"], dtype=np.float64)
        intrinsic[0] *= TARGET_WIDTH / 1920.0
        intrinsic[1] *= TARGET_HEIGHT / 1440.0
        selected_intrinsics.append(intrinsic)
        sampled += 1
        if sampled == len(selected_source_indices):
            break
    rgb_files = sorted(rgb_dir.glob("*.jpg"))
    if len(rgb_files) != sampled:
        raise RuntimeError(f"RGB/depth count mismatch for {scene_id}: {len(rgb_files)} vs {sampled}")
    median_intrinsic = np.median(np.stack(selected_intrinsics), axis=0)
    intrinsics = {
        "fx": float(median_intrinsic[0, 0]), "fy": float(median_intrinsic[1, 1]),
        "cx": float(median_intrinsic[0, 2]), "cy": float(median_intrinsic[1, 2]),
        "depth_scale": 1000.0,
    }
    return {
        "scene_id": scene_id, "source": "scannetpp", "source_frames": len(names),
        "sample_stride": stride, "nominal_input_hz": 60.0, "output_hz": 2.0,
        "sampled_frames": sampled, "source_frame_indices": selected_source_indices,
        "image_size": [TARGET_WIDTH, TARGET_HEIGHT], "intrinsics": intrinsics,
        "pose_space": "ScanNet++ aligned mesh coordinates",
        "input_pose_convention": "native ARKit/OpenGL aligned_pose", "axis_alignment": None,
        "uses_vigil3d_ground_truth": False,
    }


def main() -> None:
    args = arguments()
    for scene_id in read_scene_ids(args):
        source_name = "scannet" if scene_id.startswith("scene") else "scannetpp"
        if args.source not in ("all", source_name):
            continue
        output = args.output_root.resolve() / scene_id / "ep_000"
        if source_name == "scannet":
            report = prepare_scannet(
                scene_id, args.scannet_root.resolve(), output,
                args.max_sampled_frames, args.overwrite,
            )
        else:
            report = prepare_scannetpp(
                scene_id, args.scannetpp_root.resolve(), output,
                args.max_sampled_frames, args.overwrite,
            )
        (output / "trajectory_metadata.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
