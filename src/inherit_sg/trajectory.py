"""Load frozen HM3DSem-SQR input sequences without simulator dependencies."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def contained_path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if root not in path.parents:
        raise ValueError(f'Path must be inside its data root: {name}')
    return path


def load_scene(root, manifest, scene):
    """Verify the packaged payload and return frames, matrices, original TXT rows."""
    if manifest['schema_version'] != 1:
        raise ValueError('Unsupported trajectory manifest version')
    paths = {}
    for field in ('poses', 'frames'):
        name = scene[field]
        paths[field] = contained_path(root, name)
        if sha256(paths[field]) != manifest['sha256'][name]:
            raise ValueError(f'Trajectory checksum mismatch: {name}')
    frames = json.loads(paths['frames'].read_text())
    lines = paths['poses'].read_bytes().splitlines(keepends=True)
    if len(frames) != scene['frame_count'] or len(lines) != len(frames) or not frames:
        raise ValueError('Frame/pose count mismatch')
    poses = np.loadtxt(paths['poses'], ndmin=2).reshape(-1, 4, 4)
    if len(poses) != len(frames) or not np.isfinite(poses).all():
        raise ValueError('Invalid pose matrix values')
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-10, rtol=0):
        raise ValueError('Invalid homogeneous pose row')
    rotation = poses[:, :3, :3]
    if not (np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-6, rtol=0)
            and np.allclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError('Pose is not a rigid transform')
    for index, (frame, line) in enumerate(zip(frames, lines)):
        if frame['stem'] != f'{scene["filename_prefix"]}{index:06d}':
            raise ValueError('Unexpected filename or noncontiguous frame order')
        if Path(frame['stem']).name != frame['stem'] or '/' in frame['stem'] or '\\' in frame['stem']:
            raise ValueError('Invalid frame filename')
        if hashlib.sha256(line).hexdigest() != frame['pose_sha256']:
            raise ValueError(f'Per-frame pose checksum mismatch: {frame["stem"]}')
    return frames, poses, lines


def compare_pixels(actual, expected):
    """Compare decoded pixels, independently of PNG compression and metadata."""
    if actual.shape != expected.shape:
        raise ValueError(f'Pixel shape mismatch: {actual.shape} versus {expected.shape}')
    delta = np.abs(actual.astype(np.int64) - expected.astype(np.int64))
    return {'exact': bool(np.all(delta == 0)),
            'different_values': int(np.count_nonzero(delta)),
            'value_count': int(delta.size),
            'mean_absolute_error': float(delta.mean()),
            'max_absolute_error': int(delta.max())}


def depth_to_mm(depth):
    """Preserve capture-time truncation and uint16 conversion, without clipping."""
    return (depth * 1000).astype(np.uint16)
