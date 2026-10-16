"""Replay frozen selected poses; use --verify-only without Habitat-Sim or a GPU.

The sensor rig and observation encoding follow the project's capture script,
derived from HOV-SG's Habitat utilities; see licenses/HOV-SG.txt.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from inherit_sg.trajectory import (compare_pixels, contained_path, depth_to_mm,
                                   load_scene, sha256)


def make_simulator(asset_root, manifest, scene, gpu_device):
    import habitat_sim
    import magnum
    camera = manifest['camera']
    mesh = contained_path(asset_root, scene['asset'])
    config = contained_path(asset_root, manifest['rendering']['scene_dataset_config'])
    # Check actual readability before entering native code: some Habitat builds
    # abort rather than raise when scene files are inaccessible.
    for path in (mesh, config):
        with path.open('rb') as handle:
            handle.read(1)
    scene_hash = scene['scene_id'].split('-')[-1]
    for suffix in ('.semantic.glb', '.semantic.txt'):
        with (mesh.parent / (scene_hash + suffix)).open('rb') as handle:
            handle.read(1)
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(mesh)
    backend.scene_dataset_config_file = str(config)
    backend.gpu_device_id = gpu_device
    specs = []
    height = manifest['rendering']['sensor_height_spec_m']
    # Retain the capture rig, including the unsaved rear camera.
    for name, kind, position, orientation in (
        ('back_color_sensor', habitat_sim.SensorType.COLOR, [0, height, 1.3], [-np.pi / 8, 0, 0]),
        ('color_sensor', habitat_sim.SensorType.COLOR, [0, height, 0], [0, 0, 0]),
        ('depth_sensor', habitat_sim.SensorType.DEPTH, [0, height, 0], [0, 0, 0]),
        ('semantic', habitat_sim.SensorType.SEMANTIC, [0, height, 0], [0, 0, 0]),
    ):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type = name, kind
        spec.resolution = [camera['height'], camera['width']]
        spec.position = magnum.Vector3(*map(float, position))
        spec.orientation = magnum.Vector3(*map(float, orientation))
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.hfov = magnum.Deg(camera['hfov_degrees'])
        spec.near, spec.far = camera['near'], camera['far']
        specs.append(spec)
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = specs
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))


def render_scene(args, manifest, scene):
    import cv2
    import quaternion
    from scipy.spatial.transform import Rotation

    frames, poses, lines = load_scene(args.poses, manifest, scene)
    indices = sorted({0, len(frames) // 2, len(frames) - 1}) if args.sample else range(len(frames))
    output = args.output / scene['split'] / scene['scene_id'] / scene['episode_id']
    # Never reuse an incomplete episode or mix a sample with a full sequence.
    output.mkdir(parents=True, exist_ok=False)
    for kind in ('rgb', 'depth', 'pose') + (('semantic',) if args.semantic else ()):
        (output / kind).mkdir()
    report = {'scene_id': scene['scene_id'], 'sample': args.sample,
              'expected_full_frame_count': len(frames), 'frames': [], 'complete': False}
    mesh = contained_path(args.assets, scene['asset'])
    scene_hash = scene['scene_id'].split('-')[-1]
    assets = [mesh, contained_path(args.assets, manifest['rendering']['scene_dataset_config'])]
    assets += [mesh.parent / (scene_hash + suffix) for suffix in ('.semantic.glb', '.semantic.txt')]
    report['asset_sha256'] = {str(path.relative_to(args.assets.resolve())): sha256(path) for path in assets}
    report_path = output / 'render_report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    sim = make_simulator(args.assets, manifest, scene, args.gpu_device)
    try:
        agent = sim.initialize_agent(0)
        for index in indices:
            pose, frame = poses[index], frames[index]
            xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
            rotation = np.quaternion(xyzw[3], *xyzw[:3])
            state = agent.get_state()
            for name in ('color_sensor', 'depth_sensor', 'semantic'):
                state.sensor_states[name].position = pose[:3, 3]
                state.sensor_states[name].rotation = rotation
            state.position = pose[:3, 3]
            agent.set_state(state, reset_sensors=True, infer_sensor_states=False)
            observation = sim.get_sensor_observations(0)
            rgb_bgr = observation['color_sensor'][:, :, [2, 1, 0]]
            depth = depth_to_mm(observation['depth_sensor'])
            stem = frame['stem']
            if not cv2.imwrite(str(output / 'rgb' / (stem + '.png')), rgb_bgr):
                raise OSError(f'Failed to write RGB: {stem}')
            if not cv2.imwrite(str(output / 'depth' / (stem + '.png')), depth):
                raise OSError(f'Failed to write depth: {stem}')
            (output / 'pose' / (stem + '.txt')).write_bytes(lines[index])
            if args.semantic:
                np.save(output / 'semantic' / (stem + '.npy'), observation['semantic'], allow_pickle=False)
            result = {'index': index, 'stem': stem, 'encoded_reference_match': {}}
            for kind in ('rgb', 'depth'):
                result['encoded_reference_match'][kind] = (
                    sha256(output / kind / (stem + '.png')) == frame[kind + '_sha256'])
            if args.reference:
                for kind, actual in (('rgb', rgb_bgr), ('depth', depth)):
                    reference = args.reference / scene['scene_id'] / scene['episode_id'] / kind / (stem + '.png')
                    if sha256(reference) != frame[kind + '_sha256']:
                        raise ValueError(f'Reference is not the frozen capture input: {reference}')
                    expected = cv2.imread(str(reference), cv2.IMREAD_UNCHANGED)
                    if expected is None:
                        raise ValueError(f'Cannot decode reference: {reference}')
                    result[kind] = compare_pixels(actual, expected)
            report['frames'].append(result)
            if (index + 1) % 100 == 0 or args.sample:
                print(f'{scene["scene_id"]}: rendered {index + 1}/{len(frames)}', flush=True)
        report['complete'] = not args.sample
    finally:
        sim.close()
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poses', type=Path, default=ROOT / 'data/poses')
    parser.add_argument('--assets', type=Path, help='HM3D v0.2 root containing val/ and the annotated config')
    parser.add_argument('--output', type=Path, help='New output root; scenes go under val/<scene>/ep_000')
    parser.add_argument('--scene', action='append', help='Repeat to select scenes; default: all 36')
    parser.add_argument('--sample', action='store_true', help='Only first/middle/last frames; not a mapping input')
    parser.add_argument('--semantic', action='store_true', help='Also save per-frame semantic arrays')
    parser.add_argument('--reference', type=Path, help='Optional original val/ RGB-D root for decoded-pixel comparison')
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--verify-only', action='store_true', help='Verify all selected poses/hashes without rendering')
    args = parser.parse_args()
    manifest = json.loads((args.poses / 'manifest.json').read_text())
    scenes = manifest['scenes']
    known = {scene['scene_id'] for scene in scenes}
    if args.scene:
        if set(args.scene) - known:
            parser.error(f'Unknown scene IDs: {sorted(set(args.scene) - known)}')
        scenes = [scene for scene in scenes if scene['scene_id'] in args.scene]
    for scene in scenes:
        load_scene(args.poses, manifest, scene)
    print(f'Verified {len(scenes)} scenes, {sum(s["frame_count"] for s in scenes)} frozen poses', flush=True)
    if args.verify_only:
        return 0
    if args.assets is None or args.output is None:
        parser.error('--assets and --output are required for rendering')
    os.environ.setdefault('MAGNUM_LOG', 'quiet')
    os.environ.setdefault('HABITAT_SIM_LOG', 'quiet')
    import cv2
    import habitat_sim
    import scipy
    expected_version = manifest['rendering']['habitat_sim_version']
    if habitat_sim.__version__ != expected_version:
        parser.error(f'Replay requires habitat-sim {expected_version}; got {habitat_sim.__version__}')
    # Refuse all existing episodes before starting a multi-scene run.
    for scene in scenes:
        output = args.output / scene['split'] / scene['scene_id'] / scene['episode_id']
        if output.exists():
            parser.error(f'Output episode already exists: {output}; choose a fresh output root')
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / 'replay_summary.json'
    if summary_path.exists():
        parser.error(f'Replay summary already exists: {summary_path}; choose a fresh output root')
    summary = {'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'habitat_sim': habitat_sim.__version__, 'opencv': cv2.__version__,
                              'scipy': scipy.__version__},
               'manifest_sha256': sha256(args.poses / 'manifest.json'),
               'reference_comparison': bool(args.reference), 'scenes': []}
    try:
        for scene in scenes:
            summary['scenes'].append(render_scene(args, manifest, scene))
    finally:
        summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    if args.reference and any(not row[kind]['exact'] for scene in summary['scenes']
                              for row in scene['frames'] for kind in ('rgb', 'depth')):
        print('Replay differs from the original pixels; see replay_summary.json', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
