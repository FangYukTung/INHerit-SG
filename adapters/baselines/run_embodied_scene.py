#!/usr/bin/env python3
"""Build/query the fair-adapted Embodied-RAG graph for one INHerit-SG map."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HELPERS = Path(__file__).resolve().parent / "embodied"


def find_inherit_run(root, scene):
    root = Path(root).resolve()
    for candidate in (root / scene / "ep_000", root / scene, root):
        if (candidate / "memory_kb.json").is_file():
            return candidate
    paths = sorted(Path(root).glob(f"run_*/{scene}/ep_000"), key=lambda path: path.stat().st_mtime)
    return paths[-1] if paths else None


def run(command, env, code_root):
    subprocess.run(command, cwd=code_root, env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--inherit-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    code_root = os.environ.get("EMBODIED_RAG_ROOT")
    if not code_root:
        parser.error("Set EMBODIED_RAG_ROOT to the upstream Embodied-RAG checkout")
    code = Path(code_root).resolve()
    if not code.is_dir():
        parser.error(f"EMBODIED_RAG_ROOT does not exist: {code}")
    inherit = find_inherit_run(args.inherit_root, args.scene_id)
    if not inherit or not (inherit / "memory_kb.json").exists():
        raise RuntimeError(f"No complete INHerit-SG observations for {args.scene_id}")
    out = Path(args.output_root).resolve() / args.scene_id
    out.mkdir(parents=True, exist_ok=True)
    observations, object_graph, forest = out / "observations.jsonl", out / "object.gml", out / "forest.gml"
    env = os.environ.copy()
    if not env.get("OPENAI_API_KEY") and env.get("INHERIT_API_KEY"):
        env["OPENAI_API_KEY"] = env["INHERIT_API_KEY"]
    if not env.get("OPENAI_API_BASE") and env.get("INHERIT_API_BASE"):
        env["OPENAI_API_BASE"] = env["INHERIT_API_BASE"]
    if not env.get("OPENAI_API_BASE"):
        parser.error("Set OPENAI_API_BASE or INHERIT_API_BASE for Embodied-RAG")
    env.setdefault("OPENAI_BASE_URL", env["OPENAI_API_BASE"])
    env.setdefault("HM3D_SQR_ROOT", str(ROOT / "data/reviewed"))
    env.setdefault("EMBODIED_RAG_MODEL", "qwen3.6-flash")
    max_cost = args.max_cost_usd
    if max_cost is None:
        max_cost = float(env.get("EMBODIED_RAG_MAX_COST_USD", "2.0"))
    env.update({
        "EMBODIED_RAG_ENCODER": args.encoder,
        "EMBODIED_RAG_USAGE_LOG": str((Path(args.output_root) / "model_usage.jsonl").resolve()),
        "EMBODIED_RAG_MAX_COST_USD": str(max_cost),
        "INHERIT_SG_EXPERIMENT_RUN_ID": args.run_id,
        "PYTHONPATH": str(HELPERS) + os.pathsep + str(code) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
    })
    if not forest.exists():
        run([sys.executable, str(HELPERS / "shared_observations.py"), "--input", str(inherit / "memory_kb.json"),
             "--scene-id", args.scene_id, "--output", str(observations)], env, code)
        run([sys.executable, str(HELPERS / "build_observation_graph.py"), "--input", str(observations),
             "--output", str(object_graph), "--encoder", args.encoder], env, code)
        run([sys.executable, str(HELPERS / "generate_semantic_forest.py"), "--input", str(object_graph),
             "--output", str(forest)], env, code)
    run([sys.executable, str(HELPERS / "evaluate_scene.py"), "--scene_id", args.scene_id, "--graph_path", str(forest),
         "--output_dir", str(out / "evaluation"), "--mode", "llm", "--gpu", "cpu",
         "--manifest", str(Path(args.manifest).resolve()), "--predictions-jsonl", str(Path(args.predictions).resolve()),
         "--encoder", args.encoder], env, code)


if __name__ == "__main__":
    main()
