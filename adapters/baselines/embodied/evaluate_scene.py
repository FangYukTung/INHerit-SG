import json
import os
import cv2
import numpy as np
import networkx as nx
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# 引用你的环境包
from embodied_rag_core.embodied_rag.embodied_retriever import EmbodiedRetriever, RetrievalMethod

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, np.bool_): return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def load_scene_gt(scene_path):
    if not os.path.exists(scene_path): return {}
    with open(scene_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt_positions = {}
    registry = data.get('objects_registry', [])
    if not registry and 'object_registry' in data: registry = data['object_registry']
    for item in registry:
        obj_id = item.get('id')
        pos = item.get('obb_center', item.get('aabb_center'))
        if obj_id is not None and pos:
            gt_positions[int(obj_id)] = np.array(pos)
    return gt_positions


def load_manifest_rows(path, scene_id):
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = payload.get('queries', payload) if isinstance(payload, dict) else payload
    return {
        str(row['query_id']): row for row in rows
        if row.get('scene_id') == scene_id and row.get('episode_id', 'ep_000') == 'ep_000'
    }


def node_position(node_data):
    """Read both the adapted flat schema and the upstream nested schema."""
    pos = node_data.get('position')
    if isinstance(pos, dict):
        return np.array([pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)], dtype=float)
    if isinstance(pos, (list, tuple)) and len(pos) >= 3:
        return np.asarray(pos[:3], dtype=float)
    return np.array([
        node_data.get('x', 0), node_data.get('y', 0), node_data.get('z', 0)
    ], dtype=float)

def draw_bbox_and_save(image_path, bbox, save_path, label, color=(0, 255, 0)):
    if not image_path or not os.path.exists(image_path): return
    img = cv2.imread(image_path)
    if img is None: return
    try:
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, label, (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(save_path, img)
    except: pass

async def run_eval(args):
    SCENE_FULL = args.scene_id
    SCENE_SHORT = SCENE_FULL.split('-')[0]
    EP = "ep_000"

    BASE_DATA_DIR = os.getenv("HM3D_SQR_ROOT", "/path/to/HM3DSem-SQR/val")
    QUERY_PATH = f"{BASE_DATA_DIR}/{SCENE_FULL}/{EP}/scene_{SCENE_SHORT}_{EP}_queries.json"
    SCENE_GT_PATH = f"{BASE_DATA_DIR}/{SCENE_FULL}/{EP}/scene_{SCENE_SHORT}_{EP}.json"

    OUTPUT_DIR = args.output_dir
    GRAPH_PATH = args.graph_path
    DEVICE = args.gpu # 接收 GPU 参数

    if args.mode == 'semantic':
        RETRIEVAL_MODE = RetrievalMethod.SEMANTIC
        mode_suffix = "semantic"
    else:
        RETRIEVAL_MODE = RetrievalMethod.LLM_HIERARCHICAL
        mode_suffix = "llm"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    viz_dir = os.path.join(OUTPUT_DIR, f"viz_{SCENE_SHORT}_{mode_suffix}")
    os.makedirs(viz_dir, exist_ok=True)

    log_path = os.path.join(OUTPUT_DIR, f"eval_{SCENE_SHORT}_{mode_suffix}_{timestamp}.log")
    json_path = os.path.join(OUTPUT_DIR, f"res_{SCENE_SHORT}_{mode_suffix}_{timestamp}.json")

    logger = logging.getLogger(f'exp_{SCENE_SHORT}_{mode_suffix}')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)

    logger.info(f"Scene: {SCENE_FULL}, Mode: {args.mode}, GPU: {DEVICE}")

    if not os.path.exists(GRAPH_PATH):
        logger.error(f"Graph file not found: {GRAPH_PATH}")
        raise FileNotFoundError(f"Graph file not found: {GRAPH_PATH}")

    G = nx.read_gml(GRAPH_PATH)

    # The paper's LLM-guided traversal does not use the embedding model.
    # Load it only for the separate semantic-retrieval mode.
    encoder = None
    if RETRIEVAL_MODE == RetrievalMethod.SEMANTIC:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading SentenceTransformer on {DEVICE}...")
        encoder = SentenceTransformer(args.encoder, device=DEVICE)

    async def embedding_func(texts):
        if encoder is None:
            raise RuntimeError("Embedding function called during LLM hierarchical retrieval")
        return encoder.encode(texts)

    retriever = EmbodiedRetriever(
        graph=G,
        embedding_func=embedding_func,
        retrieval_method=RETRIEVAL_MODE
    )

    gt_map = load_scene_gt(SCENE_GT_PATH)
    if not os.path.exists(QUERY_PATH):
        raise FileNotFoundError(f"Query file not found: {QUERY_PATH}")

    with open(QUERY_PATH, 'r') as f:
        queries = json.load(f)
    manifest_rows = load_manifest_rows(args.manifest, SCENE_FULL)
    if manifest_rows is not None:
        queries = [q for q in queries if str(q.get('meta', {}).get('query_id')) in manifest_rows]
    if args.max_queries is not None:
        queries = queries[:args.max_queries]

    top1_c, top5_c, total = 0, 0, 0
    results_log = []

    for idx, q_item in enumerate(tqdm(queries, desc=f"{SCENE_SHORT} [{args.mode}]", position=0, leave=False)):
        try:
            query_id = str(q_item.get('meta', {}).get('query_id', idx))
            query_text = q_item['query']['text']
            gt_ids = q_item['ground_truth']['object_ids']
        except: continue

        total += 1
        retrieved_nodes = await retriever.retrieve(query_text, top_k=5)

        is_hit_top1 = False
        is_hit_top5 = False
        hit_details = []

        for rank, node_id in enumerate(retrieved_nodes):
            if node_id not in G.nodes: continue
            node_data = G.nodes[node_id]
            p_vec = node_position(node_data)

            min_dist = float('inf')
            for gt_id in gt_ids:
                if int(gt_id) in gt_map:
                    dist = np.linalg.norm(p_vec - gt_map[int(gt_id)])
                    if dist < min_dist: min_dist = dist

            node_hit = (min_dist < 1.0) and node_data.get('level') == 0

            if node_hit:
                if rank == 0: is_hit_top1 = True
                if rank < 5: is_hit_top5 = True

            hit_details.append({"rank": rank, "id": node_id, "dist": min_dist, "hit": node_hit})

            if rank == 0 or node_hit:
                img_p = node_data.get('full_image_path')
                if img_p:
                    fname = f"q{idx}_r{rank}_{'HIT' if node_hit else 'MISS'}.jpg"
                    draw_bbox_and_save(img_p, node_data.get('bbox'), os.path.join(viz_dir, fname), f"{min_dist:.1f}m")

        if is_hit_top1: top1_c += 1
        if is_hit_top5: top5_c += 1

        results_log.append({
            "key": (manifest_rows or {}).get(query_id, {}).get("key"),
            "qid": query_id, "query": query_text, "gt": gt_ids,
            "top1": is_hit_top1, "top5": is_hit_top5, "details": hit_details
        })

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_log, f, indent=2, cls=NumpyEncoder)

    if args.predictions_jsonl:
        prediction_path = Path(args.predictions_jsonl)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_path.open("w", encoding="utf-8") as handle:
            for result in results_log:
                if not result.get("key"):
                    raise ValueError(f"Missing manifest key for query {result['qid']}")
                candidates = []
                for detail in result["details"]:
                    node_id = detail["id"]
                    if node_id not in G.nodes or G.nodes[node_id].get("level") != 0:
                        continue
                    node = G.nodes[node_id]
                    candidates.append({
                        "id": node_id,
                        "label": node.get("name", "unknown"),
                        "position": node_position(node).tolist(),
                        "score": None,
                    })
                record = {
                    "schema_version": 1,
                    "method": f"embodied_rag_fair_adapted_{mode_suffix}",
                    "key": result["key"],
                    "query_id": result["qid"],
                    "scene_id": SCENE_FULL,
                    "prediction": candidates[0] if candidates else None,
                    "candidates": candidates[:5],
                    "shared_observation_fields": ["category", "caption", "position", "timestamp"],
                }
                handle.write(json.dumps(record, cls=NumpyEncoder) + "\n")

    if total == 0:
        raise RuntimeError(f"No eligible queries for {SCENE_FULL}")
    msg = f"Scene {SCENE_SHORT} [{args.mode}]: Top1={top1_c/total:.3f}, Top5={top5_c/total:.3f}"
    print(msg)
    logger.info(msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id', required=True)
    parser.add_argument('--graph_path', required=True)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--mode', choices=['semantic', 'llm'], required=True)
    parser.add_argument('--gpu', default='cpu', help='Device for models (e.g. cuda:0)')
    parser.add_argument('--manifest', help='Area-free Main/Core manifest JSON')
    parser.add_argument('--max-queries', type=int, help='Maximum queries to process')
    parser.add_argument('--predictions-jsonl', help='Unified strict-evaluator candidate output')
    parser.add_argument(
        '--encoder',
        default=os.getenv('EMBODIED_RAG_ENCODER', 'all-MiniLM-L6-v2'),
        help='SentenceTransformer name or frozen local snapshot path',
    )
    args = parser.parse_args()

    asyncio.run(run_eval(args))
