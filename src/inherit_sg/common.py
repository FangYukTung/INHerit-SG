import os
import sys
import json
import csv
import yaml
import base64
import time
import cv2
import numpy as np
import torch
import re
import mimetypes
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from datetime import datetime
from .model_router import ModelRouter

# --- Logger (保持不变) ---
class Logger:
    def __init__(self, log_file_path: str):
        # [修改] 使用 sys.__stdout__ 确保永远指向原始控制台
        # 这样无论你实例化多少个 Logger，它们都是并列关系，而不是嵌套关系
        self.terminal = sys.__stdout__

        self.log_file = open(log_file_path, 'w', encoding='utf-8')

    def write(self, message: str):
        self.terminal.write(message)
        self.log_file.write(message)
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    def __del__(self):
        if self.log_file and not self.log_file.closed:
            self.log_file.close()

def read_pose_matrix(pose_path: str) -> np.ndarray:
    """Read a finite camera-to-world pose; never silently invent a pose."""
    with open(pose_path) as handle:
        pose = np.asarray([float(x) for x in handle.read().split()]).reshape(4, 4)
    if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError(f"Invalid camera-to-world pose: {pose_path}")
    return pose

def pixel_to_world(u: int, v: int, d: float, pose: np.ndarray, intrinsics: Dict) -> List[float]:
    """
    将像素坐标转换到世界坐标，并修正 OpenCV 到 HM3D 的坐标系差异。
    """
    if d <= 0: return None

    # 1. 图像坐标 -> OpenCV 相机坐标 (P_cam_cv)
    # (+Z 前, +X 右, +Y 下)
    z_cam = d
    x_cam = (u - intrinsics['cx']) * z_cam / intrinsics['fx']
    y_cam = (v - intrinsics['cy']) * z_cam / intrinsics['fy']

    p_cam_cv = np.array([x_cam, y_cam, z_cam, 1.0])

    # 2. 坐标系矫正 (OpenCV -> Habitat/GL)
    # 翻转 Y 和 Z 轴: (x, y, z) -> (x, -y, -z)
    # 这样 +Z(前) 变成了 -Z(前)，+Y(下) 变成了 -Y(下=上) -> 也就是 +Y(上)
    T_fix = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])

    p_cam_habitat = T_fix @ p_cam_cv

    # 3. 相机坐标 -> 世界坐标
    p_world = pose @ p_cam_habitat

    return p_world[:3].tolist()



def extract_area_base_id(node_id):
    """
    提取区域的基础 ID（去掉时间戳部分）

    Examples:
        area_1_0_17118  -> area_1_0
        area_5_2_37424  -> area_5_2
        room_5          -> room_5 (保持不变)
        fireplace_0     -> fireplace_0 (保持不变)
    """
    # 匹配 area_{room}_{cluster}_{timestamp}
    match = re.match(r'(area_\d+_\d+)_\d+', node_id)
    if match:
        return match.group(1)  # 返回 area_1_0
    return node_id  # 其他类型节点保持原样

def process_nodes_file(input_path, output_path=None):
    """
    处理 nodes.jsonl，去重逻辑：
    1. Room/Floor/Object: 按完整 node_id 去重
    2. Area: 按 area_{room}_{cluster} 去重（忽略时间戳后缀）
    """
    if output_path is None:
        output_path = input_path.replace('.jsonl', '_final.jsonl')

    # 分类存储
    nodes_by_base_id = defaultdict(list)

    print(f"Reading nodes from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                node = json.loads(line)
                node_id = node.get('node_id')

                if not node_id:
                    print(f"Warning: Line {line_num} missing node_id")
                    continue

                # 提取基础 ID
                base_id = extract_area_base_id(node_id)
                nodes_by_base_id[base_id].append(node)

            except json.JSONDecodeError as e:
                print(f"Error: Line {line_num} - {e}")
                continue

    # 统计
    total_lines = sum(len(versions) for versions in nodes_by_base_id.values())
    print(f"\n=== Original Data ===")
    print(f"Total lines: {total_lines}")
    print(f"Unique base IDs: {len(nodes_by_base_id)}")

    # 分析重复情况
    duplicates = {k: v for k, v in nodes_by_base_id.items() if len(v) > 1}
    if duplicates:
        print(f"\nNodes with multiple versions: {len(duplicates)}")

        # 按类型分类
        by_type = defaultdict(list)
        for base_id, versions in duplicates.items():
            if base_id.startswith('area_'):
                by_type['Area'].append((base_id, len(versions)))
            elif base_id.startswith('room_'):
                by_type['Room'].append((base_id, len(versions)))
            elif base_id.startswith('floor_'):
                by_type['Floor'].append((base_id, len(versions)))
            else:
                by_type['Object'].append((base_id, len(versions)))

        for type_name, items in sorted(by_type.items()):
            print(f"\n  {type_name} duplicates: {len(items)}")
            # 显示前3个示例
            for base_id, count in sorted(items, key=lambda x: -x[1])[:3]:
                print(f"    {base_id}: {count} versions")

    # 保留最新版本
    final_nodes = {}

    for base_id, versions in nodes_by_base_id.items():
        if len(versions) == 1:
            final_nodes[base_id] = versions[0]
        else:
            # 按时间戳排序
            sorted_versions = sorted(
                versions,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )

            final_node = sorted_versions[0]
            final_nodes[base_id] = final_node

            # 打印示例（Area 类型）
            if base_id.startswith('area_') and len(final_nodes) <= 5:
                print(f"\n'{base_id}' merged {len(versions)} versions:")
                for v in versions:
                    print(f"  - {v['node_id']} ({v.get('timestamp', 'N/A')})")
                print(f"  ✅ Kept: {final_node['node_id']}")

    # 统计最终结果
    by_level = defaultdict(list)
    for node in final_nodes.values():
        level = node.get('node_level', -1)
        by_level[level].append(node)

    # 写入文件
    print(f"\n=== Writing Results ===")
    with open(output_path, 'w', encoding='utf-8') as f:
        for level in sorted(by_level.keys()):
            nodes = sorted(by_level[level], key=lambda x: x.get('node_id', ''))
            for node in nodes:
                f.write(json.dumps(node, ensure_ascii=False) + '\n')

    # 输出统计
    print(f"\n=== Final Statistics ===")
    level_names = {0: 'Floors', 1: 'Rooms', 2: 'Areas', 3: 'Objects'}
    for level in sorted(by_level.keys()):
        count = len(by_level[level])
        name = level_names.get(level, f'Level-{level}')
        print(f"  {name}: {count}")

    reduction = total_lines - len(final_nodes)
    print(f"\nRemoved {reduction} duplicates ({reduction/total_lines*100:.1f}%)")
    print(f"Output: {output_path}")

    return final_nodes


def prune_relations_file(edges_path, final_nodes):
    """Keep only unique relations whose endpoints survived node versioning."""
    valid_ids = {node["node_id"] for node in final_nodes.values()}
    kept = []
    seen = set()
    removed = 0
    with open(edges_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            edge = (
                row.get("child_node_id", ""),
                row.get("parent_node_id", ""),
                row.get("node_level", ""),
            )
            if edge[0] not in valid_ids or edge[1] not in valid_ids:
                removed += 1
                continue
            if edge in seen:
                removed += 1
                continue
            seen.add(edge)
            kept.append(edge)
    with open(edges_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["child_node_id", "parent_node_id", "node_level"])
        writer.writerows(kept)
    print(f"  [Save] Relations normalized: kept={len(kept)}, removed={removed}")
    return {"kept": len(kept), "removed": removed}


def analyze_area_quality(nodes_dict):
    """分析区域聚类质量"""
    areas = [n for n in nodes_dict.values() if n.get('node_level') == 2]

    if not areas:
        return

    print(f"\n=== Area Quality Analysis ===")

    obj_counts = [len(a.get('contained_obj_nodes', [])) for a in areas]

    print(f"Total areas: {len(areas)}")
    print(f"Objects per area: min={min(obj_counts)}, max={max(obj_counts)}, avg={sum(obj_counts)/len(obj_counts):.1f}")

    # 单物体区域
    singles = [a for a in areas if len(a.get('contained_obj_nodes', [])) == 1]
    if singles:
        pct = len(singles) / len(areas) * 100
        print(f"\n⚠️  Single-object areas: {len(singles)} ({pct:.1f}%)")
        if pct > 30:
            print(f"    Suggestion: Increase DIST_THRESH from 0.8 to 1.2-1.5 meters")

    # 按房间统计
    by_room = defaultdict(list)
    for a in areas:
        room = a.get('parent_room_id', 'unknown')
        by_room[room].append(a)

    print(f"\nAreas per room:")
    for room, areas_list in sorted(by_room.items()):
        avg_obj = sum(len(a.get('contained_obj_nodes', [])) for a in areas_list) / len(areas_list)
        print(f"  {room}: {len(areas_list)} areas, avg {avg_obj:.1f} objects/area")




class LLMHandler:
    def __init__(self, llm_config: Dict[str, Any]):
        self.config = llm_config
        self.mode = llm_config.get('deployment_type', 'api')
        self.clients = {}
        self.model_names = {}

        # --- [修改] 读取空间推理配置 ---
        self.spatial_config = llm_config.get('api_settings', {}) # 或者从 root config 传进来，这里假设在 llm_config 中
        # 注意：通常 spatial_reasoning 在 root config，建议在初始化时传入或从 self.config 寻找
        # 这里为了兼容性，我们在 detect_spatial_relationships 中动态判断，
        # 或者你可以在 TrajectoryPipeline 初始化 LLMHandler 时传入 method。
        # 这里我们假设 config 结构中有一个 explicit 的开关，如果没有则默认为 vlm
        self.spatial_method = "vlm" # 默认值

        print(f"\n--- Initializing LLM Handler (Mode: {self.mode.upper()}) ---")
        router_config = llm_config.get('model_router')
        if router_config:
            self.router = ModelRouter(router_config)
            self.routing = router_config.get('routing', {})
            self.clients = self.router.endpoints
            self.model_names = self.router.models
            self.mode = 'routed'
            for name, model in self.model_names.items():
                print(f"  - Routed Endpoint: {name} -> {model}")
        elif self.mode == 'local':
            local_cfg = llm_config.get('local_settings', {})
            endpoints = local_cfg.get('endpoints', {})
            self.routing = local_cfg.get('routing', {})
            for name, cfg in endpoints.items():
                print(f"  - Loading Local Endpoint: {name} -> {cfg['model_name']}")
                self.clients[name] = OpenAI(api_key=cfg.get('api_key', "EMPTY"), base_url=cfg['api_base_url'])
                self.model_names[name] = cfg['model_name']
        else:
            api_cfg = llm_config.get('api_settings', {})
            print(f"  - Loading API Endpoint: {api_cfg['vision_model']}")
            api_key = os.environ.get("INHERIT_SG_API_KEY", api_cfg.get('api_key', ''))
            if not api_key:
                raise ValueError("Set INHERIT_SG_API_KEY before using API deployment mode")
            client = OpenAI(api_key=api_key, base_url=api_cfg['api_base_url'])
            self.clients['default'] = client
            self.api_vision_model = api_cfg['vision_model']
            self.api_text_model = api_cfg['text_model']
            self.temperature = api_cfg.get('temperature', 0.0)
    # --- [新增] Token 统计初始化 ---
        self.token_stats = {
            "prompt_tokens": 0,      # 输入 (提示词 + 图片)
            "completion_tokens": 0,  # 输出 (模型回答)
            "total_tokens": 0        # 总计
        }
    # --- [新增] 更新 Token 使用量 ---
    def _update_token_stats(self, response):
        """从 API 响应中提取 usage 信息并累加"""
        try:
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                self.token_stats["prompt_tokens"] += usage.prompt_tokens
                self.token_stats["completion_tokens"] += usage.completion_tokens
                self.token_stats["total_tokens"] += usage.total_tokens

                # 可选：打印单次消耗日志
                # print(f"   [Token Cost] In: {usage.prompt_tokens}, Out: {usage.completion_tokens}")
        except Exception as e:
            print(f"[Warning] Failed to track tokens: {e}")

    # --- [新增] 获取当前统计结果 ---
    def get_token_report(self):
        return self.token_stats
    def _get_client_and_model(self, task_type: str):
        # 视觉任务列表
        vision_tasks = [
            'leaf_description',
            'spatial_reasoning',
            'cluster_summary',
            'scene_summary',
            'object_expansion',
            'verification',
            'map_decision'
        ]

        # [新增] 纯文本任务列表 (显式定义以便本地部署路由)
        text_tasks = [
            'tag_filtering',
            'query_parsing'
        ]

        if self.mode == 'routed':
            client, model_name = self.router.resolve(task_type)
            if task_type in ('query_parsing', 'tag_filtering'):
                temperature = 0.0
            elif task_type == 'cluster_summary':
                temperature = 0.3
            else:
                temperature = 0.1
            return client, model_name, temperature
        if self.mode == 'local':
            # 1. 尝试直接获取配置的路由
            endpoint_name = self.routing.get(task_type)

            # 2. 如果没配置，且是视觉任务，尝试借用其他视觉任务的路由
            if not endpoint_name and task_type in vision_tasks:
                endpoint_name = self.routing.get('spatial_reasoning', self.routing.get('leaf_description'))

            # 3. [新增] 如果是文本任务，优先回退到 primary 或 text 专用节点
            if not endpoint_name and task_type in text_tasks:
                endpoint_name = 'primary'

            # 4. 回退兜底
            if not endpoint_name:
                endpoint_name = 'primary'

            if endpoint_name not in self.clients:
                endpoint_name = list(self.clients.keys())[0]

            return (self.clients[endpoint_name], self.model_names[endpoint_name], 0.1)
        else:
            # API 模式：根据任务类型强制选择 Vision 或 Text 模型
            if task_type in vision_tasks:
                model_name = self.api_vision_model
            else:
                # tag_filtering 走这里 (Text Model)
                model_name = self.api_text_model

            # 针对不同任务微调 temperature
            if task_type == 'cluster_summary': temp = 0.3
            elif task_type == 'map_decision': temp = 0.1
            elif task_type == 'tag_filtering': temp = 0.0 # 过滤任务需要极度确定性
            elif task_type == 'query_parsing': temp = 0.0 # 解析任务需要极度确定性
            else: temp = self.temperature

            return self.clients['default'], model_name, temp

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _image_data_url(self, image_path: str) -> str:
        mime = mimetypes.guess_type(image_path)[0] or 'image/jpeg'
        return f"data:{mime};base64,{self._encode_image(image_path)}"

    def filter_object_tags(self, raw_tags: List[str]) -> List[str]:
        """
        使用 LLM 过滤 RAM++ 生成的标签。
        目标：只保留具体的物体名词，去除场景、动作、同义词。
        """
        if not raw_tags:
            return []

        client, model_name, temp = self._get_client_and_model('tag_filtering')

        tags_str = ", ".join(raw_tags)

        prompt_text = (
            f"Task: Filter image tags for object detection.\n"
            f"Input List: [{tags_str}]\n"
            f"Requirements:\n"
            f"1. KEEP specific physical objects (e.g., 'chair', 'sofa', 'lamp').\n"
            f"2. REMOVE scene/room names (e.g., 'living room', 'kitchen', 'bedroom', 'room', 'floor', 'ceiling', 'wall').\n"
            f"3. REMOVE actions, states, or lighting (e.g., 'burn', 'shiny', 'cluttered', 'bright').\n"
            f"4. REMOVE building structures unless removable (remove 'beam', 'pillar', 'hardwood').\n"
            f"5. MERGE synonyms (e.g., 'couch' & 'sofa' -> keep 'sofa'; 'appliance' & 'home appliance' -> keep 'appliance';'wood floor' & 'floor' -> remove).\n"
            f"6. Output strictly a JSON list of strings.\n"
            f"Example Input: ['living room', 'chair', 'seat', 'bright', 'floor','white']\n"
            f"Example Output: [\"chair\"]\n"
        )

        try:
            # 对于 API 模式，通常不需要 image_url，因为这是纯文本任务
            # 但为了保持接口一致性，如果你的 API 必须接受 vision 格式的消息结构也没问题
            # 这里使用标准的 chat completion 格式
            messages = [{"role": "user", "content": prompt_text}]

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=200,
                temperature=temp
            )

            content = response.choices[0].message.content.strip()

            # 解析 JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```", 1)[1].split("```", 1)[0].strip()

            # 处理可能的方括号丢失
            if not content.startswith("["): content = "[" + content
            if not content.endswith("]"): content = content + "]"

            filtered_tags = json.loads(content)

            # 二次保障：确保全是字符串且不为空
            final_tags = [str(t).lower() for t in filtered_tags if isinstance(t, str) and t.strip()]

            print(f"  [LLM Filter] {len(raw_tags)} -> {len(final_tags)} tags. Removed: {set(raw_tags) - set(final_tags)}")
            return final_tags

        except Exception as e:
            print(f"  [Error] Tag filtering failed: {e}. Using raw unique tags.")
            return list(set(raw_tags)) # 兜底：仅简单的去重

    def generate_leaf_node_semantics(self, image_path: str, tag: str, spatial_info: str = "") -> str:
        client, model_name, temp = self._get_client_and_model('leaf_description')
        image_url = self._image_data_url(image_path)
        prompt_text = f"任务：简要描述物体。\n物体：{tag}\n位置：{spatial_info}\n要求：包含颜色、材质、状态。50字以, 必须使用英文。"
        try:
            response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": image_url}}]}], max_tokens=100, temperature=temp)
            self._update_token_stats(response)  # <--- [新增] 插入在这里
            return response.choices[0].message.content.strip()
        except Exception as e: return f"描述生成失败: {tag}"

    # --- [修改] 主函数：支持模式切换 ---
    def detect_spatial_relationships(self, image_path: str, object_list: List[str], existing_relations: List[str],
                                     object_positions: Dict[str, List[float]] = None, method: str = 'vlm') -> List[List[Any]]:
        """
        Args:
            method: 'vlm' (默认) 或 'geometric'
            object_positions: 仅在 method='geometric' 时需要，{node_id: [x, y, z]}
        """
        # 1. 几何模式
        if method == 'geometric':
            if object_positions is None:
                print("[LLMHandler] Warning: Geometric mode selected but no positions provided.")
                return []
            return self._geometric_spatial_reasoning(object_list, object_positions)

        # 2. VLM 模式 (原有逻辑)
        client, model_name, temp = self._get_client_and_model('spatial_reasoning')
        base64_image = self._encode_image(image_path)
        obj_str = ", ".join(object_list)
        history_str = "; ".join(existing_relations) if existing_relations else "None"

        prompt_text = f"任务：分析物理空间关系。\n物体列表：[{obj_str}]\n已知关系：[{history_str}]\n要求：\n1. 输出 JSON 列表: [Source, Target, Relation, Weight(0.0-1.0)], 且必须是列表的列表形式。\n2. 只允许使用以下关系词: on, under, next_to, inside, near, beside, above, below, left_of, right_of.\n3. 权重: >0.8(接触/支撑), 0.6-0.8(紧邻), <0.6(松散).\n4. 尽量全面, 但是不能强加关系, 如果距离较远的物体则不要添加关系; 重点关注标签不是object的物品, 但是标签是object的物体也要适当关注。"

        try:
            response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}], max_tokens=600, temperature=temp)
            self._update_token_stats(response)  # <--- [新增] 插入在这里
            content = response.choices[0].message.content.strip()
            if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: content = content.split("```", 1)[1].split("```", 1)[0].strip()
            return json.loads(content)
        except Exception: return []

    # --- [新增] 纯几何空间关系计算 ---
    def _geometric_spatial_reasoning(self, object_list: List[str], object_positions: Dict[str, np.ndarray]) -> List[List[Any]]:
        """
        基于 3D 坐标 (Centroid) 计算空间关系。
        Args:
            object_list: 节点 ID 列表
            object_positions: {node_id: [x, y, z]} 字典
        Returns:
            [[Source, Target, Relation, Weight], ...]
        """
        relationships = []
        if not object_positions:
            return relationships

        # 阈值参数 (单位: 米)
        VERTICAL_THRESH = 0.15      # 垂直判定阈值
        HORIZONTAL_ON_THRESH = 0.5  # 判定为 on/under 的水平最大偏差
        NEAR_THRESH = 1.5           # 判定为 next_to 的最大距离

        ids = [oid for oid in object_list if oid in object_positions]

        for i in range(len(ids)):
            for j in range(len(ids)):
                if i == j: continue

                u, v = ids[i], ids[j]
                pos_u = np.array(object_positions[u])
                pos_v = np.array(object_positions[v])

                # Habitat 坐标系通常: Y 是高度 (Up)
                # 计算差值
                diff = pos_u - pos_v
                dx, dy, dz = diff[0], diff[1], diff[2]

                dist_xz = np.sqrt(dx**2 + dz**2) # 水平距离
                dist_3d = np.linalg.norm(diff)

                rel = None
                weight = 0.0

                # 1. 垂直关系 (On / Above / Under / Below)
                if dist_xz < HORIZONTAL_ON_THRESH:
                    if dy > VERTICAL_THRESH:
                        # u 比 v 高 -> u on v (or above)
                        rel = "on" if dy < 1.0 else "above"
                        weight = 0.9 if rel == "on" else 0.7
                    elif dy < -VERTICAL_THRESH:
                        # u 比 v 低 -> u under v (or below)
                        # 通常我们关注支撑关系，under 比较少见除非是桌子底下
                        rel = "under" if abs(dy) < 1.0 else "below"
                        weight = 0.9 if rel == "under" else 0.7

                # 2. 水平关系 (Next to / Near)
                # 只有当没有显著垂直关系时，才考虑水平关系
                if rel is None:
                    if dist_3d < NEAR_THRESH and abs(dy) < VERTICAL_THRESH:
                        rel = "next_to"
                        # 距离越近权重越高
                        weight = max(0.5, 1.0 - (dist_3d / NEAR_THRESH))

                if rel:
                    relationships.append([u, v, rel, float(f"{weight:.2f}")])

        return relationships

    def generate_spatial_cluster_summary(self, object_list: List[Dict], region_image_path: str = None) -> Dict:
        client, model_name, temp = self._get_client_and_model('cluster_summary')
        objs_text = "\n".join([f"- {item['tag']}: {item.get('desc', '')}" for item in object_list])
        prompt_text = f"任务：推断区域功能。\n包含物体：\n{objs_text}\n要求：\n1. Area Tag (如 reading_corner, dining_area).(不是判断房间，而是功能区，是房间节点的子节点)\n2. Summary (详细描述布局和功能, 必须使用英文).\n输出 JSON: {{\"area_tag\": \"...\", \"summary\": \"...\"}}"
        messages = []
        if region_image_path and os.path.exists(region_image_path):
            base64_img = self._encode_image(region_image_path)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}]}]
        else:
            messages = [{"role": "user", "content": prompt_text}]
        try:
            response = client.chat.completions.create(model=model_name, messages=messages, max_tokens=500, temperature=temp)
            self._update_token_stats(response)  # <--- [新增] 插入在这里
            content = response.choices[0].message.content.strip()
            if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: content = content.split("```", 1)[1].split("```", 1)[0].strip()
            return json.loads(content)
        except Exception: return {"area_tag": "unknown", "summary": "Failed."}











    # 在 LLMHandler 类中添加以下方法
    # [添加到 LLMHandler 类中]
    def generate_multimodal_room_summary(self, image_path: str, area_summaries: List[Dict]) -> Dict:
        """
        结合房间的最佳视角图片和包含的区域描述，生成房间的语义标签和整体描述。
        """
        # [修正点]：这里必须使用一个定义在 vision_tasks 列表中的任务名（如 'cluster_summary'），
        # 否则 _get_client_and_model 会返回不支持图片的 text_model，导致 API 报错。
        client, model_name, temp = self._get_client_and_model('scene_summary')

        # 构造区域描述文本
        areas_text = "\n".join([f"- Area {item.get('tag', 'unknown')}: {item.get('summary', '')}" for item in area_summaries])
        if not areas_text:
            areas_text = "No specific functional areas detected yet."

        prompt_text = f"""
Task: Analyze the provided image of a room and the list of functional areas within it to generate a final room definition.

Input Data:
1. **Image**: A representative view of the room.
2. **Contained Areas**:
{areas_text}

Requirements:
1. **Room Label**: Provide a short, specific semantic category for this room (e.g., "Bedroom", "Office", "Kitchen", "Living Room", "Corridor", "Bathroom").
2. **Description**: Write a concise summary (under 100 words) of the room's layout, function, and atmosphere based on both the visual visual style and the contained areas.

Output JSON only:
{{
    "room_label": "...",
    "description": "..."
}}
"""
        messages = []
        # 编码图片并构建消息
        if image_path and os.path.exists(image_path):
            base64_img = self._encode_image(image_path)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
            }]
        else:
            # 如果没有图片（极端情况），仅使用文本
            messages = [{"role": "user", "content": prompt_text}]

        last_error = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=600,
                    temperature=temp,
                    response_format={"type": "json_object"},
                    _router_cache_salt=f"room_summary_json_v1_retry_{attempt}",
                )
                self._update_token_stats(response)
                content = response.choices[0].message.content.strip()
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```", 1)[1].split("```", 1)[0].strip()
                result = json.loads(content)
                if not isinstance(result, dict) or not result.get("room_label") or not result.get("description"):
                    raise ValueError("Room summary JSON is missing required fields")
                return result
            except Exception as e:
                last_error = e
                print(f"[LLM Retry {attempt + 1}/3] Multimodal Room Summary: {e}")
        print(f"[LLM Error] Multimodal Room Summary failed after retries: {last_error}")
        return {"room_label": "Unknown Room", "description": "Failed to generate description due to LLM error."}

    # [新增] VLM 物体扩充方法
    def detect_room_objects_vlm(self, image_path: str, existing_labels: List[str]) -> List[str]:
        """
        让 VLM 观察房间图片，补充 SAM 可能漏掉的物体标签。
        """
        client, model_name, temp = self._get_client_and_model('object_expansion')

        # 编码图片
        if not image_path or not os.path.exists(image_path):
            return []
        base64_img = self._encode_image(image_path)

        # 构造 Prompt
        # 为了防止 Prompt 过长，只取前 50 个已知标签
        known_str = ", ".join(existing_labels[:50])

        prompt_text = f"""
Task: Analyze the image of the room and list visible objects that are NOT in the "Known Objects" list.

Known Objects: [{known_str}]

Requirements:
1. **Identify New Objects**: List distinct, physical objects visible in the image that are missing from the known list.
2. **Strict Exclusions**:
   - DO NOT list structural elements: wall, floor, ceiling, window, door, door_frame, window_pane.
   - DO NOT list generic terms: item, object, thing, stuff, entity, unknown.
   - DO NOT list lighting unless it's a specific lamp (e.g., exclude "light", "lighting").
3. **Format**: Output ONLY one JSON object with a `new_objects` list
   (e.g., {{"new_objects": ["magazine", "remote control", "plant"]}}).
4. **Specificity**: Be specific (e.g., use "cushion" instead of "cloth").

Output JSON:
"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]
        }]
        last_error = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=400,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    _router_cache_salt=f"object_expansion_json_v1_retry_{attempt}",
                )
                self._update_token_stats(response)
                content = response.choices[0].message.content.strip()
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```", 1)[1].split("```", 1)[0].strip()
                payload = json.loads(content)
                new_objects = payload.get("new_objects") if isinstance(payload, dict) else None
                if not isinstance(new_objects, list):
                    raise ValueError("Object expansion JSON is missing new_objects list")
                return [str(o).lower().replace(" ", "_") for o in new_objects]
            except Exception as e:
                last_error = e
                print(f"[LLM Retry {attempt + 1}/3] Object expansion: {e}")
        print(f"[LLM Error] Object expansion failed after retries: {last_error}")
        return []
