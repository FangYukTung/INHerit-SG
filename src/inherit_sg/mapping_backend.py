import os
import sys
import datetime
import json
import yaml
import csv
import time
import pickle
import shutil
from typing import List, Dict, Any, Tuple
import cv2
import gc
import copy
import argparse
import networkx as nx
import threading
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from sentence_transformers import util
from skimage.draw import line as sk_line
from scipy.ndimage import binary_dilation
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.patches import Wedge
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sentence_transformers import SentenceTransformer
try:
    from ram.models import ram_plus
    from ram import inference_ram as inference
    from ram import get_transform
except ImportError:
    ram_plus = None
    inference = None
    get_transform = None
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ================== 1. 强制修复路径 ==================
# 指向你提供的 sam3 源码根目录
SAM3_SOURCE_ROOT = os.environ.get(
    "INHERIT_SG_SAM3_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sam3")),
)
DIST_THRESH = 1.2 ## 1.2 米距离阈值，用于对象聚类
if SAM3_SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SAM3_SOURCE_ROOT)

# ================== 2. 导入 SAM3 底层模块 ==================
try:
    from sam3 import build_sam3_image_model
    # 导入底层数据结构
    from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
    # 导入变换和批处理工具
    from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
    from sam3.train.data.collator import collate_fn_api as collate
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.eval.postprocessors import PostProcessImage
except ImportError as e:
    print(f"\n[CRITICAL ERROR] 无法导入 SAM3 模块。请确认路径 {SAM3_SOURCE_ROOT} 正确。")
    print(f"详细错误: {e}\n")
    raise e

# ================== 3. 定义必要的辅助函数 (移植自官方 Notebook) ==================
# 这些函数在 pip 包里没有，必须在这里定义

_GLOBAL_COUNTER = 1

def create_empty_datapoint():
    """ 创建一个空的数据点 """
    return Datapoint(find_queries=[], images=[])

def set_image(datapoint, pil_image):
    """ 设置图片 """
    w, h = pil_image.size
    # 注意：这里直接使用 SAMImage 类
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]

def add_text_prompt(datapoint, text_query):
    """ 添加文本提示 """
    global _GLOBAL_COUNTER

    if len(datapoint.images) != 1:
        raise ValueError("Please set the image first using set_image()")

    w, h = datapoint.images[0].size

    # 构造查询对象
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=_GLOBAL_COUNTER,
                original_image_id=_GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            )
        )
    )
    _GLOBAL_COUNTER += 1
    return _GLOBAL_COUNTER - 1

# --- 引入 Common 模块 ---
from .reid import choose_visual_match
from .common import (
    Logger,
    LLMHandler,
    read_pose_matrix,
    pixel_to_world,
    process_nodes_file,
    prune_relations_file,
    analyze_area_quality,
)

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any
from transformers import AutoImageProcessor, AutoModel

def load_feature_extractor(config: Dict[str, Any], device: str) -> Any:
    cfg = config.get('feature_extractor', {})
    model_path = cfg.get('weights_path')
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError('feature_extractor.weights_path must name a trusted local model directory')
    model_dir = Path(model_path).expanduser().resolve()
    if not model_dir.is_dir() or not (model_dir / 'config.json').is_file():
        raise ValueError(f'Feature extractor must be a trusted local model directory with config.json: {model_dir}')

    print(f"\n--- Loading Transformer Model from {model_dir} ---")

    # 1. 加载处理器 (替代原本的 T.Compose)
    processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)

    # 2. 加载模型
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,  # Only execute custom code from the user-supplied local model.
        local_files_only=True,
        #device_map=device       # 自动放置到指定设备，或者手动 .to(device)
    )

    # 手动确保模型在正确的模式和设备上（如果是 device_map="auto" 可以省略 .to(device)）

    model.to(device)
    model.eval()

    # Hack: 将 processor 绑定到 model 对象上
    # 这样就不需要修改 extract_feature 的函数签名（即不需要传入 processor 参数）
    model.hf_processor = processor

    return model


def extract_feature(image_pil: Image, model: Any, device: str) -> np.ndarray:
    if not hasattr(model, 'hf_processor'):
        raise AttributeError("Model object missing 'hf_processor'.")

    processor = model.hf_processor

    # [修改核心] 在这里传入参数覆盖 preprocessor_config.json 的默认值
    # 目的：模拟旧代码 T.Compose([T.Resize(518), T.CenterCrop(518), ...])
    inputs = processor(
        images=image_pil,
        return_tensors="pt",

        # 1. 强制开启 Resize 并指定为 "shortest_edge" 模式
        #    这对应 T.Resize(518)，即短边缩放到 518，长边按比例缩放
        do_resize=True,
        size={"shortest_edge": 518},

        # 2. 强制开启 CenterCrop 并指定尺寸
        #    这对应 T.CenterCrop(518)
        do_center_crop=True,
        crop_size={"height": 518, "width": 518},

        # 3. 归一化 (preprocessor_config 中已经是 True，这里为了保险可以显式写上，也可省略)
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406], # 确保与 ImageNet 统计数据一致
        image_std=[0.229, 0.224, 0.225]
    )

    # 确保输入数据移动到模型所在的设备
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    # DINOv3 处理逻辑：优先取 pooler_output，否则取 CLS token
    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
        features = outputs.pooler_output
    else:
        features = outputs.last_hidden_state[:, 0, :]

    return features.cpu().numpy().flatten()


class MapManager:
    def __init__(self, config: Dict, intrinsics: Dict, run_dir: str):
        self.config = config
        self.intrinsics = intrinsics
        self.resolution = config.get('map_resolution', 0.05)
        self.run_dir = run_dir

        self.map_w = config.get('map_w', 2000)
        self.map_h = config.get('map_h', 2000)
        self.min_x = config.get('min_x', -50.0)
        self.min_z = config.get('min_z', -50.0)

        # ===== [NEW] log-odds occupancy mapping =====
        self.logodds = np.zeros((self.map_h, self.map_w), dtype=np.float32)
        self.L_OCC = config.get('L_OCC', 0.85)     # occupied update
        self.L_FREE = config.get('L_FREE', -0.25)  # free update
        self.L_MIN = config.get('L_MIN', -4.0)
        self.L_MAX = config.get('L_MAX', 4.0)

        # door as a separate semantic layer (binary)
        self.door_map = np.zeros((self.map_h, self.map_w), dtype=np.uint8)

        # cached binary maps (updated each keyframe for viz/seg)
        self.occ_bin = np.zeros((self.map_h, self.map_w), dtype=np.uint8)
        self.free_bin = np.zeros((self.map_h, self.map_w), dtype=np.uint8)

        # room tracking (pure geometry)
        self.room_id_map = -np.ones((self.map_h, self.map_w), dtype=np.int32)  # optional global label
        self.current_room_id = 0
        self.room_registry = {}  # room_id -> {"centroid": (x,z), "area": float}

        # [KEEP] optional room geometry overlay if you still want
        self.room_geometry = {}

        # mapping params
        self.max_range = config.get('max_range_m', 8.0)
        self.min_range = config.get('min_range_m', 0.3)
        self.raycast_stride = config.get('raycast_stride', 4)  # pixel subsample in depth image
        self.hit_stride = config.get('hit_stride', 2)          # subsample for hit points

        # occ inflation for planning/segmentation (robot radius)
        self.robot_radius_m = config.get('robot_radius_m', 0.30)
        self.inflation_m = config.get('inflation_m', 0.10)
        # [ADD in MapManager.__init__]
        self.raycast_update_occupied = bool(config.get("raycast_update_occupied", False))

        # Door evidence accumulator to avoid speckles
        self.door_hits = np.zeros((self.map_h, self.map_w), dtype=np.uint16)
        self.door_hit_inc = int(config.get("door_hit_inc", 1))
        self.door_hit_thresh = int(config.get("door_hit_thresh", 6))
        # [新增] 多楼层存储
        self.floor_maps = {} # floor_id -> dict of map arrays
        self.current_floor_idx = 0

    # ---------- [NEW] coordinate transforms ----------
    def world_to_grid(self, x: float, z: float) -> Tuple[int, int]:
        u = int((x - self.min_x) / self.resolution)
        v = int((z - self.min_z) / self.resolution)
        return u, v

    def grid_to_world(self, u: int, v: int) -> Tuple[float, float]:
        x = u * self.resolution + self.min_x
        z = v * self.resolution + self.min_z
        return x, z

    def _valid_uv(self, u: int, v: int) -> bool:
        return (0 <= u < self.map_w) and (0 <= v < self.map_h)

    # ---------- [NEW] log-odds update helpers ----------
    def _update_logodds(self, u: np.ndarray, v: np.ndarray, delta: float):
        valid = (u >= 0) & (u < self.map_w) & (v >= 0) & (v < self.map_h)
        u, v = u[valid], v[valid]
        self.logodds[v, u] = np.clip(self.logodds[v, u] + delta, self.L_MIN, self.L_MAX)

    def refresh_binary_maps(self):
        # P(occ)=sigmoid(L); threshold at 0.65-ish
        occ = self.logodds > 0.85
        free = self.logodds < -0.40
        self.occ_bin = occ.astype(np.uint8)
        self.free_bin = free.astype(np.uint8)

    # ---------- [NEW] depth raycasting mapping ----------
    def integrate_depth_raycast(self, depth_img: np.ndarray, pose: np.ndarray):
        """
        depth_img: uint16/float depth in raw units, convert using intrinsics['depth_scale']
        pose: 4x4 camera-to-world (Habitat world) like your pixel_to_world convention
        """
        if depth_img is None or pose is None:
            return

        fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
        cx, cy = self.intrinsics['cx'], self.intrinsics['cy']
        ds = self.intrinsics.get('depth_scale', 1000.0)

        h, w = depth_img.shape[:2]

        # camera origin in world
        cam_world = pose[:3, 3]
        u0, v0 = self.world_to_grid(cam_world[0], cam_world[2])

        # T_fix: OpenCV -> Habitat (same as your pixel_to_world)
        # p_hab = [x, -y, -z]
        # Here we do vectorized unprojection in OpenCV then apply sign flips.
        step = int(self.raycast_stride)
        us = np.arange(0, w, step)
        vs = np.arange(0, h, step)
        uu, vv = np.meshgrid(us, vs)
        uu = uu.reshape(-1)
        vv = vv.reshape(-1)

        d_raw = depth_img[vv, uu].astype(np.float32)
        z = d_raw / ds
        valid = (z > self.min_range) & (z < self.max_range)
        if np.count_nonzero(valid) == 0:
            return

        uu, vv, z = uu[valid], vv[valid], z[valid]

        x_cam = (uu - cx) * z / fx
        y_cam = (vv - cy) * z / fy
        # OpenCV -> Habitat
        p_hab = np.stack([x_cam, -y_cam, -z, np.ones_like(z)], axis=1)  # (N,4)
        p_world = (pose @ p_hab.T).T[:, :3]  # (N,3)

        # Raycast: mark free along ray (origin -> hit), mark occupied at hit
        # subsample hits to reduce work
        hit_stride = max(1, int(self.hit_stride))
        p_world = p_world[::hit_stride]

        for pw in p_world:
            uh, vh = self.world_to_grid(pw[0], pw[2])
            if not (self._valid_uv(u0, v0) and self._valid_uv(uh, vh)):
                continue
            rr, cc = sk_line(v0, u0, vh, uh)  # note: (row,col) = (v,u)

            if len(rr) < 2:
                continue

            # free cells: all but last
            rr_free, cc_free = rr[:-1], cc[:-1]
            self._update_logodds(cc_free, rr_free, self.L_FREE)

            # [REPLACE] occupied update is optional
            if self.raycast_update_occupied:
                self._update_logodds(np.array([uh]), np.array([vh]), self.L_OCC)

    # ---------- [NEW] integrate door / wall masks (optional but recommended) ----------
    def integrate_structure_points(self, points_3d: np.ndarray, label_type: str):
        """
        points_3d: (N,3) world coords
        label_type: 'wall' or 'door'
        """
        if points_3d is None or len(points_3d) == 0:
            return

        u = ((points_3d[:, 0] - self.min_x) / self.resolution).astype(int)
        v = ((points_3d[:, 2] - self.min_z) / self.resolution).astype(int)

        valid = (u >= 0) & (u < self.map_w) & (v >= 0) & (v < self.map_h)
        u, v = u[valid], v[valid]

        lt = (label_type or "").lower()

        # ---- DOOR: evidence with decay + saturation + dynamic thresholding (B方案) ----
        if lt in ["door", "complete door frame", "doorway"]:
            # Ensure float hits (decay needs float). If your door_hits is uint16 from old code,
            # convert once here (cheap safeguard).
            if self.door_hits.dtype != np.float32 and self.door_hits.dtype != np.float64:
                self.door_hits = self.door_hits.astype(np.float32)

            # (1) decay old evidence each integration call
            decay = float(getattr(self, "door_hits_decay", 0.98))
            self.door_hits *= decay

            # (2) add new evidence
            inc = float(getattr(self, "door_hit_inc", 1.0))
            self.door_hits[v, u] += inc

            # (3) saturate / cap
            cap = float(getattr(self, "door_hits_cap", 20.0))
            np.clip(self.door_hits, 0.0, cap, out=self.door_hits)

            # (4) compute door_map dynamically from hits (NOT permanent)
            thr = float(getattr(self, "door_hit_thresh", 6.0))
            door_map = (self.door_hits >= thr)

            # (5) optional: dilation ONLY for visualization/cut convenience;
            # do NOT feed back into door_hits, and it will naturally shrink when hits decay.
            dil_iter = int(getattr(self, "door_map_dilate_iter", 1))
            if dil_iter > 0:
                door_map = binary_dilation(door_map, iterations=dil_iter)

            self.door_map = door_map.astype(np.uint8)
            return

        # ---- WALL / OTHER: treat as occupancy evidence ----
        if lt in ["wall"]:
            self._update_logodds(u, v, self.L_OCC)
            return

        # default: also occupancy
        self._update_logodds(u, v, self.L_OCC)

    # ---------- [NEW] inflated obstacle mask for segmentation ----------
    def get_inflated_occ(self) -> np.ndarray:
        r = int((self.robot_radius_m + self.inflation_m) / self.resolution)
        if r <= 0:
            return self.occ_bin.astype(bool)
        return binary_dilation(self.occ_bin.astype(bool), iterations=r)

    # ---------- [NEW] visualization per keyframe ----------
    def render_keyframe_viz(self, trajectory: List[np.ndarray], current_pose: np.ndarray,
                            local_window_m: float = 20.0,
                            cut_mask_local: np.ndarray = None,
                            room_mask_local: np.ndarray = None) -> np.ndarray:
        """
        Returns a BGR image. Shows occ/free/unknown + doors + cut + current room region + trajectory + arrow.
        Crops to a local window around robot for clarity.
        """
        self.refresh_binary_maps()

        vis = np.zeros((self.map_h, self.map_w, 3), dtype=np.uint8)

        # unknown = dark
        # free = light
        # occ = gray
        vis[self.free_bin == 1] = (240, 240, 240)
        vis[self.occ_bin == 1] = (120, 120, 120)

        # doors in cyan
        vis[self.door_map == 1] = (255, 255, 0)

        # overlay cut/room if provided in local coords
        if current_pose is not None and (cut_mask_local is not None or room_mask_local is not None):
            cx, cz = current_pose[0, 3], current_pose[2, 3]
            cu, cv = self.world_to_grid(cx, cz)
            half = int((local_window_m / self.resolution) / 2)
            x1, x2 = max(0, cu - half), min(self.map_w, cu + half)
            y1, y2 = max(0, cv - half), min(self.map_h, cv + half)

            overlay = vis.copy()
            if cut_mask_local is not None:
                # cut in magenta
                cm = cut_mask_local.astype(bool)
                overlay[y1:y2, x1:x2][cm] = (255, 0, 255)
            if room_mask_local is not None:
                # current room in green (semi)
                rm = room_mask_local.astype(bool)
                overlay[y1:y2, x1:x2][rm] = (0, 255, 0)
            cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

        # trajectory
        pts = []
        for p in trajectory:
            tx = p[0]
            tz = p[2] if len(p) > 2 else p[1]
            u, v = self.world_to_grid(tx, tz)
            if 0 <= u < self.map_w and 0 <= v < self.map_h:
                pts.append([u, v])
        if len(pts) > 1:
            cv2.polylines(vis, [np.array(pts, np.int32)], False, (0, 0, 255), 2)
        for u, v in pts:
            cv2.circle(vis, (u, v), 2, (0, 0, 255), -1)

        # robot arrow
        if current_pose is not None:
            cx, cz = current_pose[0, 3], current_pose[2, 3]
            cu, cv = self.world_to_grid(cx, cz)
            vec_fwd = current_pose[:3, 2] * (-1)
            ex = cx + vec_fwd[0] * 1.5
            ez = cz + vec_fwd[2] * 1.5
            eu, ev = self.world_to_grid(ex, ez)
            cv2.arrowedLine(vis, (cu, cv), (eu, ev), (0, 200, 0), 3, tipLength=0.3)

        # crop local
        if current_pose is not None:
            cx, cz = current_pose[0, 3], current_pose[2, 3]
            cu, cv = self.world_to_grid(cx, cz)
            half = int((local_window_m / self.resolution) / 2)
            x1, x2 = max(0, cu - half), min(self.map_w, cu + half)
            y1, y2 = max(0, cv - half), min(self.map_h, cv + half)
            vis = vis[y1:y2, x1:x2].copy()

        return vis
        # [新增] 检查 3D 点是否在指定房间 ID 的区域内
    def is_point_in_room(self, point_3d: list | np.ndarray, room_id: int | str) -> bool:
        """
        point_3d: [x, y, z] 世界坐标
        room_id: 目标房间 ID
        """
        if point_3d is None:
            return False

        # 提取 X, Z (假设 Y 是高度，不影响二维栅格判断)
        x = point_3d[0]
        z = point_3d[2]

        u, v = self.world_to_grid(x, z)

        # 1. 检查坐标是否在地图范围内
        if not self._valid_uv(u, v):
            return False

        # 2. 获取该位置的房间 ID
        # 注意：room_id_map 存储的是 int32
        current_val = self.room_id_map[v, u]

        # 3. 比较
        try:
            # 兼容字符串或数字类型的比较
            return int(current_val) == int(room_id)
        except (ValueError, TypeError):
            return False
        # [新增] 切换楼层方法
    # [添加到 MapManager 类中]
    def calculate_visible_room_area(self, pose: np.ndarray, room_id: int, fov_deg: float = 90.0, max_range_m: float = 5.0) -> int:
        """
        计算在给定 Pose 下，视野扇形与目标房间的重叠面积（考虑墙壁遮挡）。
        逻辑：Raycast from robot -> Stop at Wall -> Count overlapping pixels.
        """
        if self.room_id_map is None: return 0
        if pose is None: return 0

        # 1. 获取机器人网格坐标
        rx, rz = pose[0, 3], pose[2, 3]
        u0, v0 = self.world_to_grid(rx, rz)

        # 边界检查
        if not self._valid_uv(u0, v0): return 0

        # 2. 计算朝向 (Heading)
        # Habitat/OpenGL: Camera -Z is forward
        # World coords: X is Right, Z is Back.
        # Grid coords: u aligns with X, v aligns with Z.
        vec_fwd = -pose[:3, 2]
        # atan2(y, x) -> atan2(delta_z, delta_x)
        heading = np.arctan2(vec_fwd[2], vec_fwd[0])

        # 3. 射线参数
        num_rays = 45 # 射线数量，越多越精确，45条意味着每2度一条，足够了
        max_dist_px = int(max_range_m / self.resolution)

        # 生成扇形角度
        angles = np.linspace(heading - np.radians(fov_deg/2), heading + np.radians(fov_deg/2), num_rays)

        # 准备地图引用
        occ_map = self.occ_bin
        room_map = self.room_id_map
        target_rid = int(room_id)

        # 使用 Set 统计独立的可见像素点 (去重，即计算面积)
        visible_room_pixels = set()

        h, w = self.map_h, self.map_w

        # 4. 执行 Raycast
        for ang in angles:
            # 计算终点
            ue = int(u0 + max_dist_px * np.cos(ang))
            ve = int(v0 + max_dist_px * np.sin(ang))

            # 获取射线路径 (Bresenham)
            rr, cc = sk_line(v0, u0, ve, ue) # sk_line returns (row, col) -> (v, u)

            # 遍历射线上的点
            for r, c in zip(rr, cc):
                # 越界检查
                if r < 0 or r >= h or c < 0 or c >= w:
                    break

                # 障碍物检查 (碰到墙就停止)
                # 容忍度：允许穿过 1 个像素的噪点，超过1个算墙
                if occ_map[r, c] > 0:
                    # 简单的穿透检查，这里简化为碰到即停
                    break

                # 检查是否属于目标房间
                if room_map[r, c] == target_rid:
                    visible_room_pixels.add((r, c))

        score = len(visible_room_pixels)

        # 5. "身在其中" 奖励
        # 如果机器人当前脚下的点就在目标房间内，给予 3.0 倍加权
        # 这能确保一旦进入房间，立刻优于门外的视角
        current_pos_rid = room_map[v0, u0]
        if int(current_pos_rid) == target_rid:
            score *= 3.0

        return int(score)
    # [新增] 规则化单个房间掩码 (使其变成直角、方正的形状)
    # [修改] 规则化单个房间掩码 (直角化、预测补全)
    def regularize_mask_rectilinear(self, mask: np.ndarray) -> np.ndarray:
        if np.sum(mask) == 0: return mask

        # 1. 闭运算填补内部空洞
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 2. 轮廓近似 (Axis-Aligned Bounding Box 的变体)
        # 这里用一种简单粗暴的方法：在低分辨率下处理，然后再放大
        # 这天然会产生锯齿状（直角）边缘

        h, w = mask.shape
        scale_down = 0.1 # 缩小10倍

        small = cv2.resize(mask, (int(w * scale_down), int(h * scale_down)), interpolation=cv2.INTER_NEAREST)

        # 在小图上做开闭运算，去除噪点
        small_kernel = np.ones((3,3), np.uint8)
        small = cv2.morphologyEx(small, cv2.MORPH_OPEN, small_kernel)
        small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, small_kernel)

        # 放大回去，使用最近邻插值保持锯齿（直角）
        final_mask = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        return final_mask
    # [新增] 全局重组房间 (解决重叠，分配空地)
    def resolve_global_room_overlaps(self, room_masks: Dict[int, np.ndarray]) -> np.ndarray:
        """
        输入: {room_id: binary_mask}
        输出: 全局唯一的 room_id_map (int32)，无重叠，紧邻。
        原理: 距离场竞争算法 (Distance Field Competition)
        """
        if not room_masks:
            return np.zeros((self.map_h, self.map_w), dtype=np.int32)

        # 1. 构建距离场堆栈
        # shape: (N, H, W), N 是房间数
        valid_ids = list(room_masks.keys())
        h, w = self.map_h, self.map_w

        # 初始化距离图：背景为 0
        distance_stack = np.zeros((len(valid_ids), h, w), dtype=np.float32)

        for idx, rid in enumerate(valid_ids):
            mask = room_masks[rid]
            if np.sum(mask) == 0: continue

            # 计算欧氏距离变换：像素点离背景（0）越远，值越大
            # 这代表了“房间核心”的强度
            dist = distance_transform_edt(mask)
            distance_stack[idx] = dist

        # 2. 竞争归属
        # 找出每个像素在哪个房间的距离值最大
        max_dist_vals = np.max(distance_stack, axis=0)
        max_indices = np.argmax(distance_stack, axis=0)

        # 3. 生成结果图
        new_map = np.zeros((h, w), dtype=np.int32)

        # 只有当最大距离 > 0 时才分配 ID (过滤掉背景)
        mask_valid = max_dist_vals > 0

        # 将 index 映射回 room_id
        id_array = np.array(valid_ids, dtype=np.int32)
        new_map[mask_valid] = id_array[max_indices[mask_valid]]

        # 4. (可选) 二次膨胀填补缝隙
        # 因为距离变换可能会让边界处产生细微缝隙，用最近邻插值填补
        # 这里简单略过，通常 distance transform 竞争产生的边界是完美的 Voronoi 边界

        self.room_id_map = new_map
        return new_map

    def switch_floor(self, new_floor_idx: int):
        if self.current_floor_idx == new_floor_idx:
            return

        print(f"[MapManager] Switching Context: Floor {self.current_floor_idx} -> {new_floor_idx}")

        # 1. 保存当前楼层状态
        self.floor_maps[self.current_floor_idx] = {
            "logodds": self.logodds.copy(),
            "door_map": self.door_map.copy(),
            "occ_bin": self.occ_bin.copy(),
            "free_bin": self.free_bin.copy(),
            "room_id_map": self.room_id_map.copy(),
            "door_hits": self.door_hits.copy()
        }

        # 2. 加载或初始化新楼层
        if new_floor_idx in self.floor_maps:
            print(f"  -> Restoring existing map for Floor {new_floor_idx}")
            data = self.floor_maps[new_floor_idx]
            self.logodds = data["logodds"]
            self.door_map = data["door_map"]
            self.occ_bin = data["occ_bin"]
            self.free_bin = data["free_bin"]
            self.room_id_map = data["room_id_map"]
            self.door_hits = data["door_hits"]
        else:
            print(f"  -> Initializing NEW map for Floor {new_floor_idx}")
            self.logodds = np.zeros((self.map_h, self.map_w), dtype=np.float32)
            self.door_map = np.zeros((self.map_h, self.map_w), dtype=np.uint8)
            self.occ_bin = np.zeros((self.map_h, self.map_w), dtype=np.uint8)
            self.free_bin = np.zeros((self.map_h, self.map_w), dtype=np.uint8)
            self.room_id_map = -np.ones((self.map_h, self.map_w), dtype=np.int32)
            self.door_hits = np.zeros((self.map_h, self.map_w), dtype=np.uint16) # or float32 if updated

        self.current_floor_idx = new_floor_idx

class GeometryRoomTracker:
    def __init__(self, map_manager, config: dict):
        self.mm = map_manager
        self.config = config

        # [修改] 增大最小房间半径，强制种子点分开 (1.5m)
        # 假设分辨率 0.05，则 1.5m = 30px
        self.min_room_radius_px = int(1.5 / self.mm.resolution)

        # [修改] 腐蚀核适中，太小会导致噪点，太大会腐蚀掉门
        self.erosion_kernel = 3  # 3x3 kernel (半径1)

        self.current_room_id = 0
        self.next_global_room_id = 1

        # Global Maps
        self.global_id_grid = np.zeros((self.mm.map_h, self.mm.map_w), dtype=np.int32)
        self.global_conf_grid = np.zeros((self.mm.map_h, self.mm.map_w), dtype=np.uint8)

        # 置信度参数
        self.CONF_INC = 10
        self.CONF_DEC = 2   # 减分慢一点，防止闪烁
        self.CONF_MAX = 100
        self.CONF_THRESH = 1 # 调试模式：只要有就显示

        # 颜色盘
        self.colors = np.random.randint(50, 255, (100, 3), dtype=np.uint8)
        self.colors[0] = [0, 0, 0]

        self.floor_states = {}
        self.current_floor_idx = 0

    # ... (switch_floor 保持不变) ...
    def switch_floor(self, new_floor_idx: int):
        self.floor_states[self.current_floor_idx] = {
            "id_grid": self.global_id_grid.copy(),
            "conf_grid": self.global_conf_grid.copy(),
            "next_id": self.next_global_room_id
        }
        if new_floor_idx in self.floor_states:
            st = self.floor_states[new_floor_idx]
            self.global_id_grid = st["id_grid"]
            self.global_conf_grid = st["conf_grid"]
            self.next_global_room_id = st["next_id"]
        else:
            self.global_id_grid = np.zeros((self.mm.map_h, self.mm.map_w), dtype=np.int32)
            self.global_conf_grid = np.zeros((self.mm.map_h, self.mm.map_w), dtype=np.uint8)
        self.current_floor_idx = new_floor_idx
        self.current_room_id = 0

    def step(self, pose: np.ndarray, floor_trajectory: List[np.ndarray] = None) -> dict:
        if pose is None: return {"room_id": 0, "viz": None}

        # 1. 准备 Free Space Mask
        door_obstacle = (self.mm.door_map > 0).astype(np.uint8)
        grid_free = self.mm.free_bin.copy()

        if np.sum(grid_free) == 0:
            return {"room_id": 0, "viz": self._render_empty_viz(pose)}

        # 减去门，物理阻断
        effective_free = cv2.bitwise_and(grid_free, grid_free, mask=cv2.bitwise_not(door_obstacle))

        # 腐蚀
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded_free = cv2.erode(effective_free, kernel, iterations=1)

        # 2. 生成局部建议
        local_proposal = np.zeros_like(grid_free, dtype=np.int32)

        if np.sum(eroded_free) > 0:
            # 计算距离变换
            dt = distance_transform_edt(eroded_free)

            # [核心修改] 对 DT 进行高斯模糊，平滑掉局部小峰值，防止切太碎
            # sigma 设为 2.0 左右，能平滑掉 10-20cm 的噪点
            dt = cv2.GaussianBlur(dt, (7, 7), 2.0)

            # 寻找峰值
            local_maxi = peak_local_max(dt, min_distance=self.min_room_radius_px, labels=eroded_free, exclude_border=False)

            if len(local_maxi) > 0:
                markers = np.zeros_like(grid_free, dtype=np.int32)
                for idx, (r, c) in enumerate(local_maxi):
                    markers[r, c] = idx + 1

                # 分水岭
                ws_labels = watershed(-dt, markers, mask=effective_free)
                local_proposal = ws_labels.astype(np.int32)

        # 3. 更新全局
        self._update_global_map(local_proposal)

        # 4. 输出
        current_global_mask = np.zeros_like(self.global_id_grid)
        valid_mask = self.global_conf_grid >= self.CONF_THRESH
        current_global_mask[valid_mask] = self.global_id_grid[valid_mask]

        # 5. 确定 ID
        rx, rz = pose[0, 3], pose[2, 3]
        ru, rv = self.mm.world_to_grid(rx, rz)
        if 0 <= ru < self.mm.map_w and 0 <= rv < self.mm.map_h:
            rid = current_global_mask[rv, ru]
            if rid > 0: self.current_room_id = int(rid)

        # 6. 可视化
        viz_img = self._render_viz(current_global_mask, pose, trajectory=floor_trajectory)

        return {
            "room_id": self.current_room_id,
            "room_mask_local": current_global_mask,
            "viz": viz_img,
            "changed": False
        }

    def _update_global_map(self, local_proposal):
        unique_local = np.unique(local_proposal)
        unique_local = unique_local[unique_local > 0]

        local_to_global = {}
        used_global_ids = set()

        # A. 匹配
        for lid in unique_local:
            local_mask = (local_proposal == lid)
            global_region = self.global_id_grid[local_mask]
            conf_region = self.global_conf_grid[local_mask]

            # 只要置信度 > 0 就参与投票 (放宽条件)
            valid_globals = global_region[conf_region > 0]

            target_gid = -1
            if len(valid_globals) > 0:
                ids, counts = np.unique(valid_globals, return_counts=True)
                # 排除 0
                if ids[0] == 0 and len(ids) > 1:
                    ids = ids[1:]
                    counts = counts[1:]
                elif ids[0] == 0:
                    pass

                if len(ids) > 0 and ids[0] != 0:
                    best_idx = np.argmax(counts)
                    target_gid = ids[best_idx]

            if target_gid == -1 or target_gid in used_global_ids:
                target_gid = self.next_global_room_id
                self.next_global_room_id += 1

            local_to_global[lid] = target_gid
            used_global_ids.add(target_gid)

        # B. 更新
        for lid, gid in local_to_global.items():
            local_mask = (local_proposal == lid)

            # 1. 之前是空的或者ID相同 -> 加分
            # 注意：如果是新区域(0)，直接初始化为 CONF_THRESH + INC，确保立刻显示
            new_area_mask = local_mask & (self.global_id_grid == 0)
            self.global_id_grid[new_area_mask] = gid
            self.global_conf_grid[new_area_mask] = self.CONF_THRESH + self.CONF_INC # 立即显示

            match_mask = local_mask & (self.global_id_grid == gid)
            self.global_conf_grid[match_mask] = np.clip(self.global_conf_grid[match_mask] + self.CONF_INC, 0, self.CONF_MAX)

            # 2. 冲突 -> 减分/翻转
            mismatch_mask = local_mask & (self.global_id_grid != gid) & (self.global_id_grid != 0)
            self.global_conf_grid[mismatch_mask] = np.clip(self.global_conf_grid[mismatch_mask] - self.CONF_DEC, 0, self.CONF_MAX)

            flip_mask = mismatch_mask & (self.global_conf_grid < self.CONF_INC) # 阈值放宽
            self.global_id_grid[flip_mask] = gid
            self.global_conf_grid[flip_mask] = self.CONF_THRESH + self.CONF_INC

    def _render_viz(self, room_mask, pose, trajectory: List[np.ndarray] = None):
        h, w = room_mask.shape
        viz = np.ones((h, w, 3), dtype=np.uint8) * 240 # Light Gray background

        # 1. 绘制房间
        valid_mask = room_mask > 0
        if np.sum(valid_mask) > 0:
            max_id = np.max(room_mask)
            if max_id >= len(self.colors):
                more_colors = np.random.randint(50, 255, (max_id - len(self.colors) + 50, 3), dtype=np.uint8)
                self.colors = np.vstack((self.colors, more_colors))
            viz[valid_mask] = self.colors[room_mask[valid_mask]]

        # 2. 绘制障碍物 (黑色)
        occ_mask = self.mm.occ_bin > 0
        viz[occ_mask] = [0, 0, 0]

        # 3. 绘制 Free Space (但在 Room Mask 之外的) - 调试用 (白色)
        # 这样可以看出哪些地方是 Free 但没被分割成房间
        free_mask = (self.mm.free_bin > 0) & (room_mask == 0)
        viz[free_mask] = [255, 255, 255]

        # 4. 绘制轨迹
        cx, cz = pose[0, 3], pose[2, 3]
        u, v = self.mm.world_to_grid(cx, cz)
        cv2.circle(viz, (u, v), 5, (0, 0, 255), -1)

        # 绘制朝向
        vec_fwd = pose[:3, 2] * (-1)
        end_u = u + int(vec_fwd[0] * 20)
        end_v = v + int(vec_fwd[2] * 20)
        cv2.arrowedLine(viz, (u, v), (end_u, end_v), (0, 0, 255), 2, tipLength=0.3)

        text = f"F{self.current_floor_idx} | Room: {self.current_room_id}"
        cv2.putText(viz, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return viz

    def _render_empty_viz(self, pose):
        viz = np.ones((self.mm.map_h, self.mm.map_w, 3), dtype=np.uint8) * 240
        cx, cz = pose[0, 3], pose[2, 3]
        u, v = self.mm.world_to_grid(cx, cz)
        cv2.circle(viz, (u, v), 5, (0, 0, 255), -1)
        cv2.putText(viz, "Building Map...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return viz

class Sam3LocalClient:
    def __init__(self, config: dict, device='cuda'):
        self.device = device
        self.config = config.get('segmentation_module', {})

        # 路径设置
        self.sam3_root = SAM3_SOURCE_ROOT # 使用上面定义的常量
        self.bpe_path = self.config.get('bpe_path', os.path.join(self.sam3_root, "assets/bpe_simple_vocab_16e6.txt.gz"))
        self.checkpoint_path = self.config.get('checkpoint_path', "weights/sam3.pt")

        print(f"\n--- Loading SAM 3 Model (Batch Mode) ---")
        print(f"device: {self.device}")
        # 显存优化设置
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        try:
            # 构建模型
            self.model = build_sam3_image_model(
                bpe_path=self.bpe_path,
                checkpoint_path=self.checkpoint_path
            )
            self.model.to(self.device)
            self.model.eval()

            self.user_threshold = self.config.get('box_threshold', 0.20)

            # [关键] 构建 Transform Pipeline (使用导入的 API)
            # consistent_transform=False 是参考 notebook 的参数
            self.transform = ComposeAPI([
                RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

            print(f"SAM 3 Loaded. Default Threshold: {self.user_threshold}")
        except Exception as e:
            print(f"Error loading SAM3: {e}")
            raise e
        # [新增] 批次大小配置
        self.category_batch_size = self.config.get('category_batch_size', 16)
        print(f"[SAM3] Category batch size: {self.category_batch_size}")


         # --- 确保加载语义模型 (懒加载示例) ---
        if not hasattr(self, 'text_encoder'):
            print("  [Init] Loading SentenceTransformer for semantic check...")
            try:
                text_encoder_path = os.environ.get("INHERIT_SG_TEXT_ENCODER", "sentence-transformers/all-MiniLM-L6-v2")
                self.text_encoder = SentenceTransformer(text_encoder_path)
            except Exception as e:
                print(f"  [Warning] Failed to load local model: {e}. Semantic check will be disabled.")
                self.text_encoder = None

        # --- 新增：RAM++ 模型初始化 ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 路径根据实际情况修改
        ram_pretrained = os.environ.get(
            "INHERIT_SG_RAM_CHECKPOINT",
            "weights/ram_plus.pth",
        )
        self.ram_image_size = 384

        if config["use_predefined_object_classes"]:
            print("[Info] Using predefined object classes for")
            return

        if ram_plus is None:
            raise ImportError(
                "RAM++ is required when use_predefined_object_classes is false. "
                "Install recognize-anything or add it to PYTHONPATH."
            )


        try:
            self.ram_transform = get_transform(image_size=self.ram_image_size)
            self.ram_model = ram_plus(pretrained=ram_pretrained, image_size=self.ram_image_size, vit='swin_l')
            self.ram_model.eval()
            self.ram_model.to(self.device)
            print("[Init] RAM++ model loaded successfully.")
        except Exception as e:
            print(f"[Warning] RAM++ model failed to load: {e}")
            self.ram_model = None

        try:
            self.llm_handler = LLMHandler(config['llm_config'])
        except Exception as e:
            print(f"[Warning] LLM Handler failed to initialize: {e}")
            self.llm_handler = None


    def predict_masks(self, image_pil: Image.Image, prompts: list, threshold: float = None) -> Dict[str, List[np.ndarray]]:
        result = {p: [] for p in prompts}
        eff_threshold = threshold if threshold is not None else 0.10

        # 调用批量推理
        batch_results = self._batch_inference(image_pil, prompts, eff_threshold)

        for prompt in prompts:
            res = batch_results.get(prompt)
            if res is None: continue

            masks_np = res['masks']
            for i in range(masks_np.shape[0]):
                mask_binary = masks_np[i] > 0
                result[prompt].append(mask_binary)

        return result

    def _batch_inference(self, image_pil: Image.Image, prompts: list, threshold: float):
        """
        内部核心函数：执行一次性批量推理
        """
        # 1. 创建数据点 (调用全局定义的辅助函数)
        datapoint = create_empty_datapoint()
        set_image(datapoint, image_pil)

        # 2. 批量添加文本 Prompt
        query_map = {}
        for p in prompts:
            # 调用全局定义的辅助函数
            query_id = add_text_prompt(datapoint, p)
            query_map[p] = query_id

        # 3. 变换 (Transform)
        datapoint = self.transform(datapoint)

        # 4. Batch 组装 (Collate)
        # collate 接受一个 list
        batch = collate([datapoint], dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, torch.device(self.device), non_blocking=True)

        # 5. 模型推理
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model(batch)

        # 6. 后处理 (PostProcess)
        postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=threshold,
            to_cpu=True
        )

        processed_results = postprocessor.process_results(output, batch.find_metadatas)

        # 7. 结果映射
        final_results = {}
        for prompt, q_id in query_map.items():
            if q_id in processed_results:
                res = processed_results[q_id]
                if res['boxes'] is not None and len(res['boxes']) > 0:

                    # [修改开始] --------------------------------------------
                    # 1. 转换 boxes, scores 为 float32
                    # 2. 转换 masks 为 float32，并且去掉第1个维度 (1, H, W) -> (H, W)

                    masks_np = res['masks'].float().numpy()
                    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                        masks_np = masks_np.squeeze(1) # 变成 (N, H, W)

                    final_results[prompt] = {
                        'boxes': res['boxes'].float().numpy(),
                        'scores': res['scores'].float().numpy(),
                        'masks': masks_np  # 使用处理后的 masks
                    }
                    # [修改结束] --------------------------------------------

                else:
                    final_results[prompt] = None
            else:
                final_results[prompt] = None

        return final_results


    # 辅助函数：RLE 编码 (建议放在类外或作为静态方法)
    def _mask_to_rle(self, binary_mask):
        rle = {'counts': [], 'size': list(binary_mask.shape)}
        counts = rle.get('counts')
        for i, (value, length) in enumerate(self._run_length_encoding(binary_mask.flatten())):
            if i == 0 and value == 1:
                counts.append(0) # COCO RLE 格式要求以 0 (背景) 开始
            counts.append(length)
        return rle

    def _run_length_encoding(self, x):
        # 简单的行程编码生成器
        from itertools import groupby
        return ((x, len(list(y))) for x, y in groupby(x))

    def predict(self, image_pil: Image.Image, text_prompts: list,
                    using_predefined_classes: bool = False,
                    output_dir: str = None, file_id: str = "temp") -> str:

            all_detections = []

            image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            h_img, w_img = image_pil.size[1], image_pil.size[0]

            # [修改点 1] 增加 containment_dropped 目录
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                cropped_kept_dir = os.path.join(output_dir, "cropped_objects", "kept")
                cropped_dropped_dir = os.path.join(output_dir, "cropped_objects", "dropped")
                # [新增] 专门存放因包含+语义相似被剔除的图片
                cropped_contain_dir = os.path.join(output_dir, "cropped_objects", "containment_dropped")

                os.makedirs(cropped_kept_dir, exist_ok=True)
                os.makedirs(cropped_dropped_dir, exist_ok=True)
                os.makedirs(cropped_contain_dir, exist_ok=True)

                debug_vis_dir = os.path.join(output_dir, "debug_overlaps")
                os.makedirs(debug_vis_dir, exist_ok=True)


            # =========================================================
            # [Step 1] 确定 Prompts (保持不变)
            # =========================================================
            final_prompts = []
            current_batch_size = self.category_batch_size

            if using_predefined_classes:
                print(f"  [Mode] Using Predefined Classes ({len(text_prompts)} tags)")
                final_prompts = text_prompts
            else:
                print(f"  [Mode] Using RAM++ for Image Tagging...")
                if not hasattr(self, 'ram_model') or self.ram_model is None:
                    raise RuntimeError("RAM++ model is not initialized.")

                with torch.no_grad():
                    ram_img_tensor = self.ram_transform(image_pil).unsqueeze(0).to(self.device)

                    res = inference(ram_img_tensor, self.ram_model)
                    tags_str = res[0]
                    print(f"  [RAM++] Raw Tags: {tags_str}")
                    generated_tags = [tag.strip() for tag in tags_str.split('|') if tag.strip()]

                    if hasattr(self, 'llm_handler') and self.llm_handler:
                        print(f"  [Filter] Filtering {len(generated_tags)} raw tags via LLM...")
                        final_prompts = self.llm_handler.filter_object_tags(generated_tags)
                        print(f"  [Filter] {final_prompts} tags remain")
                    else:
                        print(f"  [Filter] No LLM handler found, skipping filtering.")
                        final_prompts = generated_tags

                    if len(final_prompts) > 0:
                        current_batch_size = len(final_prompts)
                        print(f"  [RAM++] Processed {len(final_prompts)} tags (Processing ALL at once).")

            if not final_prompts:
                print("  [Warning] No prompts found to process.")
                return ""

            # =========================================================
            # [Step 2] 目标检测推理
            # =========================================================
            total_prompts = len(final_prompts)
            print(f"  [SAM3] Processing {total_prompts} categories with batch size {current_batch_size}...")

            for batch_idx in range(0, total_prompts, current_batch_size):
                batch_prompts = final_prompts[batch_idx:batch_idx + current_batch_size]
                batch_results = self._batch_inference(image_pil, batch_prompts, self.user_threshold)

                for prompt in batch_prompts:
                    res = batch_results.get(prompt)
                    if res is None: continue

                    boxes = res['boxes']
                    scores = res['scores']
                    masks = res['masks']

                    for i, score in enumerate(scores):
                        x1, y1, x2, y2 = boxes[i]
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))

                        if (x2 - x1) < 5 or (y2 - y1) < 5: continue

                        mask_binary = masks[i] > 0
                        object_id_str = f"{prompt}_{len(all_detections)}"

                        temp_crop_data = None
                        temp_filename = ""

                        if output_dir:
                            cropped_object_cv2 = image_cv2[y1:y2, x1:x2]
                            mask_crop = mask_binary[y1:y2, x1:x2]

                            cropped_alpha = np.zeros((cropped_object_cv2.shape[0], cropped_object_cv2.shape[1], 4), dtype=np.uint8)
                            cropped_alpha[:, :, :3] = cropped_object_cv2
                            if mask_crop.shape == cropped_object_cv2.shape[:2]:
                                cropped_alpha[:, :, 3] = mask_crop * 255
                            else:
                                cropped_alpha[:, :, 3] = 255

                            safe_label = "".join(c for c in prompt if c.isalnum() or c in ('_', '-')).rstrip()
                            temp_filename = f"{safe_label}_{object_id_str}.png"
                            temp_crop_data = cropped_alpha

                        all_detections.append({
                            "object_id": object_id_str,
                            "label": prompt,
                            "score": float(score),
                            "bounding_box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                            "cropped_image_path": "",
                            # [修改点 2] 暂存 Mask 和 Image 数据
                            "_temp_crop_data": temp_crop_data,
                            "_temp_filename": temp_filename,
                            "_temp_mask": mask_binary, # 暂存完整的二进制 Mask
                            "_drop_reason": None # 记录被剔除的原因
                        })

                torch.cuda.empty_cache()

            # =========================================================
            # [Post-Processing] NMS + Mask Containment + Semantic Check
            # =========================================================

            # 1. 按照分数从高到低排序
            all_detections.sort(key=lambda x: x['score'], reverse=True)

            num_dets = len(all_detections)
            keep = [True] * num_dets
            overlap_threshold = 0.85

            # [新增] 语义相似度阈值 & 包含率阈值
            SEMANTIC_THRESH = 0.6
            CONTAIN_THRESH = 0.90

            print(f"\n  [NMS] Checking overlaps & containment across {num_dets} detections...")

            if num_dets > 0:
                for i in range(num_dets):
                    if not keep[i]: continue

                    for j in range(i + 1, num_dets):
                        if not keep[j]: continue

                        det_a = all_detections[i] # High Score
                        det_b = all_detections[j] # Low Score

                        # --- 1. 标准 Box IoU ---
                        bbox_a = det_a['bounding_box_xyxy']
                        bbox_b = det_b['bounding_box_xyxy']
                        iou = self.calculate_iou(bbox_a, bbox_b)

                        if iou > overlap_threshold:
                            keep[j] = False # 标准 IoU 剔除低分
                            det_b['_drop_reason'] = "iou_dropped"
                            continue

                        # --- 2. [新增] Mask Containment + Semantic Check ---
                        # 计算 Mask 包含率
                        mask_a = det_a['_temp_mask']
                        mask_b = det_b['_temp_mask']

                        intersection = np.logical_and(mask_a, mask_b).sum()
                        area_a = mask_a.sum()
                        area_b = mask_b.sum()

                        if area_a == 0 or area_b == 0: continue

                        # 计算包含比率
                        ratio_b_in_a = intersection / area_b # B 在 A 里面
                        ratio_a_in_b = intersection / area_a # A 在 B 里面

                        is_contained = False
                        container_is_a = True # 谁是那个包含者（大物体）

                        if ratio_b_in_a > CONTAIN_THRESH:
                            is_contained = True
                            container_is_a = True # A 是大的
                        elif ratio_a_in_b > CONTAIN_THRESH:
                            is_contained = True
                            container_is_a = False # B 是大的

                        if is_contained and self.text_encoder:
                            # 计算语义相似度
                            label_a = det_a['label']
                            label_b = det_b['label']

                            emb = self.text_encoder.encode([label_a, label_b], convert_to_tensor=True)
                            sim_score = util.cos_sim(emb[0], emb[1]).item()

                            if sim_score > SEMANTIC_THRESH:
                                # 包含且语义相似 -> 保留面积大的那个！

                                target_drop_idx = -1
                                drop_label = ""
                                keep_label = ""

                                if container_is_a:
                                    # A 是大物体 (分数也高)，剔除 B (小物体)
                                    target_drop_idx = j
                                    drop_label = label_b
                                    keep_label = label_a
                                    det_b['_drop_reason'] = "containment_dropped"
                                else:
                                    # B 是大物体 (但分数低)，A 是小物体 (分数高)
                                    # 此时我们遇到了一个冲突：按分数该保留 A，按面积该保留 B。
                                    # 策略：为了获得完整物体，我们剔除 A (High Score Small Part)。
                                    target_drop_idx = i
                                    drop_label = label_a
                                    keep_label = label_b
                                    det_a['_drop_reason'] = "containment_dropped"

                                keep[target_drop_idx] = False

                                print(f"    ❌ [Containment Drop] '{drop_label}' is part of '{keep_label}' "
                                      f"(Sim: {sim_score:.2f}). Dropping smaller part.")

                                # 如果删除了 i，必须 break 内部循环，因为 i 已经无效了
                                if target_drop_idx == i:
                                    break

            # [修改点 3] 根据 keep 状态保存图片到不同目录
            if output_dir:
                print(f"  [Save] Saving crops...")
                for i, det in enumerate(all_detections):
                    img_data = det.pop('_temp_crop_data', None)
                    fname = det.pop('_temp_filename', None)
                    drop_reason = det.pop('_drop_reason', None)

                    if img_data is not None and fname:
                        if keep[i]:
                            # Case A: 保留
                            save_path = os.path.join(cropped_kept_dir, fname)
                            try:
                                cv2.imwrite(save_path, img_data)
                                det['cropped_image_path'] = os.path.join("cropped_objects", "kept", fname)
                            except: pass
                        else:
                            # Case B: 被剔除
                            if drop_reason == "containment_dropped":
                                # 包含关系剔除 -> 存入 specialized 文件夹便于 Debug
                                save_path = os.path.join(cropped_contain_dir, fname)
                            else:
                                # 普通 IoU 剔除
                                save_path = os.path.join(cropped_dropped_dir, fname)

                            try:
                                cv2.imwrite(save_path, img_data)
                            except: pass
            else:
                for det in all_detections:
                    det.pop('_temp_crop_data', None)
                    det.pop('_temp_filename', None)
                    det.pop('_drop_reason', None)

            # 4. 过滤列表并生成 Mask RLE


            # [新增] 初始化小物体存放目录
            cropped_small_dir = ""
            if output_dir:
                cropped_small_dir = os.path.join(output_dir, "cropped_objects", "small_dropped")
                os.makedirs(cropped_small_dir, exist_ok=True)

            # [新增] 尺寸阈值 (针对 1080x720 分辨率)
            MIN_AREA = 1500  # 约 28x28 像素
            MIN_SIDE = 30   # 最小边长

            # 4. 过滤列表 + 尺寸检查 + 生成 Mask RLE
            final_detections = []

            # 注意：此时 self.all_detections 中的图片数据已经被上面的 "Save Saving crops..." 循环 pop 掉了
            # 如果我们想在这一步保存 "small_dropped"，我们需要调整逻辑顺序。

            # === [重要修正] ===
            # 上面的 "[Save] Saving crops..." 循环已经把图片写入磁盘并 pop 掉了内存数据。
            # 如果那是 "kept" 的图片，它已经写入了 cropped_kept_dir。
            # 我们现在要对这些 "kept" 的结果进行二次过滤。
            # 如果判定太小，我们需要：
            # 1. 从 final_detections 剔除
            # 2. 将已经保存到 kept 的图片【移动】到 small_dropped 目录
            # 3. 更新/删除 JSON 中的 path

            import shutil # 需要导入 shutil

            count_small_removed = 0

            for k, det in zip(keep, all_detections):
                mask_bin = det.pop('_temp_mask', None)

                if k:
                    # 检查尺寸
                    x1, y1, x2, y2 = det['bounding_box_xyxy']
                    w_box = x2 - x1
                    h_box = y2 - y1
                    area = w_box * h_box

                    if area < MIN_AREA or w_box < MIN_SIDE or h_box < MIN_SIDE:
                        # [判定为太小] -> 剔除
                        count_small_removed += 1

                        # 处理图片移动 (如果 output_dir 存在)
                        if output_dir and 'cropped_image_path' in det:
                            rel_path = det['cropped_image_path']
                            full_src_path = os.path.join(output_dir, rel_path)

                            if os.path.exists(full_src_path):
                                fname = os.path.basename(full_src_path)
                                full_dst_path = os.path.join(cropped_small_dir, fname)
                                try:
                                    shutil.move(full_src_path, full_dst_path)
                                except Exception as e:
                                    print(f"Warning: Failed to move small object image: {e}")

                        continue # 跳过，不加入 final_detections

                    # [保留] -> 加入最终结果
                    if mask_bin is not None:
                        det['segmentation'] = self._mask_to_rle(mask_bin)
                    final_detections.append(det)

            print(f"  [Size Filter] Removed {count_small_removed} small objects (Area < {MIN_AREA}).")
            print(f"  [NMS] Finished. {num_dets} -> {len(final_detections)} objects remaining.")


            # 保存 JSON
            json_path = ""
            if output_dir:
                json_path = os.path.join(output_dir, f"pred_{file_id}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    # 注意：mask RLE 包含大量数字，生成的 json 会变大
                    json.dump(final_detections, f, indent=4, ensure_ascii=False)

            return json_path

    # def predict(self, image_pil: Image.Image, text_prompts: list,
    #             using_predefined_classes: bool = False,
    #             output_dir: str = None, file_id: str = "temp") -> str:

    #     all_detections = []

    #     image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    #     h_img, w_img = image_pil.size[1], image_pil.size[0]

    #     # [修改点 1] 创建 kept 和 dropped 两个子目录
    #     if output_dir:
    #         os.makedirs(output_dir, exist_ok=True)
    #         # 定义两个路径
    #         cropped_kept_dir = os.path.join(output_dir, "cropped_objects", "kept")
    #         cropped_dropped_dir = os.path.join(output_dir, "cropped_objects", "dropped")
    #         os.makedirs(cropped_kept_dir, exist_ok=True)
    #         os.makedirs(cropped_dropped_dir, exist_ok=True)

    #         debug_vis_dir = os.path.join(output_dir, "debug_overlaps")
    #         os.makedirs(debug_vis_dir, exist_ok=True)
    #     # =========================================================
    #     # [Step 1] 确定 Prompts 和 批处理策略
    #     # =========================================================
    #     final_prompts = []
    #     current_batch_size = self.category_batch_size # 默认批次大小

    #     if using_predefined_classes:
    #         # --- 模式 A: 使用预定义类别 (分批次) ---
    #         print(f"  [Mode] Using Predefined Classes ({len(text_prompts)} tags)")
    #         final_prompts = text_prompts
    #         # 保持 current_batch_size 不变
    #     else:
    #         # --- 模式 B: 使用 RAM++ 生成标签 (一次性处理) ---
    #         print(f"  [Mode] Using RAM++ for Image Tagging...")

    #         if not hasattr(self, 'ram_model') or self.ram_model is None:
    #             raise RuntimeError("RAM++ model is not initialized. Please check __init__.")

    #         with torch.no_grad():
    #             # RAM++ 推理
    #             ram_img_tensor = self.ram_transform(image_pil).unsqueeze(0).to(self.device)
    #             # 假设 inference 函数已导入: from ram import inference_ram as inference
    #             # 如果是类成员函数请调整为 self.inference(...)
    #             res = inference(ram_img_tensor, self.ram_model)

    #             tags_str = res[0] # e.g., "chair | table | lamp"
    #             print(f"  [RAM++] Raw Tags: {tags_str}")

    #             # 解析标签
    #             generated_tags = [tag.strip() for tag in tags_str.split('|') if tag.strip()]



    #             if hasattr(self, 'llm_handler') and self.llm_handler:
    #                 print(f"  [Filter] Filtering {len(generated_tags)} raw tags via LLM...")
    #                 final_prompts = self.llm_handler.filter_object_tags(generated_tags)
    #                 print(f"  [Filter] {final_prompts} tags remain")
    #             else:
    #                 print(f"  [Filter] No LLM handler found, skipping filtering.")
    #                 final_prompts = generated_tags

    #             # *** 关键修改：RAM++模式下，强制一次性处理所有标签 ***
    #             if len(final_prompts) > 0:
    #                 current_batch_size = len(final_prompts)
    #                 print(f"  [RAM++] Processed {len(final_prompts)} tags (Processing ALL at once).")
    #             # --- [新增] 调用 LLM 过滤标签 ---

    #     if not final_prompts:
    #         print("  [Warning] No prompts found to process.")
    #         return ""

    #     # =========================================================
    #     # [Step 2] 目标检测推理
    #     # =========================================================
    #     total_prompts = len(final_prompts)
    #     print(f"  [SAM3] Processing {total_prompts} categories with batch size {current_batch_size}...")

    #     # 使用 final_prompts 和 current_batch_size 进行循环
    #     # 如果是 RAM++ 模式，这里的循环只会执行一次 (因为 batch_size == total_prompts)
    #     for batch_idx in range(0, total_prompts, current_batch_size):
    #         batch_prompts = final_prompts[batch_idx:batch_idx + current_batch_size]

    #         # 执行推理
    #         batch_results = self._batch_inference(image_pil, batch_prompts, self.user_threshold)

    #         # 收集结果
    #         for prompt in batch_prompts:
    #             res = batch_results.get(prompt)
    #             if res is None: continue

    #             boxes = res['boxes']
    #             scores = res['scores']
    #             masks = res['masks']

    #             for i, score in enumerate(scores):
    #                 x1, y1, x2, y2 = boxes[i]
    #                 x1, y1 = max(0, int(x1)), max(0, int(y1))
    #                 x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))

    #                 if (x2 - x1) < 5 or (y2 - y1) < 5: continue

    #                 mask_binary = masks[i] > 0
    #                 object_id_str = f"{prompt}_{len(all_detections)}"

    #                 # [修改点 2] 生成图像数据，但暂存到内存，不立即保存
    #                 temp_crop_data = None
    #                 temp_filename = ""

    #                 if output_dir:
    #                     cropped_object_cv2 = image_cv2[y1:y2, x1:x2]
    #                     mask_crop = mask_binary[y1:y2, x1:x2]

    #                     cropped_alpha = np.zeros((cropped_object_cv2.shape[0], cropped_object_cv2.shape[1], 4), dtype=np.uint8)
    #                     cropped_alpha[:, :, :3] = cropped_object_cv2

    #                     if mask_crop.shape == cropped_object_cv2.shape[:2]:
    #                         cropped_alpha[:, :, 3] = mask_crop * 255
    #                     else:
    #                         cropped_alpha[:, :, 3] = 255

    #                     safe_label = "".join(c for c in prompt if c.isalnum() or c in ('_', '-')).rstrip()
    #                     temp_filename = f"{safe_label}_{object_id_str}.png"
    #                     temp_crop_data = cropped_alpha # 暂存 numpy 数组

    #                 # 添加到列表，包含临时数据
    #                 all_detections.append({
    #                     "object_id": object_id_str,
    #                     "label": prompt,
    #                     "score": float(score),
    #                     "bounding_box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
    #                     "cropped_image_path": "", # 路径稍后回填
    #                     # --- 私有临时字段 ---
    #                     "_temp_crop_data": temp_crop_data,
    #                     "_temp_filename": temp_filename
    #                 })


    #         torch.cuda.empty_cache()
    #     # =========================================================
    #     # [Post-Processing] NMS (非极大值抑制) 去除重复框
    #     # =========================================================

    #     # 1. 按照分数从高到低排序 (这对 NMS 很重要)
    #     all_detections.sort(key=lambda x: x['score'], reverse=True)

    #     num_dets = len(all_detections)
    #     keep = [True] * num_dets  # 标记是否保留
    #     overlap_threshold = 0.85  # 重叠阈值

    #     print(f"\n  [NMS] Checking for overlaps across {num_dets} detections (Threshold: {overlap_threshold})...")

    #     if output_dir and num_dets > 0:
    #         # 2. 双重循环检查重叠
    #         for i in range(num_dets):
    #             if not keep[i]: continue  # 如果 i 已经被剔除，跳过

    #             for j in range(i + 1, num_dets):
    #                 if not keep[j]: continue  # 如果 j 已经被剔除，跳过

    #                 det_a = all_detections[i] # 分数较高 (因为排过序)
    #                 det_b = all_detections[j] # 分数较低

    #                 bbox_a = det_a['bounding_box_xyxy']
    #                 bbox_b = det_b['bounding_box_xyxy']

    #                 iou = self.calculate_iou(bbox_a, bbox_b)

    #                 if iou > overlap_threshold:
    #                     # 判定为重复，剔除分数低的 j
    #                     keep[j] = False

    #                     id_a = det_a['object_id']
    #                     id_b = det_b['object_id']
    #                     score_a = det_a['score']
    #                     score_b = det_b['score']

    #                     print(f"    ❌ [Drop] {id_b}({score_b:.2f}) overlaps with {id_a}({score_a:.2f}) | IoU: {iou:.3f}")

    #                     # --- 可视化重复项 (仅在 output_dir 存在时) ---
    #                     if output_dir:
    #                         img_vis = image_cv2.copy()

    #                         # 画框 A (保留 - 绿色)
    #                         cv2.rectangle(img_vis, (bbox_a[0], bbox_a[1]), (bbox_a[2], bbox_a[3]), (0, 255, 0), 3)
    #                         text_a = f"KEEP: {id_a} ({score_a:.2f})"
    #                         cv2.putText(img_vis, text_a, (bbox_a[0], max(20, bbox_a[1]-10)),
    #                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    #                         # 画框 B (剔除 - 红色)
    #                         cv2.rectangle(img_vis, (bbox_b[0], bbox_b[1]), (bbox_b[2], bbox_b[3]), (0, 0, 255), 3)
    #                         text_b = f"DROP: {id_b} ({score_b:.2f})"
    #                         # 为了防止文字重叠，把B的文字稍微向下移一点或者画在底部
    #                         cv2.putText(img_vis, text_b, (bbox_b[0], bbox_b[3] + 20),
    #                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    #                         # 在左上角写 IoU
    #                         cv2.putText(img_vis, f"IoU Collision: {iou:.3f}", (20, 30),
    #                                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    #                         safe_a = "".join(c for c in id_a if c.isalnum())
    #                         safe_b = "".join(c for c in id_b if c.isalnum())


    #                         filename = f"nms_drop_{i}_{j}_{safe_b}_vs_{safe_a}.jpg"
    #                         save_path = os.path.join(debug_vis_dir, filename)
    #                         cv2.imwrite(save_path, img_vis)

    #     # [修改点 3] 根据 keep 状态保存图片到不同目录
    #     if output_dir:
    #         print(f"  [Save] Saving crops to kept/dropped folders...")
    #         for i, det in enumerate(all_detections):
    #             # 取出并移除临时数据，防止这些字段进入 JSON
    #             img_data = det.pop('_temp_crop_data', None)
    #             fname = det.pop('_temp_filename', None)

    #             if img_data is not None and fname:
    #                 if keep[i]:
    #                     # Case A: 保留的物体 -> 存入 kept
    #                     save_path = os.path.join(cropped_kept_dir, fname)
    #                     try:
    #                         cv2.imwrite(save_path, img_data)
    #                         # 回填 JSON 路径 (相对于 output_dir)
    #                         det['cropped_image_path'] = os.path.join("cropped_objects", "kept", fname)
    #                     except Exception: pass
    #                 else:
    #                     # Case B: 被剔除的物体 -> 存入 dropped
    #                     save_path = os.path.join(cropped_dropped_dir, fname)
    #                     try:
    #                         cv2.imwrite(save_path, img_data)
    #                         # 被剔除的物体不需要更新路径，因为它们通常不写入 JSON
    #                     except Exception: pass
    #     else:
    #         # 如果没有 output_dir，也要清理临时数据以防万一
    #         for det in all_detections:
    #             det.pop('_temp_crop_data', None)
    #             det.pop('_temp_filename', None)

    #     # 4. 过滤列表，只保留 keep 为 True 的项
    #     final_detections = [d for k, d in zip(keep, all_detections) if k]

    #     print(f"  [NMS] Finished. {num_dets} -> {len(final_detections)} objects remaining.")


    #     # 保存 JSON (使用过滤后的 final_detections)
    #     json_path = ""
    #     if output_dir:
    #         json_path = os.path.join(output_dir, f"pred_{file_id}.json")
    #         with open(json_path, 'w', encoding='utf-8') as f:
    #             json.dump(final_detections, f, indent=4, ensure_ascii=False)

    #     return json_path


    def calculate_iou(self,box1, box2):
        """计算两个 bbox 的 IoU"""
        x1_a, y1_a, x2_a, y2_a = box1
        x1_b, y1_b, x2_b, y2_b = box2

        x1_inter = max(x1_a, x1_b)
        y1_inter = max(y1_a, y1_b)
        x2_inter = min(x2_a, x2_b)
        y2_inter = min(y2_a, y2_b)

        inter_w = max(0, x2_inter - x1_inter)
        inter_h = max(0, y2_inter - y1_inter)
        inter_area = inter_w * inter_h

        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)

        union_area = area_a + area_b - inter_area
        if union_area == 0: return 0
        return inter_area / union_area



# [新增] 放在 MapManager 类定义之后，TrajectoryPipeline 之前
from .floor_tracking import FloorTracker

        # 这里的逻辑是：Pipeline 调用 finalize_floor 后，
        # 会显式调用 MapManager.switch_floor(self.current_floor_id)
        # 所以这里只需要更新 self.current_floor_id 即可

# ==============================================================================
# Pipeline Class
# ==============================================================================

class TrajectoryPipeline:
    def __init__(self, config_path: str, override_settings: Dict = None, shared_models: Dict = None):
        # =========================================================
        # 1. 配置加载与覆盖 (Config Loading & Overrides)
        # =========================================================
        if isinstance(config_path, str):
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found at '{config_path}'")
            with open(config_path, 'r') as f: self.config = yaml.safe_load(f)
        else:
            self.config = config_path

        # 应用动态覆盖 (必须最先执行，因为这决定了 run_dir)
        if override_settings:
            if 'trajectory_data' in override_settings:
                self.config['trajectory_data'].update(override_settings['trajectory_data'])
            if 'main_output_directory' in override_settings:
                self.config['main_output_directory'] = override_settings['main_output_directory']
            if 'debug_mode' in override_settings:
                self.config['debug_mode'] = override_settings['debug_mode']

        # =========================================================
        # 2. 路径初始化 (Path Initialization) - 修复 AttributeError 的关键
        # =========================================================
        self.run_dir = self.config['main_output_directory']
        self.main_output_directory = self.run_dir
        os.makedirs(self.run_dir, exist_ok=True)

        self.external_context = None
        traj_cfg = self.config.get('trajectory_data', {})
        self.trajectory_data_dir = traj_cfg.get('base_dir', './temp_data')
        self.filename_prefix = traj_cfg.get('filename_prefix', "")

        self.rgb_dir = os.path.join(self.trajectory_data_dir, traj_cfg.get('rgb_subdir', 'rgb'))
        self.depth_dir = os.path.join(self.trajectory_data_dir, traj_cfg.get('depth_subdir', 'depth'))
        self.pose_dir = os.path.join(self.trajectory_data_dir, traj_cfg.get('pose_subdir', 'pose'))

        for directory in (self.rgb_dir, self.depth_dir, self.pose_dir):
            if not os.path.isdir(directory):
                raise FileNotFoundError(directory)




        # 初始化 Logger


        # 保存 Config 快照 (Retriever 需要这个来找回原始数据路径)
        with open(os.path.join(self.run_dir, "config_snapshot.yaml"), 'w') as f:
            yaml.dump(self.config, f)

        # =========================================================
        # 3. Debug 配置 (依赖 run_dir)
        # =========================================================
        self.debug_mode = self.config.get('debug_mode', False)

        # SAM3 输出目录
        self.sam3_output_root = os.path.join(self.run_dir, "sam3_outputs")
        if self.debug_mode:
            os.makedirs(self.sam3_output_root, exist_ok=True)

        self.debug_config = self.config.get('debug_config', {})

        # 在线建图 Debug 目录
        self.online_map_viz_dir = os.path.join(self.run_dir, "debug_online_mapping")
        self.room_viz_dir = os.path.join(self.run_dir, "debug_room_segmentation")
        if self.debug_mode:
            os.makedirs(self.online_map_viz_dir, exist_ok=True)
            os.makedirs(self.room_viz_dir, exist_ok=True)

        # Re-ID 可视化目录
        self.enable_reid_viz = self.debug_mode and self.debug_config.get('save_reid_visualizations', False)
        self.reid_viz_dir = os.path.join(self.run_dir, "debug_reid_viz") if self.enable_reid_viz else None
        if self.reid_viz_dir: os.makedirs(self.reid_viz_dir, exist_ok=True)

        # LLM 标注图目录
        self.save_annotated_frames = self.debug_mode and self.debug_config.get('save_llm_annotated_frames', False)
        self.annotated_frames_dir = os.path.join(self.run_dir, "debug_annotated_frames") if self.save_annotated_frames else None
        if self.annotated_frames_dir: os.makedirs(self.annotated_frames_dir, exist_ok=True)

        # [新增] VLM 区域总结图片保存目录
        self.room_summary_img_dir = os.path.join(self.run_dir, "room_summary_images")
        os.makedirs(self.room_summary_img_dir, exist_ok=True)
        # =========================================================
        # 4. 核心组件初始化
        # =========================================================
        self.spatial_graph = nx.DiGraph()
        self.robot_trajectory = []
        self.memory_kb = []
        self.label_counters = {}
        self.floor_trajectories = {}
        self.room_frame_history = {}
        self.pose_cache = {}
        self.room_metadata_cache = {}
        self.room_image_buffer = {}
        self.current_room_id = 0
        self.room_topology_cache = {}
        self.room_exit_buffer = {}

        # [Fix] 状态变量初始化
        self.visited_room_ids = set()
        self.global_update_history = []
        self.last_kf_feature = None

        # [新增] 用于检测房间切换
        self.last_room_id = 0

        # 语义森林文件路径
        self.nodes_path = os.path.join(self.run_dir, self.config['semantic_forest']['nodes_file'])
        self.edges_path = os.path.join(self.run_dir, self.config['semantic_forest']['edges_file'])

        # 初始化空文件
        open(self.nodes_path, 'w').close()
        with open(self.edges_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["child_node_id", "parent_node_id", "node_level"])

        # 参数读取
        reid_cfg = self.config.get('reidentification', {})
        self.reid_sim_threshold = reid_cfg.get('similarity_threshold', 0.75)
        self.reid_spatial_threshold = reid_cfg.get('spatial_threshold', 0.5)

        spatial_cfg = self.config.get('spatial_reasoning', {})
        self.spatial_cluster_thresh = spatial_cfg.get('clustering_distance_threshold', 1.0)

        self.finalize_delay_kf = int(self.config.get("room_segmentation", {}).get("finalize_delay_kf", 20))

        # 轨迹与姿态路径
        self.traj_base_dir = self.config['trajectory_data']['base_dir']
        self.pose_dir = os.path.join(self.traj_base_dir, traj_cfg.get("pose_subdir", "pose"))

        # 相机内参
        cam_cfg = self.config.get('camera_intrinsics', {})
        self.intrinsics = {
            'fx': cam_cfg.get('fx', 540.0),
            'fy': cam_cfg.get('fy', 540.0),
            'cx': cam_cfg.get('cx', 540.0),
            'cy': cam_cfg.get('cy', 360.0),
            'depth_scale': cam_cfg.get('depth_scale', 1000.0)
        }
        print(self.intrinsics)

        # 建图与分割器
        self.map_output_dir = os.path.join(self.run_dir, "trajectory_maps")
        os.makedirs(self.map_output_dir, exist_ok=True)
        self.map_manager = MapManager(self.config.get('map_config', {}), self.intrinsics, self.run_dir)
        self.room_segmenter = GeometryRoomTracker(self.map_manager, self.config.get('room_segmentation', {}))
        initial_floor = self.config.get("floor_tracking", {}).get("initial_floor", 0)
        self.floor_tracker = FloorTracker(self.nodes_path, self.edges_path, initial_floor=initial_floor)
        self.map_manager.current_floor_idx = initial_floor
        self.room_segmenter.current_floor_idx = initial_floor

        # 设备设置
        # Device selection is supplied by the user configuration.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        # =========================================================
        # [修改] 多GPU模型加载策略
        # =========================================================
        self.gpu_config = self.config.get('multi_gpu', {
            'enabled': True,
            'sam3_device': 'cuda:0',      # SAM3 放在 GPU 0
            'dinov3_device': 'cuda:0',    # DINOv3 放在 GPU 1
            'text_encoder_device': 'cuda:0'  # Text Encoder 放 CPU 或 GPU 2
        })

        if self.gpu_config['enabled']:
            print("[Multi-GPU] Enabled. Model distribution:")
            for k, v in self.gpu_config.items():
                if k != 'enabled':
                    print(f"  {k}: {v}")

        # 设置各模型的设备
        sam3_device = self.gpu_config.get('sam3_device', 'cuda:0') if self.gpu_config['enabled'] else self.device
        dinov3_device = self.gpu_config.get('dinov3_device', 'cuda:0') if self.gpu_config['enabled'] else self.device
        text_device = self.gpu_config.get('text_encoder_device', 'cuda:0') if self.gpu_config['enabled'] else self.device

        # =========================================================
        # 模型加载 (使用不同设备)
        # =========================================================
        if shared_models and 'dinov3' in shared_models:
            print("[Pipeline] Using shared DINOv3 model.")
            self.dinov3_model = shared_models['dinov3']
            # 如果启用多GPU,需要移动模型
            if self.gpu_config['enabled']:
                self.dinov3_model = self.dinov3_model.to(dinov3_device)
        else:
            print(f"[Pipeline] Loading DINOv3 model on {dinov3_device}...")
            self.dinov3_model = load_feature_extractor(self.config, dinov3_device)

        if shared_models and 'text_encoder' in shared_models:
            self.text_encoder = shared_models['text_encoder']
            if self.gpu_config['enabled']:
                self.text_encoder = self.text_encoder.to(text_device)
        else:
            from sentence_transformers import SentenceTransformer
            self.text_encoder = SentenceTransformer(
                os.environ.get("INHERIT_SG_TEXT_ENCODER", "sentence-transformers/all-MiniLM-L6-v2"),
                device=text_device,
            )

        if shared_models and 'sam3' in shared_models:
            print("[Pipeline] Using shared SAM 3 client.")
            self.sam3_client = shared_models['sam3']
        else:
            print(f"[Pipeline] Loading SAM 3 client on {sam3_device}...")
            seg_config = self.config.get("segmentation_module", {})
            self.sam3_client = Sam3LocalClient(self.config, device=sam3_device)

        # 保存设备信息供后续使用
        self.sam3_device = sam3_device
        self.dinov3_device = dinov3_device
        self.text_device = text_device

        print(f"filter_edge_detections: {self.config['filter_edge_detections']}")

        try: self.llm = LLMHandler(self.config['llm_config'])
        except NameError: pass

        self.kf_config = self.config.get('keyframe_detection', {})
        self.enable_kf = self.kf_config.get('enable_keyframe_detection', False)
        self.kf_threshold = self.kf_config.get('keyframe_similarity_threshold', 0.98)
        self.last_kf_feature = None



    def _annotate_image_with_clusters(self, image_path: str, nodes_in_frame: List[Dict],
                                      clusters: List[List[str]], frame_index: int) -> str:
        """
        [新增] 根据聚类结果绘制图像：
        - 同一聚类的物体使用相同颜色。
        - 只有属于同一聚类的物体才会被画在同一张图上供 LLM 分析 (或者画在一张图上但颜色区分)。
        - 这里实现：在一张图上画出所有物体，但用颜色区分聚类。
        """
        img = cv2.imread(image_path)
        if img is None: return image_path

        # 生成一组颜色 (BGR)
        # 预定义一些鲜明的颜色，不够随机生成
        base_colors = [
            (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
            (255, 0, 255), (255, 255, 0), (128, 0, 128), (0, 128, 128)
        ]

        # 建立 node_id -> color 的映射
        node_color_map = {}
        for i, cluster in enumerate(clusters):
            color = base_colors[i % len(base_colors)]
            # 如果颜色不够用，随机生成
            if i >= len(base_colors):
                color = np.random.randint(50, 255, size=3).tolist()

            for node_id in cluster:
                node_color_map[node_id] = color

        font = cv2.FONT_HERSHEY_SIMPLEX

        for node in nodes_in_frame:
            node_id = node['node_id']
            # 如果物体没有被聚类（例如没有3D坐标），给一个灰色
            color = node_color_map.get(node_id, (128, 128, 128))

            x1, y1, x2, y2 = map(int, node['bbox'])

            # 画框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            # 画标签背景和文字
            label_text = node_id
            text_size, _ = cv2.getTextSize(label_text, font, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - text_size[1] - 8), (x1 + text_size[0] + 4, y1), color, -1)
            # 文字用白色或黑色，取决于背景深浅，这里简单用白色
            cv2.putText(img, label_text, (x1 + 2, y1 - 5), font, 0.6, (255, 255, 255), 1)

        if self.save_annotated_frames:
            filename = f"frame_{frame_index:04d}_clustered.jpg"
            save_path = os.path.join(self.annotated_frames_dir, filename)
        else:
            save_path = os.path.splitext(image_path)[0] + "_temp_spatial.jpg"

        cv2.imwrite(save_path, img)
        return save_path

    def _get_image_path(self, frame_index: int) -> Tuple[str, str]:
        traj_cfg = self.config['trajectory_data']
        file_id = traj_cfg['filename_start_index'] + frame_index
        stem = f"{traj_cfg['filename_prefix']}{file_id:0{traj_cfg['filename_padding']}}"
        rgb_extension = str(traj_cfg.get('rgb_extension', '.png'))
        if not rgb_extension.startswith('.'):
            rgb_extension = '.' + rgb_extension
        return (
            os.path.join(traj_cfg['base_dir'], traj_cfg['rgb_subdir'], stem + rgb_extension),
            os.path.join(traj_cfg['base_dir'], traj_cfg['depth_subdir'], stem + '.png'),
        )

    def _get_pose_path(self, frame_index: int) -> str:
        """根据帧号获取 Pose 文件路径"""
        traj_cfg = self.config['trajectory_data']
        file_id = traj_cfg['filename_start_index'] + frame_index
        image_filename = f"{traj_cfg['filename_prefix']}{file_id:0{traj_cfg['filename_padding']}}.txt"
        return os.path.join(traj_cfg['base_dir'], traj_cfg['pose_subdir'], image_filename)

    def _run_segmentation(self, frame_index: int, rgb_path: str, depth_path: str) -> str:
        """
        调用 SAM3 进行推理，并统计耗时。
        """
        image_pil = Image.open(rgb_path).convert("RGB")

        # 解析 prompt
        prompts_raw = self.config["predefined_object_classes"]
        if isinstance(prompts_raw, str):
            prompts = [p.strip() for p in prompts_raw.split('.') if p.strip()]
        else:
            prompts = prompts_raw

        # [修改] 根据 debug 开关决定输出路径
        if self.debug_mode:
            # 原来的方式：每帧一个文件夹
            frame_output_dir = os.path.join(self.sam3_output_root, f"frame_{frame_index:04d}")
        else:
            # 非 Debug 模式：建议输出到一个临时目录，或者统一的一个目录，用完即删（如果后续不需要）
            # 但由于 OfflineRetriever 可能需要读取 crop 图片 (memory_kb 存了路径)，所以 crop 必须保留！
            # 我们可以把 crop 统一放在 objects 文件夹下，不再每帧建一个文件夹
            frame_output_dir = os.path.join(self.run_dir, "objects", f"frame_{frame_index:04d}")

        # 确保目录存在
        os.makedirs(frame_output_dir, exist_ok=True)

        print(f"  [SAM3] Frame {frame_index} inference...", flush=True)
        sam_start_time = time.time()

        # 调用 SAM3 Client
        json_output_path = self.sam3_client.predict(
            image_pil=image_pil,
            text_prompts=prompts,
            using_predefined_classes = self.config['use_predefined_object_classes'],
            output_dir=frame_output_dir,
            file_id=f"frame_{frame_index:04d}"
        )

        sam_duration = time.time() - sam_start_time
        print(f"  [SAM3] Inference Complete. Duration: {sam_duration:.4f}s. Output: {os.path.basename(frame_output_dir)}", flush=True)

        # 如果没有检测到物体，predict 可能返回 "" 或不生成文件
        if not json_output_path or not os.path.exists(json_output_path):
            print("  [SAM3] No objects detected.", flush=True)
            empty_json_path = os.path.join(self.run_dir, f"frame_{frame_index:04d}_empty.json")
            with open(empty_json_path, 'w') as f:
                json.dump([], f)
            return empty_json_path

        return json_output_path


    def _visualize_reid_match(self, old_image_path: str, old_bbox: List[int],
                              new_image_path: str, new_bbox: List[int],
                              old_node_id: str, new_obj_id: int, score: float, frame_index: int):
        if not self.enable_reid_viz or not self.reid_viz_dir: return
        try:
            img_old, img_new = cv2.imread(old_image_path), cv2.imread(new_image_path)
            if img_old is None or img_new is None: return

            if old_bbox:
                x1, y1, x2, y2 = map(int, old_bbox)
                cv2.rectangle(img_old, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img_old, f"KB: {old_node_id}", (x1, max(y1-10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if new_bbox:
                x1, y1, x2, y2 = map(int, new_bbox)
                cv2.rectangle(img_new, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(img_new, f"New: {new_obj_id}", (x1, max(y1-10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            h_new, w_new = img_new.shape[:2]
            h_old, w_old = img_old.shape[:2]
            if h_old == 0: return
            scale = h_new / h_old
            img_old_resized = cv2.resize(img_old, (int(w_old * scale), h_new))
            viz_img = np.hstack((img_old_resized, img_new))

            info_h, info_bg = 80, np.ones((80, viz_img.shape[1], 3), dtype=np.uint8) * 255
            cv2.putText(info_bg, f"Frame {frame_index} Re-ID | Sim: {score:.4f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            cv2.putText(info_bg, f"Old: {old_node_id} | New: {new_obj_id}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
            final_img = np.vstack((info_bg, viz_img))

            cv2.imwrite(os.path.join(self.reid_viz_dir, f"frame_{frame_index:04d}_reid_{new_obj_id}_to_{old_node_id}.jpg"), final_img)
        except Exception as e: print(f"    [Viz Error] {e}")

    def visualize_spatial_graph(self):
        if not hasattr(self, 'spatial_graph') or self.spatial_graph.number_of_nodes() == 0: return
        plt.figure(figsize=(26, 26))
        pos = nx.spring_layout(self.spatial_graph, weight='weight', k=0.6, seed=42, iterations=100)
        nx.draw_networkx_edges(self.spatial_graph, pos, width=[d.get('weight', 0.5) * 3.0 for u, v, d in self.spatial_graph.edges(data=True)], edge_color='#888888', alpha=0.5, arrows=True, arrowsize=15)
        edge_labels = {(u, v): d.get('relation', '') for u, v, d in self.spatial_graph.edges(data=True) if d.get('relation', '') and d.get('relation') != 'connected'}
        nx.draw_networkx_edge_labels(self.spatial_graph, pos, edge_labels=edge_labels, font_size=7, font_color='darkblue')

        for node_id in self.spatial_graph.nodes():
            if node_id not in pos: continue
            nx.draw_networkx_nodes(self.spatial_graph, pos, nodelist=[node_id], node_color='red', node_size=50)
            plt.text(pos[node_id][0], pos[node_id][1]+0.02, node_id, fontsize=8, ha='center')

        plt.axis('off')
        plt.savefig(os.path.join(self.run_dir, "semantic_topology_web.jpg"), bbox_inches='tight', dpi=150)
        plt.close()

    def _filter_detections_advanced(self, raw_detections: List[Dict], depth_img: np.ndarray,
                                    img_w: int, img_h: int) -> List[Dict]:
        """
        综合过滤逻辑：
        1. 边缘截断过滤 (Border Rejection)
        2. 包含+深度一致性过滤 (Containment + Depth Consistency)
        """
        if not raw_detections:
            return []

        # --- 配置参数 ---
        BORDER_MARGIN = 5         # 边缘像素阈值
        DEPTH_FLAT_THRESH = 0.05   # 深度差阈值 (米)，小于此值视为纹理。建议 3cm-5cm
        CONTAINMENT_RATIO = 0.90   # 包含面积比例阈值
        DEPTH_SCALE = self.intrinsics.get('depth_scale', 1000.0)

        # 1. 预处理：计算面积、中心、并进行边缘过滤
        valid_candidates = []
        for obj in raw_detections:
            x1, y1, x2, y2 = map(int, obj['bounding_box_xyxy'])

            # [Step 1: 边缘截断过滤]
            # 如果物体紧贴边缘，且不是占据全图的大背景（如墙、地），则丢弃
            box_w, box_h = x2 - x1, y2 - y1
            is_touching_edge = (x1 < BORDER_MARGIN) or (y1 < BORDER_MARGIN) or \
                               (x2 > img_w - BORDER_MARGIN) or (y2 > img_h - BORDER_MARGIN)

            # 允许大物体触边 (占画面 85% 以上)
            is_huge = (box_w > img_w * 0.85) and (box_h > img_h * 0.85)

            if is_touching_edge and not is_huge and self.config['filter_edge_detections']:
                continue # 丢弃不完整物体

            # 补充一些辅助信息方便后续计算
            obj['_area'] = box_w * box_h
            obj['_bbox'] = (x1, y1, x2, y2)
            valid_candidates.append(obj)

        if depth_img is None:
            return valid_candidates

        # 2. 包含与深度过滤
        # 按面积从大到小排序，这样我们总是检查“小物体”是否在“大物体”上
        valid_candidates.sort(key=lambda x: x['_area'], reverse=True)

        keep_indices = set(range(len(valid_candidates)))

        for i in range(len(valid_candidates)):
            if i not in keep_indices: continue
            parent = valid_candidates[i]
            px1, py1, px2, py2 = parent['_bbox']

            for j in range(len(valid_candidates)):
                if i == j: continue
                if j not in keep_indices: continue

                child = valid_candidates[j]

                # 优化：如果 Child 比 Parent 还大，肯定不是包含关系 (列表已排序，其实j>i时Child一定<=Parent)
                if child['_area'] > parent['_area']: continue

                cx1, cy1, cx2, cy2 = child['_bbox']

                # [Step 2a: 计算包含关系 (IoU-Min)]
                # 计算 Child 在 Parent 内部的面积
                ix1 = max(px1, cx1); iy1 = max(py1, cy1)
                ix2 = min(px2, cx2); iy2 = min(py2, cy2)
                inter_w = max(0, ix2 - ix1)
                inter_h = max(0, iy2 - iy1)
                inter_area = inter_w * inter_h

                # 如果 Child 90% 以上都在 Parent 里
                if child['_area'] > 0 and (inter_area / child['_area']) > CONTAINMENT_RATIO:

                    # [Step 2b: 深度平坦性测试 (The Flatness Test)]
                    # 关键：对比 Child 的深度 和 Parent 在 Child 附近的深度

                    # 1. 获取 Child 区域深度中位数
                    child_depth_crop = depth_img[cy1:cy2, cx1:cx2]
                    valid_c_d = child_depth_crop[child_depth_crop > 0]
                    if valid_c_d.size == 0: continue # 深度无效，保守起见保留
                    median_child_depth = np.median(valid_c_d)

                    # 2. 获取 Parent 在 Child "周围一圈" 的深度 (Local Background)
                    # 扩张 Child 的框来采样 Parent
                    margin = 15
                    bx1 = max(px1, cx1 - margin); by1 = max(py1, cy1 - margin)
                    bx2 = min(px2, cx2 + margin); by2 = min(py2, cy2 + margin)

                    surround_crop = depth_img[by1:by2, bx1:bx2]

                    # 创建掩码：只取扩张区域，扣掉 Child 自身
                    mask = np.ones(surround_crop.shape, dtype=bool)
                    # 计算 Child 在 surround_crop 中的相对位置
                    rel_x1 = cx1 - bx1; rel_y1 = cy1 - by1
                    rel_x2 = cx2 - bx1; rel_y2 = cy2 - by1
                    # 边界保护
                    rel_x1, rel_y1 = max(0, rel_x1), max(0, rel_y1)
                    rel_x2, rel_y2 = min(surround_crop.shape[1], rel_x2), min(surround_crop.shape[0], rel_y2)

                    mask[rel_y1:rel_y2, rel_x1:rel_x2] = False # 挖空中心

                    valid_surround_d = surround_crop[mask & (surround_crop > 0)]

                    if valid_surround_d.size == 0:
                        continue # 无法获取周围深度，保守保留

                    median_surround_depth = np.median(valid_surround_d)

                    # 3. 计算深度差
                    diff_m = abs(median_child_depth - median_surround_depth) / DEPTH_SCALE

                    # [判定]
                    if diff_m < DEPTH_FLAT_THRESH:
                        # 深度差 < 5cm -> 认为是平面纹理 -> 删除 Child
                        # 比如：毛巾上的花，深度和毛巾几乎一样
                        print(f"  [Filter] Removed '{child['label']}' inside '{parent['label']}' (Depth Diff: {diff_m:.3f}m < {DEPTH_FLAT_THRESH}m)")
                        keep_indices.remove(j)
                    else:
                        # 深度差 > 5cm -> 认为是立体物体 -> 保留 Child
                        # 比如：桌子上的杯子，杯子比桌面近
                        # print(f"  [Keep] Kept '{child['label']}' on '{parent['label']}' (Depth Diff: {diff_m:.3f}m)")
                        pass

        # 重组结果
        final_results = []
        for k in sorted(list(keep_indices)):
            # 清理临时 key
            obj = valid_candidates[k]
            if '_area' in obj: del obj['_area']
            if '_bbox' in obj: del obj['_bbox']
            final_results.append(obj)

        return final_results

    # [新增] 辅助函数：计算视图评分 (越小越好，表示越靠近中心)
    def _calculate_view_score(self, bbox, img_w, img_h):
        if img_w <= 0 or img_h <= 0: return 1.0
        # 计算 BBox 中心
        bx = (bbox[0] + bbox[2]) / 2.0
        by = (bbox[1] + bbox[3]) / 2.0
        # 计算画面中心
        cx, cy = img_w / 2.0, img_h / 2.0

        # 归一化欧氏距离 (0.0 = 正中心, 1.0 = 角落)
        max_dist = np.sqrt(cx**2 + cy**2)
        curr_dist = np.sqrt((bx - cx)**2 + (by - cy)**2)

        return curr_dist / max_dist

    def _decode_rle(self, rle):
        """将 RLE 字典 ({'counts': [], 'size': []}) 解码为二进制 Mask (numpy array)"""
        h, w = rle['size']
        counts = rle['counts']

        # 展平的 mask
        mask_flat = np.zeros(h * w, dtype=np.uint8)

        current_idx = 0
        # 根据 COCO 标准，RLE counts 通常从 0 (背景) 开始交替
        current_val = 0

        for count in counts:
            mask_flat[current_idx : current_idx + count] = current_val
            current_idx += count
            current_val = 1 - current_val # 0 变 1, 1 变 0

        return mask_flat.reshape((h, w)) > 0 # 返回 bool 类型

    def _mask_depth_aabb(self, depth_img, mask, pose):
        """Estimate a world-space AABB from native RGB-D mask evidence.

        This is prediction geometry, not evaluator alignment: the bounds are
        computed only from the current instance mask, depth image, camera
        intrinsics, and pose. Per-observation percentile trimming limits depth
        boundary noise before persistent observations are merged.
        """
        if depth_img is None or mask is None or pose is None or mask.shape != depth_img.shape:
            return None
        valid = mask & (depth_img > 0)
        max_depth_mm = float(self.config.get('depth_threshold_mm', 5000))
        depth_scale = float(self.intrinsics.get('depth_scale', 1000.0))
        if max_depth_mm > 0:
            valid &= depth_img.astype(np.float64) / depth_scale <= max_depth_mm / 1000.0
        ys, xs = np.where(valid)
        if len(xs) < 20:
            return None
        # Bound cost for large wall/floor masks without changing their extent distribution.
        sample_step = max(1, len(xs) // 20000)
        xs, ys = xs[::sample_step], ys[::sample_step]
        z = depth_img[ys, xs].astype(np.float64) / depth_scale
        x = (xs - float(self.intrinsics['cx'])) * z / float(self.intrinsics['fx'])
        y = (ys - float(self.intrinsics['cy'])) * z / float(self.intrinsics['fy'])
        camera_cv = np.stack([x, y, z, np.ones_like(z)], axis=1)
        cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
        world = (np.asarray(pose, dtype=np.float64) @ cv_to_gl @ camera_cv.T).T[:, :3]
        finite = world[np.all(np.isfinite(world), axis=1)]
        if len(finite) < 20:
            return None
        lower_q, upper_q = self.config.get('vigil3d_aabb_percentiles', [2.0, 98.0])
        minimum = np.percentile(finite, float(lower_q), axis=0)
        maximum = np.percentile(finite, float(upper_q), axis=0)
        if np.any(maximum <= minimum):
            return None
        return [minimum.tolist(), maximum.tolist()]

    def _process_frame_results(self, frame_index: int, segmentation_json_path: str, current_rgb_path: str, current_depth_path: str, current_pose: np.ndarray):
        # 1. 加载数据
        with open(segmentation_json_path, 'r') as f:
            raw_results = json.load(f)

        depth_img = cv2.imread(current_depth_path, cv2.IMREAD_UNCHANGED)

        # 获取图像尺寸 (用于边缘检测)
        # 假设 RGB 和 Depth 尺寸一致
        if depth_img is not None:
            h_img, w_img = depth_img.shape[:2]
        else:
            # 如果没有深度图，尝试读取 RGB 获取尺寸
            temp_img = Image.open(current_rgb_path)
            w_img, h_img = temp_img.size

        # ==========================================
        # [Step 0] 执行高级过滤 (IoU + Depth + Border)
        # ==========================================
        print(f"  [Filter] Raw objects: {len(raw_results)}", end="")

        filtered_results = self._filter_detections_advanced(raw_results, depth_img, w_img, h_img)

        print(f" -> Filtered: {len(filtered_results)}")
        # ==========================================
        seg_dir = os.path.dirname(segmentation_json_path)
        depth_img = cv2.imread(current_depth_path, cv2.IMREAD_UNCHANGED)

        re_id_config = self.config.get('re_identification', {})
        # 1. 预处理当前帧物体 (保持不变)
        feat_start = time.time()
        current_objects_data = []
        for obj in filtered_results:
            obj_id = obj['object_id']
            try:
                crop_path = os.path.join(seg_dir, obj['cropped_image_path'])
                feat = extract_feature(Image.open(crop_path).convert("RGB"), self.dinov3_model, self.device)

                global_pos_3d = None
                observation_aabb_3d = None

                # [修改开始] =======================================================
                if depth_img is not None and current_pose is not None:
                    x1, y1, x2, y2 = map(int, obj['bounding_box_xyxy'])
                    h, w = depth_img.shape[:2]

                    # 边界安全检查
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    depth_crop = depth_img[y1:y2, x1:x2]

                    # 准备 Mask 数据
                    mask_crop = None
                    if 'segmentation' in obj and obj['segmentation']:
                        try:
                            # 1. 解码完整 Mask
                            full_mask = self._decode_rle(obj['segmentation'])
                            # 2. 裁剪到 bbox 区域
                            mask_crop = full_mask[y1:y2, x1:x2]
                        except Exception as e:
                            print(f"Warning: Mask decode failed for {obj_id}: {e}")
                            mask_crop = None

                    valid_depths = np.array([])
                    cx, cy = (x1+x2)//2, (y1+y2)//2 # 默认使用 Box 中心

                    if depth_crop.size > 0:
                        if mask_crop is not None and mask_crop.shape == depth_crop.shape:
                            # === 方案 A: 使用 Mask 过滤 ===
                            # 只取 Mask 范围内的深度像素
                            valid_depths = depth_crop[(depth_crop > 0) & mask_crop]

                            # [进阶] 如果有 Mask，计算 Mask 的质心作为 cx, cy，而不是 Box 中心
                            if valid_depths.size > 0:
                                ys, xs = np.where(mask_crop)
                                if len(ys) > 0:
                                    cy = y1 + int(np.mean(ys))
                                    cx = x1 + int(np.mean(xs))
                        else:
                            # === 方案 B: 没有 Mask，回退到 Box ===
                            valid_depths = depth_crop[depth_crop > 0]

                        # 计算最终 3D 坐标
                        if valid_depths.size > 0:
                            # 过滤掉极端的深度值 (可选，取中值更稳)
                            median_d = np.median(valid_depths) / self.intrinsics['depth_scale']

                            # 简单的深度范围过滤 (比如太远或太近可能是噪声)
                            # if 0.1 < median_d < 10.0:
                            global_pos_3d = pixel_to_world(cx, cy, median_d, current_pose, self.intrinsics)
                            if mask_crop is not None and mask_crop.shape == depth_crop.shape:
                                full_valid_mask = np.zeros(depth_img.shape, dtype=bool)
                                full_valid_mask[y1:y2, x1:x2] = mask_crop
                                observation_aabb_3d = self._mask_depth_aabb(
                                    depth_img, full_valid_mask, current_pose
                                )
                # [修改结束] =======================================================

                obj_floor_id = self.floor_tracker.current_floor_id
                if global_pos_3d:
                    obj_floor_id = self.floor_tracker.get_floor_id_by_object_y(global_pos_3d[1])

                current_objects_data.append({
                    "raw_obj": obj,
                    "feature": feat,
                    "pos_3d": global_pos_3d,
                    "aabb_3d": observation_aabb_3d,
                    "floor_id": obj_floor_id,
                    "label": obj.get('label', 'object').lower().strip(),
                    "match_id": None,
                    "match_kb_idx": -1
                })
            except Exception as e:
                print(f"Error processing object {obj_id}: {e}")

        feat_duration = time.time() - feat_start
        print(f"  [Feature] Extracted features for {len(current_objects_data)} objects in {feat_duration:.4f}s")
        # 2. 准备知识库 (保持不变)
        kb_candidates = []
        kb_indices_map = []
        kb_labels = []
        for idx, item in enumerate(self.memory_kb):
            # if item.get('floor_id') == self.floor_tracker.current_floor_id:
            kb_candidates.append(item)
            kb_indices_map.append(idx)
            kb_labels.append(item.get('label', 'object').split(' ')[0].lower())

        kb_features = np.array([item['dinov3_feature'] for item in kb_candidates]) if kb_candidates else np.empty((0, 0))
        used_kb_local_indices = set()
        unmatched_curr_indices = list(range(len(current_objects_data)))
        GENERIC_LABELS = {'object', 'item', 'thing', 'stuff', 'unknown', 'entity'}

        # =================================================
        # [修改后] Stage 1: 视觉强匹配 (基于全局排序，防止节点被抢占)
        # =================================================
        VISUAL_SIM_THRESH = re_id_config.get('visual_similarity_threshold', 0.8)
        VISUAL_DIST_THRESH = re_id_config.get('spatial_distance_threshold', 1.5)
        FORCE_MERGE_DIST = re_id_config.get('force_merge_distance_threshold', 1.0)

        # 1. 构建所有可能的匹配对 (Potential Matches)
        potential_matches = [] # 格式: (score, curr_idx, kb_local_idx)

        reid_start = time.time()

        if len(kb_candidates) > 0 and len(unmatched_curr_indices) > 0:
            # 批量计算相似度矩阵 [N_curr, N_kb]
            curr_feats_list = [current_objects_data[i]['feature'] for i in unmatched_curr_indices]
            curr_feats_matrix = np.stack(curr_feats_list)

            # cosine_similarity 返回 (N_curr, N_kb)
            sim_matrix = cosine_similarity(curr_feats_matrix, kb_features)

            for i, real_curr_idx in enumerate(unmatched_curr_indices):
                curr_obj = current_objects_data[real_curr_idx]
                if curr_obj['pos_3d'] is None: continue

                curr_pos = np.array(curr_obj['pos_3d'])

                for k_local_idx, kb_item in enumerate(kb_candidates):
                    # 获取相似度分数
                    score = sim_matrix[i][k_local_idx]

                    # 基础阈值过滤
                    if score > VISUAL_SIM_THRESH:
                        k_pos = kb_item.get('global_pos_3d')
                        if k_pos:
                            dist = np.linalg.norm(curr_pos - np.array(k_pos))
                            # 距离过滤
                            if dist < VISUAL_DIST_THRESH:
                                # 加入候选队列
                                potential_matches.append({
                                    'score': score,
                                    'dist': dist,
                                    'curr_idx': real_curr_idx,
                                    'kb_idx': k_local_idx,
                                    'kb_label': kb_item.get('label', ''),
                                    'curr_label': curr_obj['label']
                                })

        # 2. 全局排序：优先匹配分数高的，分数相同时优先距离近的
        # sort by score (desc), then distance (asc)
        potential_matches.sort(key=lambda x: (-x['score'], x['dist']))

        # 3. 执行匹配 (确保一对一)
        matched_curr_set = set()

        for match in potential_matches:
            c_idx = match['curr_idx']
            k_idx = match['kb_idx']

            # 检查是否已经被占用
            if c_idx in matched_curr_set or k_idx in used_kb_local_indices:
                continue
            # [修改后的代码] -----------------------------------------------------------
            kb_node_id = kb_candidates[k_idx]['node_id']
            kb_pos = kb_candidates[k_idx]['global_pos_3d']
            curr_pos = current_objects_data[c_idx]['pos_3d']

            # 获取原始 object_id
            raw_obj_id = current_objects_data[c_idx]['raw_obj']['object_id']

            # 格式化坐标以便阅读 (保留2位小数)
            k_pos_str = f"[{kb_pos[0]:.1f}, {kb_pos[1]:.1f}, {kb_pos[2]:.1f}]" if kb_pos else "None"
            c_pos_str = f"[{curr_pos[0]:.1f}, {curr_pos[1]:.1f}, {curr_pos[2]:.1f}]" if curr_pos else "None"

            print(f"  [Stage 1] MATCH: {raw_obj_id} ('{match['curr_label']}') -> Node '{kb_node_id}'")
            print(f"            Sim: {match['score']:.3f} | Dist: {match['dist']:.2f}m")
            print(f"            Pos: {c_pos_str} -> {k_pos_str}")
            # -------------------------------------------------------------------------

            current_objects_data[c_idx]['match_kb_idx'] = kb_indices_map[k_idx]
            current_objects_data[c_idx]['match_id'] = kb_candidates[k_idx]['node_id']

            # 标记占用
            used_kb_local_indices.add(k_idx)
            matched_curr_set.add(c_idx)
            if c_idx in unmatched_curr_indices:
                unmatched_curr_indices.remove(c_idx)

        reid1_duration = time.time() - reid_start
        print(f"  [Re-ID Stage 1] Completed in {reid1_duration:.4f}s. Matches found: {len(matched_curr_set)}")
        # =================================================
        # Stage 2: 空间 + 语义/泛化特征 混合排序
        # =================================================


        SEMANTIC_SIM_THRESH = re_id_config.get('semantic_similarity_threshold', 0.6)
        GENERIC_VISUAL_THRESH = re_id_config.get('generic_object_visual_threshold', 0.65)

        if len(kb_candidates) > 0 and len(unmatched_curr_indices) > 0:

            kb_label_embeddings = None
            if len(unmatched_curr_indices) > 0:
                 kb_label_embeddings = self.text_encoder.encode(kb_labels, convert_to_tensor=True, show_progress_bar=False)

            for curr_idx in list(unmatched_curr_indices):
                curr_obj = current_objects_data[curr_idx]
                if curr_obj['pos_3d'] is None: continue

                # 获取原始 object_id
                raw_obj_id = curr_obj['raw_obj']['object_id']

                curr_label = curr_obj['label']
                base_label = ''.join([i for i in curr_label if not i.isdigit()]).strip('_').strip()
                is_generic = base_label in GENERIC_LABELS

                candidates_in_range = []

                # 1. 筛选距离
                for k_idx, kb_item in enumerate(kb_candidates):
                    if k_idx in used_kb_local_indices: continue

                    k_pos = kb_item.get('global_pos_3d')
                    if k_pos:
                        dist = np.linalg.norm(np.array(curr_obj['pos_3d']) - np.array(k_pos))
                        if dist < FORCE_MERGE_DIST:
                            candidates_in_range.append({
                                'idx': k_idx,
                                'dist': dist,
                                'kb_label': kb_labels[k_idx]
                            })

                if not candidates_in_range:
                    print(f"  [Stage 2] Fail: {raw_obj_id} ('{curr_label}') no neighbor within {FORCE_MERGE_DIST}m")
                    continue

                # Spatial gating is already applied above. All second-stage
                # matches additionally require visual evidence, including specific labels.
                compatible = []
                if is_generic:
                    compatible = [c['idx'] for c in candidates_in_range]
                else:
                    text = self.text_encoder.encode(curr_label, convert_to_tensor=True, show_progress_bar=False)
                    for candidate in candidates_in_range:
                        k = candidate['idx']
                        exact = curr_label in candidate['kb_label'] or candidate['kb_label'] in curr_label
                        semantic = float(util.cos_sim(text, kb_label_embeddings[k])[0, 0])
                        if exact or semantic > SEMANTIC_SIM_THRESH:
                            compatible.append(k)
                threshold = float(re_id_config.get('second_stage_visual_threshold', GENERIC_VISUAL_THRESH))
                selected = choose_visual_match(curr_obj['feature'], compatible, kb_features, threshold)
                best_match_idx = -1 if selected is None else selected
                match_reason = "Spatial + semantic + visual match"

                # 3. 执行匹配
                if best_match_idx != -1:
                    # [修改后的代码] -----------------------------------------------------------
                    target_node = kb_candidates[best_match_idx]
                    target_id = target_node['node_id']
                    target_pos = target_node.get('global_pos_3d')
                    curr_pos = curr_obj['pos_3d']

                    t_pos_str = f"[{target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f}]" if target_pos else "None"
                    c_pos_str = f"[{curr_pos[0]:.1f}, {curr_pos[1]:.1f}, {curr_pos[2]:.1f}]" if curr_pos else "None"

                    print(f"  [Stage 2] MATCH ({match_reason}): {raw_obj_id} ('{curr_label}') -> Node '{target_id}'")
                    print(f"            Pos: {c_pos_str} -> {t_pos_str}")
                    # -------------------------------------------------------------------------

                    current_objects_data[curr_idx]['match_kb_idx'] = kb_indices_map[best_match_idx]
                    current_objects_data[curr_idx]['match_id'] = kb_candidates[best_match_idx]['node_id']
                    used_kb_local_indices.add(best_match_idx)
                    unmatched_curr_indices.remove(curr_idx)

        reid2_duration = time.time() - reid1_duration - reid_start
        print(f"  [Re-ID Stage 2] Completed in {reid2_duration:.4f}s. New nodes created: {len(unmatched_curr_indices)}")

        # 3. 结果应用
        current_frame_nodes = []
        for idx, curr_obj in enumerate(current_objects_data):
            clean_label = "".join([c if c.isalnum() else "_" for c in curr_obj['label']]).strip('_').lower()
            raw_obj_id = curr_obj['raw_obj']['object_id']

            # 计算当前观测的评分
            current_view_score = self._calculate_view_score(curr_obj['raw_obj']['bounding_box_xyxy'], w_img, h_img)

            if curr_obj['match_kb_idx'] != -1:

                merge_start = time.time()

                # --- MERGE 逻辑修改开始 ---
                real_kb_idx = curr_obj['match_kb_idx']
                target_node = self.memory_kb[real_kb_idx]
                final_node_id = target_node['node_id']

                # 1. 更新 3D 位置 (加权平均)
                if curr_obj['pos_3d']:
                    old_p = target_node['global_pos_3d']
                    if old_p:
                        self.memory_kb[real_kb_idx]['global_pos_3d'] = [(0.7*o + 0.3*n) for o, n in zip(old_p, curr_obj['pos_3d'])]
                    else:
                        self.memory_kb[real_kb_idx]['global_pos_3d'] = curr_obj['pos_3d']

                # Merge robust per-observation RGB-D bounds without GT geometry.
                if curr_obj.get('aabb_3d'):
                    new_min = np.asarray(curr_obj['aabb_3d'][0], dtype=float)
                    new_max = np.asarray(curr_obj['aabb_3d'][1], dtype=float)
                    old_box = target_node.get('aabb_3d')
                    if old_box:
                        new_min = np.minimum(new_min, np.asarray(old_box[0], dtype=float))
                        new_max = np.maximum(new_max, np.asarray(old_box[1], dtype=float))
                    self.memory_kb[real_kb_idx]['aabb_3d'] = [new_min.tolist(), new_max.tolist()]
                    self.memory_kb[real_kb_idx]['aabb_observations'] = int(
                        target_node.get('aabb_observations', 0)
                    ) + 1

                # 2. 更新特征 (加权平均)
                old_f = np.array(target_node['dinov3_feature'])
                self.memory_kb[real_kb_idx]['dinov3_feature'] = (0.8 * curr_obj['feature'] + 0.2 * old_f).tolist()

                # 3. 更新 Label
                base_label = ''.join([i for i in clean_label if not i.isdigit()]).strip('_')
                if base_label not in GENERIC_LABELS and clean_label not in target_node['label']:
                    self.memory_kb[real_kb_idx]['label'] += f" {clean_label}"

                # 4. [核心修改] 最佳视图更新策略
                # 获取旧的评分，如果没有则默认为 1.0 (最差)
                old_view_score = target_node.get('view_score', 1.0)

                # 如果当前物体更靠近图像中心 (score 越小越好)
                if current_view_score < old_view_score:
                    # print(f"  [Update View] Node {final_node_id} updated with better view (Score: {old_view_score:.2f} -> {current_view_score:.2f})")
                    self.memory_kb[real_kb_idx]['bbox'] = curr_obj['raw_obj']['bounding_box_xyxy']
                    self.memory_kb[real_kb_idx]['full_image_path'] = current_rgb_path
                    self.memory_kb[real_kb_idx]['cropped_image_path'] = curr_obj['raw_obj']['cropped_image_path'] # 同时也更新 Crop 路径
                    self.memory_kb[real_kb_idx]['view_score'] = current_view_score
                    # 注意：node_id 不变，feature 已融合
                # --- MERGE 逻辑修改结束 ---
                print(f"merge time :{time.time()-merge_start:.4f}s")
            else:
                # NEW
                new_start = time.time()

                target_room = self.current_room_id if (curr_obj['floor_id'] == self.floor_tracker.current_floor_id and self.current_room_id != 0) else None
                final_node_id = self._create_leaf_node(
                    curr_obj['raw_obj'], curr_obj['feature'], frame_index, seg_dir,
                    current_rgb_path, clean_label, curr_obj['pos_3d']
                )
                self.memory_kb[-1]['floor_id'] = curr_obj['floor_id']
                self.memory_kb[-1]['parent_room'] = target_room
                self.memory_kb[-1]['aabb_3d'] = curr_obj.get('aabb_3d')
                self.memory_kb[-1]['aabb_observations'] = int(curr_obj.get('aabb_3d') is not None)
                # [新增] 记录初始评分
                self.memory_kb[-1]['view_score'] = current_view_score

                # 回填 ID 用于调试可视化
                current_objects_data[idx]['match_id'] = final_node_id

                curr_pos = curr_obj['pos_3d']
                c_pos_str = f"[{curr_pos[0]:.1f}, {curr_pos[1]:.1f}, {curr_pos[2]:.1f}]" if curr_pos else "None"

                # 输出 New Node 日志
                print(f"  [Action] NEW NODE: {raw_obj_id} ('{clean_label}') -> ID '{final_node_id}'")
                print(f"           At Pos: {c_pos_str}")

                print(f"           Created in {time.time() - new_start:.4f}s")

            if final_node_id:
                current_frame_nodes.append({"node_id": final_node_id, "bbox": curr_obj['raw_obj']['bounding_box_xyxy']})
                if not self.spatial_graph.has_node(final_node_id):
                    self.spatial_graph.add_node(final_node_id)

        # =========================================================
        # [Step 4] 拓扑分析 (3D Clustering)
        # =========================================================

        valid_nodes_for_spatial = []
        node_positions_2d = {} # 用于聚类
        node_positions_3d = {} # [新增] 用于几何空间推理

        for node_info in current_frame_nodes:
            nid = node_info['node_id']
            kb_item = next((item for item in self.memory_kb if item['node_id'] == nid), None)
            if kb_item and kb_item.get('global_pos_3d'):
                if kb_item.get('floor_id') == self.floor_tracker.current_floor_id:
                    pos = kb_item['global_pos_3d']
                    node_positions_2d[nid] = np.array([pos[0], pos[2]])
                    node_positions_3d[nid] = pos # [新增] 保存完整3D坐标
                    valid_nodes_for_spatial.append(node_info)

        if len(valid_nodes_for_spatial) > 1:
            print(f"  [Topology] Clustering {len(valid_nodes_for_spatial)} nodes (Current Floor)...", end='', flush=True)

            # ... [聚类代码保持不变] ...
            cluster_graph = nx.Graph()
            cluster_graph.add_nodes_from([n['node_id'] for n in valid_nodes_for_spatial])
            node_ids = list(node_positions_2d.keys())
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    id_i, id_j = node_ids[i], node_ids[j]
                    dist = np.linalg.norm(node_positions_2d[id_i] - node_positions_2d[id_j])
                    if dist <= self.spatial_cluster_thresh:
                        cluster_graph.add_edge(id_i, id_j)
            clusters = [list(c) for c in nx.connected_components(cluster_graph)]
            print(f" Found {len(clusters)} clusters.", flush=True)

            # [修改] 读取配置决定使用哪种方法
            spatial_method = self.config.get('spatial_reasoning', {}).get('method', 'vlm') # 默认为 vlm

            # 如果是 vlm 模式，才需要画图
            annotated_img_path = current_rgb_path
            if spatial_method == 'vlm':
                 annotated_img_path = self._annotate_image_with_clusters(
                    current_rgb_path, current_frame_nodes, clusters, frame_index
                )




# ... (前面的代码) ...

# LLM / Geometric 分析
            topo_start = time.time()
            for cluster_nodes in clusters:
                if len(cluster_nodes) < 2: continue

                print(f"    -> Analyzing cluster ({spatial_method}): {cluster_nodes}...", end='', flush=True)
                existing_relations = []
                subgraph = self.spatial_graph.subgraph(cluster_nodes)
                for u, v, data in subgraph.edges(data=True):
                    existing_relations.append(f"{u}->{v}({data.get('relation', 'connected')})")

                if hasattr(self, 'llm'):
                    # [Timer Start] 记录推理开始时间
                    edge_gen_start = time.time()

                    # [修改] 调用 detect_spatial_relationships
                    spatial_quads = self.llm.detect_spatial_relationships(
                        image_path=annotated_img_path,
                        object_list=cluster_nodes,
                        existing_relations=existing_relations,
                        object_positions=node_positions_3d,
                        method=spatial_method
                    )

                    # [Timer End] 记录推理结束时间
                    edge_gen_cost = time.time() - edge_gen_start

                    # 打印该簇的推理总耗时
                    print(f" [LLM/Algo Time: {edge_gen_cost:.4f}s] ", end='', flush=True)

                    for i, item in enumerate(spatial_quads):
                        # ... [解析返回结果的代码保持不变] ...
                        u, v, rel, w = None, None, None, 0.0
                        if isinstance(item, dict):
                            u, v, rel, w = item.get('Source'), item.get('Target'), item.get('Relation'), item.get('Weight')
                            if not u: u, v, rel, w = item.get('source'), item.get('target'), item.get('relation'), item.get('weight')
                        elif isinstance(item, list) and len(item) >= 3:
                            u, v, rel = item[0], item[1], item[2]
                            w = item[3] if len(item) > 3 else 0.5

                        if u in cluster_nodes and v in cluster_nodes:
                            if self.spatial_graph.has_node(u) and self.spatial_graph.has_node(v):
                                try:
                                    weight_val = float(w)
                                    threshold = 0.6 if spatial_method == 'vlm' else 0.4
                                    if weight_val > threshold:
                                        self.spatial_graph.add_edge(u, v, relation=rel, weight=weight_val)

                                        # [Output] 输出具体的边和它被处理的时间点（或分摊时间）
                                        # 这里的 time.time() 是边被添加到图中的时刻
                                        print(f"\n      + Edge [{u}->{v} ({rel})] added. (Batch Latency: {edge_gen_cost:.4f}s)", end='')

                                except (ValueError, TypeError): pass
                print(" Done.")

            if "_temp_spatial" in annotated_img_path and os.path.exists(annotated_img_path):
                os.remove(annotated_img_path)

            # 这里的 topo_start 是整个拓扑分析的开始，计算总耗时
            total_topo_time = time.time() - topo_start
            print(f"  [Topology] Analysis finished ({total_topo_time:.2f}s).", flush=True)

            # # LLM / Geometric 分析
            # topo_start = time.time()
            # for cluster_nodes in clusters:
            #     if len(cluster_nodes) < 2: continue

            #     print(f"    -> Analyzing cluster ({spatial_method}): {cluster_nodes}...", end='', flush=True)
            #     existing_relations = []
            #     subgraph = self.spatial_graph.subgraph(cluster_nodes)
            #     for u, v, data in subgraph.edges(data=True):
            #         existing_relations.append(f"{u}->{v}({data.get('relation', 'connected')})")

            #     if hasattr(self, 'llm'):
            #         # [修改] 调用 detect_spatial_relationships，传入 method 和 positions
            #         spatial_quads = self.llm.detect_spatial_relationships(
            #             image_path=annotated_img_path,
            #             object_list=cluster_nodes,
            #             existing_relations=existing_relations,
            #             object_positions=node_positions_3d, # 传入3D坐标字典
            #             method=spatial_method
            #         )

            #         for item in spatial_quads:
            #             # ... [解析返回结果的代码保持不变] ...
            #             u, v, rel, w = None, None, None, 0.0
            #             if isinstance(item, dict):
            #                 u, v, rel, w = item.get('Source'), item.get('Target'), item.get('Relation'), item.get('Weight')
            #                 if not u: u, v, rel, w = item.get('source'), item.get('target'), item.get('relation'), item.get('weight')
            #             elif isinstance(item, list) and len(item) >= 3:
            #                 u, v, rel = item[0], item[1], item[2]
            #                 w = item[3] if len(item) > 3 else 0.5

            #             if u in cluster_nodes and v in cluster_nodes:
            #                 if self.spatial_graph.has_node(u) and self.spatial_graph.has_node(v):
            #                     try:
            #                         weight_val = float(w)
            #                         # 几何模式的权重通常由距离决定，可能不需要这么高的阈值，或者在内部已经处理好了
            #                         threshold = 0.6 if spatial_method == 'vlm' else 0.4
            #                         if weight_val > threshold:
            #                             self.spatial_graph.add_edge(u, v, relation=rel, weight=weight_val)
            #                     except (ValueError, TypeError): pass
            #     print(" Done.")

            # if "_temp_spatial" in annotated_img_path and os.path.exists(annotated_img_path):
            #     os.remove(annotated_img_path)
            # print(f"  [Topology] Analysis finished ({time.time() - topo_start:.2f}s).", flush=True)

        # --- [新增] 绘制并保存俯瞰图 ---
        self._update_and_save_bev_map(frame_index, current_pose)
        # ================= DEBUG VISUALIZATION =================

        # 读取当前帧原图
        debug_img = cv2.imread(current_rgb_path)
        if debug_img is not None:
            for idx, obj_data in enumerate(current_objects_data):
                # 获取 BBox
                x1, y1, x2, y2 = map(int, obj_data['raw_obj']['bounding_box_xyxy'])

                # [修改点]：判断 match_kb_idx 是否为 -1 来区分新旧
                if obj_data['match_kb_idx'] != -1:
                    # 匹配到了老节点：绿色
                    color = (0, 255, 0)
                    text = f"{obj_data['match_id']} (Old)"
                else:
                    # 新建节点：红色
                    color = (0, 0, 255)
                    # 此时 match_id 已经被你回填了新ID，所以显示出来正好
                    text = f"{obj_data['match_id']} (New)"

                # 画框
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                # 画文字
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(debug_img, (x1, y1-15), (x1+tw, y1), color, -1)
                cv2.putText(debug_img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 保存图片到 debug 文件夹
            debug_dir = os.path.join(os.path.dirname(segmentation_json_path), "debug_merge")
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"debug_{frame_index:05d}.jpg"), debug_img)
        # =======================================================


    def _prepare_caption_input(
        self, mask_crop_path: str, full_image_path: str, bbox: List[int],
        node_id: str, frame_index: int,
    ) -> str:
        settings = self.config.get('semantic_caption', {})
        mode = settings.get('input_mode', 'mask')
        if mode == 'mask':
            return mask_crop_path
        image = cv2.imread(full_image_path, cv2.IMREAD_COLOR)
        if image is None:
            return mask_crop_path
        x1, y1, x2, y2 = map(int, bbox)
        if mode == 'bbox':
            caption_image = image[y1:y2, x1:x2]
        elif mode == 'context':
            scale = max(1.0, float(settings.get('context_scale', 2.0)))
            width, height = x2 - x1, y2 - y1
            extra_x = int(width * (scale - 1.0) / 2.0)
            extra_y = int(height * (scale - 1.0) / 2.0)
            cx1, cy1 = max(0, x1 - extra_x), max(0, y1 - extra_y)
            cx2, cy2 = min(image.shape[1], x2 + extra_x), min(image.shape[0], y2 + extra_y)
            caption_image = image[cy1:cy2, cx1:cx2].copy()
            cv2.rectangle(caption_image, (x1 - cx1, y1 - cy1), (x2 - cx1 - 1, y2 - cy1 - 1), (0, 0, 255), 4)
        else:
            raise ValueError(f"Unknown semantic_caption.input_mode: {mode}")
        output_dir = os.path.join(self.run_dir, 'caption_inputs')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{node_id}_{frame_index:06d}_{mode}.jpg")
        cv2.imwrite(output_path, caption_image)
        return output_path

    def _create_leaf_node(self, obj_data: Dict, feature: np.ndarray, frame_index: int, seg_dir: str, current_rgb_path: str, clean_label: str, global_pos_3d: List[float] = None) -> str:
        """[修改] 增加了 global_pos_3d 参数"""
        count = self.label_counters.get(clean_label, 0)
        node_id = f"{clean_label}_{count}"
        self.label_counters[clean_label] = count + 1

        crop_path = os.path.join(seg_dir, obj_data['cropped_image_path'])
        bbox = obj_data['bounding_box_xyxy']
        caption_path = self._prepare_caption_input(
            crop_path, current_rgb_path, bbox, node_id, frame_index
        )

        llm_desc = "Auto-generated description unavailable"
        if hasattr(self, 'llm'):
            llm_desc = self.llm.generate_leaf_node_semantics(caption_path, clean_label, f"bbox: {bbox}, frame: {frame_index}")

        leaf_node = {
            "node_id": node_id,
            "node_level": 3,
            "semantic_tag": clean_label,
            "llm_desc": llm_desc,
            "bbox": bbox,
            "global_pos_3d": global_pos_3d, # Save 3D pos
            "timestamp": datetime.datetime.now().isoformat()
        }
        with open(self.nodes_path, 'a', encoding='utf-8') as f: f.write(json.dumps(leaf_node, ensure_ascii=False) + "\n")

        self.memory_kb.append({
            "node_id": node_id,
            "dinov3_feature": feature.tolist(),
            "label": clean_label,
            "llm_desc": llm_desc,
            "full_image_path": current_rgb_path,
            "bbox": bbox,
            "global_pos_3d": global_pos_3d # Save 3D pos in Memory
        })
        return node_id

    # [修改] 绘制 BEV 地图：包含 Mask 和 历史更新点
    # -----------------------------------------------------------
    # [修改] 核心绘图函数：解决坐标、物体显示、比例尺问题
    def _update_and_save_bev_map(self, frame_index: int, current_pose: np.ndarray, return_path_only=False) -> str:
        if current_pose is None: return None

        curr_pos = current_pose[:3, 3]
        curr_floor = self.floor_tracker.current_floor_id

        # 记录轨迹
        if curr_floor not in self.floor_trajectories:
            self.floor_trajectories[curr_floor] = []
        self.floor_trajectories[curr_floor].append(curr_pos)

        # 准备画布
        floor_map_dir = os.path.join(self.map_output_dir, f"floor_{curr_floor:02d}")
        os.makedirs(floor_map_dir, exist_ok=True)

        fig = plt.figure(figsize=(14, 14))
        ax = plt.gca()

        # --- 坐标转换函数 ---
        def to_plot_coords(x, z): return x, -z

        # --- 1. 绘制 Room Mask 底图 ---
        room_map = self.map_manager.room_id_map
        if room_map is not None:
            min_x, min_z = self.map_manager.min_x, self.map_manager.min_z
            res = self.map_manager.resolution
            h, w = room_map.shape
            max_x = min_x + w * res
            max_z = min_z + h * res

            extent = [min_x, max_x, -max_z, -min_z]
            masked_data = np.ma.masked_where(room_map == 0, room_map)
            cmap = plt.cm.get_cmap('tab20b', np.max(room_map) + 1) if np.max(room_map) > 0 else plt.cm.gray
            plt.imshow(masked_data, extent=extent, cmap=cmap, alpha=0.8, origin='upper', interpolation='nearest')

        # --- 2. 绘制物体 & 关系 ---
        plotted_objects = {}
        current_floor_objs = [item for item in self.memory_kb if item.get('floor_id') == curr_floor]

        cx, cz = to_plot_coords(curr_pos[0], curr_pos[2]) # 当前机器人绘图坐标

        for item in current_floor_objs:
            pos = item.get('global_pos_3d')
            if pos:
                px, pz = to_plot_coords(pos[0], pos[2])
                plotted_objects[item['node_id']] = (px, pz)
                ax.scatter(px, pz, c='grey', s=30, alpha=0.9, edgecolors='white', linewidth=0.8, zorder=3)

        if hasattr(self, 'spatial_graph'):
            for u, v in self.spatial_graph.edges():
                if u in plotted_objects and v in plotted_objects:
                    u_pos = plotted_objects[u]
                    v_pos = plotted_objects[v]
                    plt.plot([u_pos[0], v_pos[0]], [u_pos[1], v_pos[1]], color='green', linewidth=1.5, alpha=0.5, zorder=4)

        # --- 3. 绘制历史更新点 (Blue Wedges) - 带衰老效果 ---
        if hasattr(self, 'global_update_history'):
            FADE_OUT_FRAMES = 500  # 经过500帧后完全消失
            MIN_ALPHA = 0.05       # 最小透明度

            for record in self.global_update_history:
                if record['floor'] != curr_floor:
                    continue

                hist_pose = record['pose']
                # [修正] 这里的 key 必须和 finalize_room_semantics 中存储的一致 ('frame')
                record_frame_idx = record.get('frame', -1)

                age_frames = frame_index - record_frame_idx
                if age_frames < 0:
                    continue  # 防止异常

                # 计算衰减后的 alpha（线性衰减）
                alpha_factor = 1.0 - (age_frames / FADE_OUT_FRAMES)
                alpha = max(alpha_factor * 0.4, MIN_ALPHA)  # 初始alpha≈0.4，逐渐减弱
                if alpha <= MIN_ALPHA and alpha_factor <= 0:
                    continue  # 完全消失的不画

                vec_fwd = -hist_pose[:3, 2]
                angle = np.degrees(np.arctan2(vec_fwd[2] * -1, vec_fwd[0]))
                hx, hz = to_plot_coords(hist_pose[0, 3], hist_pose[2, 3])

                # 楔形（FOV） - 透明度随年龄衰减
                wedge = Wedge((hx, hz), r=5.0, theta1=angle-45, theta2=angle+45,
                              color='blue', alpha=alpha, edgecolor=None)
                ax.add_patch(wedge)

                # 中心点（Diamond） - 也衰减
                point_alpha = max(alpha_factor * 0.8, MIN_ALPHA * 2)
                ax.scatter(hx, hz, c='blue', s=40, marker='D', zorder=4,
                           edgecolors='black', linewidth=0.8, alpha=point_alpha)

        # --- 4. 绘制轨迹 & 当前位置 ---
        traj_arr = np.array(self.floor_trajectories[curr_floor])
        if len(traj_arr) > 0:
            tx, tz = to_plot_coords(traj_arr[:, 0], traj_arr[:, 2])
            plt.plot(tx, tz, color='red', linestyle='-', linewidth=4.0, alpha=0.8, label='Path', zorder=3)

        curr_fwd = -current_pose[:3, 2]
        curr_angle = np.degrees(np.arctan2(curr_fwd[2] * -1, curr_fwd[0]))

        # 当前位置红色扇形
        curr_wedge = Wedge((cx, cz), r=5.0, theta1=curr_angle-45, theta2=curr_angle+45, color='red', alpha=0.3)
        ax.add_patch(curr_wedge)
        ax.scatter(cx, cz, c='red', s=50, edgecolors='black', zorder=5)

        # --- 5. 局部聚焦缩放 (Local Zoom Focus) ---

        FOCUS_RADIUS_M = 4.0  # 聚焦半径 4米

        zoom_xs = [cx]
        zoom_zs = [cz]

        # 筛选：只加入附近的物体坐标用于计算边界
        for (px, pz) in plotted_objects.values():
            dist = np.sqrt((px - cx)**2 + (pz - cz)**2)
            if dist < FOCUS_RADIUS_M:
                zoom_xs.append(px)
                zoom_zs.append(pz)

        # 如果轨迹点也在附近，也加进去 (只取最近50个)
        if len(traj_arr) > 0:
            recent_traj = traj_arr[-20:]
            tx_rec, tz_rec = to_plot_coords(recent_traj[:, 0], recent_traj[:, 2])
            zoom_xs.extend(tx_rec)
            zoom_zs.extend(tz_rec)

        # 计算局部边界
        min_x, max_x = min(zoom_xs), max(zoom_xs)
        min_z, max_z = min(zoom_zs), max(zoom_zs)

        center_x = (min_x + max_x) / 2
        center_z = (min_z + max_z) / 2

        # 强制最小视窗大小
        MIN_WINDOW_SIZE = 2.0
        span_x = max(max_x - min_x, MIN_WINDOW_SIZE)
        span_z = max(max_z - min_z, MIN_WINDOW_SIZE)

        max_span = max(span_x, span_z)

        # 留一点 Padding
        pad = max_span * 0.1
        half_side = (max_span / 2) + pad
        half_side = half_side / 2
        # 设置正方形视口
        ax.set_xlim(center_x - half_side, center_x + half_side)
        ax.set_ylim(center_z - half_side, center_z + half_side)

        # --- 设置与保存 ---
        plt.title(f"Map F{curr_floor} | Frame {frame_index} | {len(plotted_objects)} Objects", fontsize=12)
        plt.axis('equal')
        plt.grid(True, linestyle=':', alpha=0.3)
        plt.xticks([]); plt.yticks([])

        if return_path_only:
            save_path = os.path.join(self.run_dir, "vlm_decision_map_temp.jpg")
        else:
            save_path = os.path.join(floor_map_dir, f"bev_map_{frame_index:04d}.jpg")

        plt.savefig(save_path, dpi=120, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        return save_path
    # [修改] 执行全局 Life-Long 更新

    # [辅助] 重新分配物体

    def _compute_centroid(self, objects: List[Dict]) -> np.ndarray:
        if not objects: return np.array([0.0, 0.0])
        coords = []
        for obj in objects:
            pos = obj.get('global_pos_3d')
            if pos:
                coords.append([pos[0], pos[2]]) # 只取 X, Z
        if not coords: return np.array([0.0, 0.0])
        return np.mean(coords, axis=0)

    # [修改] 离线合并逻辑：基于接触面宽度 (Interface Width)

    # [修改] 核心优化函数 (已移除合并步骤)
    def cleanup_and_optimize_rooms(self):
        print("\n>>> Starting Global Room Optimization & Cleanup (NO MERGE) <<<")

        # --- 0. 准备原始 Masks ---
        raw_masks = {}
        # 优先从 GeometryRoomTracker 获取（如果有），否则从 MapManager 获取
        if hasattr(self, 'room_segmenter') and hasattr(self.room_segmenter, 'global_id_grid'):
            # 适配新版 GeometryRoomTracker
            unique_ids = np.unique(self.room_segmenter.global_id_grid)
            for uid in unique_ids:
                if uid > 0:
                    mask = (self.room_segmenter.global_id_grid == uid).astype(np.uint8)
                    if np.sum(mask) > 0: raw_masks[int(uid)] = mask
        elif hasattr(self.room_segmenter, 'persistent_masks') and self.room_segmenter.persistent_masks:
            # 适配旧版
            raw_masks = self.room_segmenter.persistent_masks.copy()
        else:
            # 兜底
            unique_ids = np.unique(self.map_manager.room_id_map)
            for uid in unique_ids:
                if uid > 0: raw_masks[int(uid)] = (self.map_manager.room_id_map == uid).astype(np.uint8)

        if not raw_masks: return []

        # --- 1. [修改] 跳过合并 (Skip Merge) ---
        # 原逻辑：merged_masks, merge_mapping = self._merge_overfragmented_rooms(raw_masks)

        # 新逻辑：直接透传，不进行合并
        merged_masks = raw_masks

        # 由于没有合并，ID 没有变化，因此不需要同步 room_frame_history
        # (原有的 history merge 循环已被移除)

        # --- 2. 统计物体 (基于当前 Mask) ---
        room_obj_counts = defaultdict(int)
        current_floor_objs = [item for item in self.memory_kb if item.get('floor_id') == self.floor_tracker.current_floor_id]

        for item in current_floor_objs:
            pos = item.get('global_pos_3d')
            if pos:
                u, v = self.map_manager.world_to_grid(pos[0], pos[2])
                if self.map_manager._valid_uv(u, v):
                    for rid, mask in merged_masks.items():
                        if mask[v, u] > 0:
                            room_obj_counts[rid] += 1
                            break

        optimized_masks = {}
        valid_room_ids = []

        # --- 3. 剪枝与规则化 ---
        # 稍微调低剪枝阈值，防止把切碎的小房间误删
        AREA_PRUNE_THRESH = 150 # 原来是300，既然不合并了，小房间可能就是合理的，稍微宽容点

        for rid, mask in merged_masks.items():
            area_px = np.sum(mask)
            obj_count = room_obj_counts[rid]

            # 1. 极小噪点 (太小的还是得删)
            if area_px < 50: continue

            # 2. 无物体的小房间
            if obj_count == 0 and area_px < AREA_PRUNE_THRESH:
                print(f"  [Prune] Room {rid} removed (Area: {area_px}, Objs: 0)")
                continue

            # 3. 巨大的空地 (可选保留，这里删除)
            if obj_count == 0 and area_px < 2000:
                print(f"  [Prune] Large Empty Room {rid} removed (Area: {area_px}, Objs: 0)")
                continue

            # [规则化] 直角化处理
            if hasattr(self.map_manager, 'regularize_mask_rectilinear'):
                reg_mask = self.map_manager.regularize_mask_rectilinear(mask)
            else:
                reg_mask = mask

            optimized_masks[rid] = reg_mask
            valid_room_ids.append(rid)

        # --- 4. 去重叠 ---
        if hasattr(self.map_manager, 'resolve_global_room_overlaps'):
            final_map = self.map_manager.resolve_global_room_overlaps(optimized_masks)
            self.map_manager.room_id_map = final_map
        else:
            # 简单的兜底：如果没有去重叠函数，直接覆盖
            final_map = np.zeros((self.map_manager.map_h, self.map_manager.map_w), dtype=np.int32)
            for rid, mask in optimized_masks.items():
                final_map[mask > 0] = rid
            self.map_manager.room_id_map = final_map

        # --- 5. 最终确认 ---
        final_valid_ids = []
        unique_final = np.unique(self.map_manager.room_id_map)
        for uid in unique_final:
            if uid > 0: final_valid_ids.append(int(uid))

        print(f"  Optimization Done. Final Valid Rooms: {final_valid_ids}")
        return final_valid_ids
    # -----------------------------------------------------------
    # [修改] 结算逻辑：保存 RGB 图片
    # -----------------------------------------------------------
    # [修改] 结算房间语义：增加 Fallback 机制
    def finalize_room_semantics(self, room_id, subset_objects=None, force_vlm_expansion=False):
        if room_id is None or room_id == 0: return None
        try: r_key = int(room_id)
        except: r_key = room_id

        print(f"\n>>> Finalizing Semantic Tree for Room ID: {r_key} (Force: {force_vlm_expansion}) <<<")
        current_time_str = datetime.datetime.now().isoformat()

        # 1. Mask
        mask = (self.map_manager.room_id_map == r_key).astype(np.uint8)
        if np.sum(mask) == 0:
            print(f"  [Skip] Room {r_key} has no area.")
            return None

        save_path = os.path.join(self.run_dir, f"mask_room_{r_key}.png")
        cv2.imwrite(save_path, (mask * 255).astype(np.uint8))

        # 2. Objects
        room_objects = []
        for item in self.memory_kb:
            if item.get('floor_id') != self.floor_tracker.current_floor_id: continue
            pos = item.get('global_pos_3d')
            if pos is None: continue
            if self.map_manager.is_point_in_room(pos, r_key):
                item['parent_room'] = r_key
                room_objects.append(item)
            elif item.get('parent_room') == r_key:
                item['parent_room'] = None

        if len(room_objects) == 0:
            print(f"  [Skip] Room {r_key} is empty.")
            return None

        # 3. Best View Selection (计算最佳视角)
        best_frame_idx = -1
        max_overlap = -1
        # [修改] 此时 history 应该已经通过 cleanup_and_optimize_rooms 进行了合并
        candidate_frames = self.room_frame_history.get(r_key, [])

        if candidate_frames:
            step = max(1, len(candidate_frames) // 50)
            for i in range(0, len(candidate_frames), step):
                f_idx = candidate_frames[i]
                pose = self.pose_cache.get(f_idx)
                if pose is not None:
                    overlap = self.map_manager.calculate_visible_room_area(pose, r_key)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_frame_idx = f_idx

            # [新增] Fallback 机制：如果计算失败 (overlap=0)，强制取中间帧
            if best_frame_idx == -1 and len(candidate_frames) > 0:
                print(f"  [View Selection] No optimal view found (max_overlap={max_overlap}). Using middle frame as fallback.")
                best_frame_idx = candidate_frames[len(candidate_frames) // 2]

        best_image_path = None
        saved_summary_img_path = None

        if best_frame_idx >= 0:
            # 获取原始 RGB 路径
            best_image_path, _ = self._get_image_path(best_frame_idx)
            best_pose = self.pose_cache.get(best_frame_idx)

            # 记录历史更新点
            if best_pose is not None:
                if not hasattr(self, 'global_update_history'): self.global_update_history = []
                self.global_update_history.append({
                    'pose': best_pose,
                    'floor': self.floor_tracker.current_floor_id,
                    'frame': best_frame_idx,
                    'room_id': r_key
                })

            # 强制保存图片
            if best_image_path and os.path.exists(best_image_path):
                if not os.path.exists(self.room_summary_img_dir):
                    os.makedirs(self.room_summary_img_dir, exist_ok=True)

                img_name = f"room_{r_key}_floor_{self.floor_tracker.current_floor_id}_summary.jpg"
                saved_summary_img_path = os.path.join(self.room_summary_img_dir, img_name)

                try:
                    shutil.copy(best_image_path, saved_summary_img_path)
                    print(f"  [IO] Saved Best View Image: {saved_summary_img_path}")
                except Exception as e:
                    print(f"  [Error] Failed to copy summary image: {e}")

                if self.config.get('debug_mode', True):
                    vlm_debug_dir = os.path.join(self.run_dir, "debug_vlm_images")
                    os.makedirs(vlm_debug_dir, exist_ok=True)
                    shutil.copy(best_image_path, os.path.join(vlm_debug_dir, img_name))
        else:
            print(f"  [Warning] No valid best view frame found for Room {r_key} (History Len: {len(candidate_frames)})")

        # 4. VLM Expansion (保持不变)
        vlm_augmented_labels = []
        if r_key in self.room_metadata_cache and 'augmented_labels' in self.room_metadata_cache[r_key]:
            vlm_augmented_labels.extend(self.room_metadata_cache[r_key]['augmented_labels'])

        should_run_vlm = force_vlm_expansion or (not vlm_augmented_labels)
        if should_run_vlm and best_image_path and os.path.exists(best_image_path) and hasattr(self, 'llm'):
            print(f"  [VLM] Scanning room image (Force={force_vlm_expansion})...")
            existing_labels = list(set([obj['label'].split(' ')[0].split('_')[0] for obj in room_objects]))
            detected_new = self.llm.detect_room_objects_vlm(best_image_path, existing_labels)

            ignore_set = {'wall', 'walls', 'floor', 'floors', 'ceiling', 'ceilings', 'item', 'object', 'unknown'}
            for label in detected_new:
                clean_l = label.lower().strip()
                if clean_l not in existing_labels and clean_l not in ignore_set and clean_l not in vlm_augmented_labels:
                    vlm_augmented_labels.append(clean_l)

            if r_key not in self.room_metadata_cache: self.room_metadata_cache[r_key] = {}
            self.room_metadata_cache[r_key]['augmented_labels'] = vlm_augmented_labels

        # 5. Clustering (保持不变)
        cluster_graph = nx.Graph()
        for i in range(len(room_objects)): cluster_graph.add_node(i)

        for i in range(len(room_objects)):
            for j in range(i + 1, len(room_objects)):
                p1 = room_objects[i].get('global_pos_3d')
                p2 = room_objects[j].get('global_pos_3d')
                if p1 and p2:
                    dist = np.linalg.norm(np.array([p1[0], p1[2]]) - np.array([p2[0], p2[2]]))
                    if dist < DIST_THRESH: cluster_graph.add_edge(i, j)
        new_area_clusters = list(nx.connected_components(cluster_graph))

        # 6. Generate Nodes (保持不变)
        new_area_ids = []
        csv_file = open(self.edges_path, 'a', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        room_node_id = f"room_{r_key}"
        all_area_summaries = []


        for idx, indices in enumerate(new_area_clusters):
            cluster_objs = [room_objects[i] for i in indices]
            centroid = self._compute_centroid(cluster_objs)
            llm_input = [{"tag": o['label'], "desc": o.get('llm_desc', '')} for o in cluster_objs]

            if len(cluster_objs) > 0:
                llm_area_summary_start = time.time()
                llm_res = self.llm.generate_spatial_cluster_summary(llm_input)
                print(f"  [LLM] Area summaries generated in {time.time() - llm_area_summary_start:.2f}s.")

                area_tag = llm_res.get('area_tag', 'area')
                area_summary = llm_res.get('summary', 'Cluster of objects.')
            else:
                area_tag = "empty_area"; area_summary = "Empty."

            uid = int(time.time() * 1000) % 100000
            area_id = f"area_{r_key}_{idx}_{uid}"
            new_area_ids.append(area_id)
            all_area_summaries.append({"tag": area_tag, "summary": area_summary, "id": area_id})

            area_node = {
                "node_id": area_id, "node_level": 2, "area_tag": area_tag, "llm_summary": area_summary,
                "contained_obj_nodes": [o['node_id'] for o in cluster_objs],
                "parent_room_id": room_node_id, "centroid": centroid.tolist(), "timestamp": current_time_str
            }
            with open(self.nodes_path, 'a', encoding='utf-8') as f: f.write(json.dumps(area_node, ensure_ascii=False) + "\n")

            for o in cluster_objs:
                csv_writer.writerow([o['node_id'], area_id, 3])
                o['parent_area'] = area_id
            csv_writer.writerow([area_id, room_node_id, 2])

        # 7. Generate Room Node (保持不变)
        llm_room_summary_start = time.time()
        room_vlm_result = self.llm.generate_multimodal_room_summary(best_image_path, all_area_summaries)
        print(f"  [LLM] Room summary generated in {time.time() - llm_room_summary_start:.2f}s.")

        room_label = room_vlm_result.get("room_label", "Room")
        room_desc = room_vlm_result.get("description", "A generic room.")

        room_centroid = [0.0, 0.0]
        if room_objects: room_centroid = self._compute_centroid(room_objects).tolist()

        room_node = {
            "node_id": room_node_id, "node_level": 1, "room_label": room_label, "llm_summary": room_desc,
            "floor_id": self.floor_tracker.current_floor_id, "mask_file": f"mask_room_{r_key}.png",
            "centroid": room_centroid, "contained_area_ids": new_area_ids,
            "best_view_image": saved_summary_img_path, # 使用保存的路径
            "vlm_augmented_labels": vlm_augmented_labels,
            "timestamp": current_time_str, "room_type": room_label
        }

        with open(self.nodes_path, 'a', encoding='utf-8') as f: f.write(json.dumps(room_node, ensure_ascii=False) + "\n")
        csv_file.close()
        print(f"  Room Finalized: {room_node_id} | Label: {room_label} | New Areas: {len(new_area_ids)}")
        return room_node_id


    # [新增] 生成用于 VLM 决策的地图图片

    # [新增] 辅助：重新分配物体 (用于 Map 优化后)
    def _reassign_objects_to_rooms(self):
        """Map 形状改变后，重新计算物体归属"""
        for item in self.memory_kb:
            if item.get('floor_id') == self.floor_tracker.current_floor_id:
                pos = item.get('global_pos_3d')
                if pos:
                    u, v = self.map_manager.world_to_grid(pos[0], pos[2])
                    if self.map_manager._valid_uv(u, v):
                        new_rid = self.map_manager.room_id_map[v, u]
                        item['parent_room'] = int(new_rid) if new_rid > 0 else None

    # [新增] 辅助：立即执行标签扩充
    def _perform_immediate_label_expansion(self, room_id: int, frame_index: int):
        """
        根据 VLM 建议，立即对当前房间的最佳视角图进行标签扩充检查。
        """
        # 1. 获取当前帧图像作为观察点 (或者使用 room_frame_history 中已有的最佳帧)
        image_path, _ = self._get_image_path(frame_index)

        # 2. 获取已知标签
        room_objs = [obj for obj in self.memory_kb if obj.get('parent_room') == room_id]
        existing_labels = list(set([o['label'].split('_')[0] for o in room_objs]))

        # 3. 调用 LLM
        if hasattr(self, 'llm'):
            new_labels = self.llm.detect_room_objects_vlm(image_path, existing_labels)
            if new_labels:
                print(f"  [VLM Expansion] Immediate scan found: {new_labels}")
                # 将这些新标签存入缓存，供后续 finalize 使用
                if room_id not in self.room_metadata_cache:
                    self.room_metadata_cache[room_id] = {}

                current_aug = self.room_metadata_cache[room_id].get('augmented_labels', [])
                current_aug.extend(new_labels)
                # 去重
                self.room_metadata_cache[room_id]['augmented_labels'] = list(set(current_aug))


    # [NEW]
    # [修改] 核心运行逻辑

    def _save_map_data_pkl(self):
        map_data = {
            "resolution": self.map_manager.resolution,
            "min_x": self.map_manager.min_x,
            "min_z": self.map_manager.min_z,
            # 可视化脚本用的是 room_mask（每格一个房间id；没有房间则 -1）
            "room_mask": self.map_manager.room_id_map.astype(np.int32),

            # 这些不是必须，但一般有用（可留可删）
            "occ_bin": self.map_manager.occ_bin.astype(np.uint8),
            "free_bin": self.map_manager.free_bin.astype(np.uint8),
            "door_map": self.map_manager.door_map.astype(np.uint8),
        }
        out_path = os.path.join(self.run_dir, "map_data.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(map_data, f)
    # [新增] 楼层索引归一化 (问题 5)

    # [修改] 核心运行 run 方法
    # [修改] 运行主循环
    # =========================================================
    # [新增] 算法结算逻辑封装
    # =========================================================

    # =========================================================
    # [新增] 文件保存逻辑封装
    # =========================================================
    def save_all_data(self):
        """
        统一保存所有关键数据文件。
        包含: Config, Memory KB, Spatial Graph, Map Data。
        (Nodes 和 Edges 文件已在运行过程中流式写入，此处无需再次保存)
        """
        print("\n" + "="*50)
        print(f">>> Saving All Data to: {self.run_dir}")
        print("="*50)

        os.makedirs(self.run_dir, exist_ok=True)

        # 1. 保存 Config 快照
        config_path = os.path.join(self.run_dir, "config_snapshot.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f)
        print(f"  [Save] Config saved to: config_snapshot.yaml")

        # 2. 保存 Memory KB (JSON)
        # 需要处理 numpy/tensor 类型，防止 JSON 序列化报错
        kb_path = os.path.join(self.run_dir, "memory_kb.json")
        serializable_kb = []
        for item in self.memory_kb:
            s_item = item.copy()
            # 将特征向量转为 list
            if 'dinov3_feature' in s_item:
                feat = s_item['dinov3_feature']
                if isinstance(feat, (np.ndarray, torch.Tensor)):
                    s_item['dinov3_feature'] = feat.tolist()
            # 确保 3D 坐标也是 list
            if 'global_pos_3d' in s_item and s_item['global_pos_3d'] is not None:
                if isinstance(s_item['global_pos_3d'], (np.ndarray, torch.Tensor)):
                    s_item['global_pos_3d'] = s_item['global_pos_3d'].tolist()
            serializable_kb.append(s_item)

        with open(kb_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_kb, f, indent=2, ensure_ascii=False)
        print(f"  [Save] Memory KB ({len(serializable_kb)} items) saved to: memory_kb.json")

        # 3. 保存 Spatial Graph (Pickle)
        graph_path = os.path.join(self.run_dir, "spatial_graph.pkl")
        with open(graph_path, 'wb') as f:
            pickle.dump(self.spatial_graph, f)
        print(f"  [Save] Spatial Graph ({self.spatial_graph.number_of_nodes()} nodes) saved to: spatial_graph.pkl")
        with open(os.path.join(self.run_dir, "spatial_graph.json"), "w") as handle:
            json.dump([{"source": str(u), "target": str(v), "relation": data.get("relation", "related_to")}
                       for u, v, data in self.spatial_graph.edges(data=True)], handle)


        # 4. 保存 Map Data (Pickle)
        # 包含分辨率、原点、以及最重要的 room_id_map, occ_bin 等
        self._save_map_data_pkl() # 复用已有的内部方法
        print(f"  [Save] Map Data saved to: map_data.pkl")

        # 处理
        final_nodes = process_nodes_file(self.nodes_path, self.nodes_path)
        prune_relations_file(self.edges_path, final_nodes)

        # 分析
        analyze_area_quality(final_nodes)


        print(">>> All Files Saved Successfully.")


    # -----------------------------------------------------------
    # [修改] Run 循环：整合新的 BEV 生成与 VLM 决策
    # -----------------------------------------------------------




# --- 在 builder.py 文件末尾添加 ---
# ==============================================================================
# 独立运行主函数 (基于 config_sam3.yaml 配置批量建图)
# ==============================================================================

# 设置无头模式，防止 ssh 连接时绘图报错


# 假设 TrajectoryPipeline, Sam3LocalClient, load_feature_extractor 在其他模块中
# from ... import ... (保留你原本的 import)

# ==============================================================================
# [新增] GPUMonitor 类：监控当前 PyTorch 进程显存占用
# ==============================================================================
class GPUMonitor(threading.Thread):
    def __init__(self, save_dir, device, filename="gpu_memory_log.csv", interval=1.0, plot_name="gpu_memory_plot.png"):
        """
        :param save_dir: 保存目录
        :param device: 要监控的设备，例如 'cuda:0' 或 'cuda'
        :param interval: 采样间隔(秒)
        """
        super().__init__()
        self.save_dir = save_dir
        self.csv_path = os.path.join(save_dir, filename)
        self.plot_path = os.path.join(save_dir, plot_name)
        self.interval = interval
        self.device = device
        self.running = False
        self.history = [] # 存储 (time_elapsed, memory_used)
        self.start_time = 0

    def get_gpu_memory(self):
        """
        使用 PyTorch 内置函数获取当前进程在指定设备上的显存占用 (MB)。
        memory_reserved: 返回 PyTorch 缓存分配器管理的当前显存总量。
        """
        try:
            if torch.cuda.is_available():
                # memory_reserved 返回的是字节，转换为 MB
                mem_bytes = torch.cuda.memory_reserved(self.device)
                return int(mem_bytes / 1024 / 1024)
            else:
                return 0
        except Exception:
            return 0

    def run(self):
        self.running = True
        self.start_time = time.time()

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        # 初始化 CSV
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time(s)", "Memory_Reserved(MB)"])

        while self.running:
            current_mem = self.get_gpu_memory()
            elapsed_time = time.time() - self.start_time

            self.history.append((elapsed_time, current_mem))

            # 实时追加写入
            try:
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f"{elapsed_time:.2f}", current_mem])
            except:
                pass

            time.sleep(self.interval)

    def stop(self):
        self.running = False

    def plot_and_save(self):
        """绘制并保存显存曲线"""
        if not self.history:
            return

        times = [x[0] for x in self.history]
        mems = [x[1] for x in self.history]

        plt.figure(figsize=(10, 6))
        plt.plot(times, mems, label=f'VRAM Reserved ({self.device})', color='orange', linewidth=1.5)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory (MB)')
        plt.title(f'Process GPU Memory Usage Over Time ({self.device})')
        plt.grid(True)
        plt.legend()

        try:
            plt.savefig(self.plot_path)
            plt.close()
            print(f"     [Monitor] Plot saved to {self.plot_path}")
        except Exception as e:
            print(f"     [Monitor] Failed to save plot: {e}")
