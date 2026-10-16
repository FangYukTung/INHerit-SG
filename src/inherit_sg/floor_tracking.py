"""Signed floor tracking with the original height quantum and transition debounce."""
import csv
import datetime
import json
from collections import defaultdict, deque

import numpy as np


class FloorTracker:
    def __init__(self, output_path: str, edges_path: str = None, *, initial_floor: int = 0):
        if type(initial_floor) is not int:
            raise ValueError("floor_tracking.initial_floor must be an integer")
        self.initial_floor = initial_floor
        self.nodes_path = output_path
        self.edges_path = edges_path

        # --- 核心参数 ---
        self.floor_height_step = 3.5  # 标准层高 (米)，可根据场景调整
        self.min_consistent_frames = 20 # 确认切换所需的稳定帧数

        # --- 状态管理 ---
        self.current_floor_id = initial_floor
        self.base_height = None  # 启动时的基准高度

        # 缓冲队列，用于消抖
        self.height_buffer = deque(maxlen=self.min_consistent_frames)
        self.id_buffer = deque(maxlen=self.min_consistent_frames)

        # 存储楼层数据: floor_id -> {'heights': [], 'rooms': set(), 'finalized_count': 0}
        # 使用 defaultdict 方便自动创建新楼层
        self.floors_data = defaultdict(lambda: {'heights': [], 'rooms': set(), 'finalized_count': 0})

        # 临时缓存，用于在切换确认前存储数据
        self.pending_data_cache = []

    def _calculate_floor_id(self, y: float) -> int:
        """根据相对于基准面的高度计算楼层 ID (核心量化逻辑)"""
        if self.base_height is None:
            return self.initial_floor
        # 使用 round 进行四舍五入量化
        # 例如: (3.2 - 0) / 3.0 = 1.06 -> 1
        #       (5.8 - 0) / 3.0 = 1.93 -> 2
        return self.initial_floor + int(np.round((y - self.base_height) / self.floor_height_step))

    def add_room(self, room_node_id: str):
        """添加房间到当前楼层"""
        if room_node_id:
            self.floors_data[self.current_floor_id]['rooms'].add(room_node_id)

    def update(self, pose: np.ndarray, current_room_node_id: str) -> bool:
        """
        更新状态，返回 True 表示检测到楼层切换
        """
        if pose is None: return False

        # 获取当前高度 (Habitat 通常 Y 是高度，请根据实际 Pose 矩阵确认)
        # 如果是 OpenCV 坐标系转换来的，通常 [1, 3] 是高度
        y = float(pose[1, 3])

        # 1. 初始化基准面 (第一帧)
        if self.base_height is None:
            self.base_height = y
            self.floors_data[self.initial_floor]['heights'].append(y)
            if current_room_node_id:
                self.floors_data[self.initial_floor]['rooms'].add(current_room_node_id)
            print(f"[FloorTracker] Initialized Base Height: {self.base_height:.2f}m")
            return False

        # 2. 计算当前帧的 建议楼层ID
        suggested_id = self._calculate_floor_id(y)

        # 3. 更新缓冲区
        self.height_buffer.append(y)
        self.id_buffer.append(suggested_id)
        self.pending_data_cache.append(y) # 暂存高度数据

        # 4. 判断是否需要切换
        # 只有当缓冲区满，且缓冲区内所有 ID 都一致，且与当前 ID 不同时
        if len(self.id_buffer) == self.min_consistent_frames:
            unique_ids = set(self.id_buffer)

            if len(unique_ids) == 1:
                target_id = list(unique_ids)[0]

                # 如果检测到的 ID 与当前不同 -> 触发切换
                if target_id != self.current_floor_id:
                    print(f"[FloorTracker] Floor Switch Detected: {self.current_floor_id} -> {target_id}")

                    # 在这里，我们只返回 True 通知 Pipeline 进行结算
                    # 真正的数据更新在 finalize_floor 和后续逻辑中处理
                    # 但为了防止 Pipeline 还没调用 finalize 就继续 update，
                    # 我们需要在这里暂存目标 ID，或者由外部逻辑控制

                    # 策略：Pipeline 收到 True 后会调用 finalize_floor，然后 switch_floor
                    # 我们在这里先不改变 self.current_floor_id

                    # [重要] 这是一个预判，我们把 target_id 存起来供外部获取
                    self.next_floor_id = target_id
                    return True

        # 5. 如果没有切换，将暂存数据归入当前楼层
        # 为了避免数据无限积压，我们定期清理 pending_data_cache 到当前楼层
        # 但在楼梯上时（ID 不稳定），我们希望暂时不归入，等稳定了再归入
        # 简化策略：只要没触发切换信号，就认为是当前楼层的数据
        if len(self.pending_data_cache) > 0:
            # 注意：如果正在上楼梯（buffer里有混杂ID），这些高度会被加到旧楼层
            # 这其实是合理的，楼梯属于连接区域。
            # 只有当完全进入新楼层平面后，才算新楼层。
            self.floors_data[self.current_floor_id]['heights'].extend(self.pending_data_cache)
            self.pending_data_cache = []

            if current_room_node_id:
                self.floors_data[self.current_floor_id]['rooms'].add(current_room_node_id)

        return False

    def get_floor_id_by_object_y(self, obj_y: float) -> int:
        """
        根据物体高度返回楼层 ID。
        优先使用量化公式，如果量化结果对应的楼层有记录，则进行微调。
        """
        # 1. 直接量化计算 (最鲁棒的方法)
        calc_id = self._calculate_floor_id(obj_y)
        # if calc_id == 1 or calc_id == -1:
        #     print(f"[FloorTracker] Object Y: {obj_y:.2f}m -> Calculated Floor ID: {calc_id}")
        #     assert False, "Debug Breakpoint"

        # 2. 检查该 ID 是否存在数据，如果存在，可以做个简单的范围验证 (可选)
        # 这里直接返回计算值，因为它支持泛化到还没去过的楼层
        return calc_id

    def finalize_floor(self):
        """结算当前楼层节点并保存"""
        fid = self.current_floor_id
        data = self.floors_data[fid]

        # 计算平均高度
        heights = data['heights']
        if not heights:
            avg_h = self.base_height + ((fid - self.initial_floor) * self.floor_height_step)
        else:
            # 简单的去噪平均：取中间 80% 的数据
            if len(heights) > 10:
                sorted_h = np.sort(heights)
                lower = int(len(heights) * 0.1)
                upper = int(len(heights) * 0.9)
                avg_h = np.mean(sorted_h[lower:upper])
            else:
                avg_h = np.mean(heights)

        rooms = list(data['rooms'])
        floor_node_id = f"floor_{fid:02d}" if fid >= 0 else f"floor_neg_{abs(fid):02d}"

        # 增加 finalized_count，支持多次进入同一楼层
        data['finalized_count'] += 1

        summary = f"Floor {fid} (Avg Height: {avg_h:.2f}m). Contains {len(rooms)} rooms."

        node = {
            "node_id": floor_node_id,
            "node_level": 0,
            "floor_index": fid, # 数值型索引
            "height_info": {"avg": avg_h, "min": np.min(heights) if heights else avg_h, "max": np.max(heights) if heights else avg_h},
            "contained_room_nodes": rooms,
            "llm_summary": summary,
            "timestamp": datetime.datetime.now().isoformat(),
            "visit_count": data['finalized_count']
        }

        # 写入 Floor Node
        # 注意：因为可能多次回到同一楼层，这里只是追加记录。
        # 下游读取时可以根据 timestamp 取最新的，或者合并。
        with open(self.nodes_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(node, ensure_ascii=False) + "\n")

        # 写入 Room -> Floor 边
        if self.edges_path and rooms:
            with open(self.edges_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for rid in rooms:
                    # 避免重复写入边 (可选优化，这里简单追加)
                    writer.writerow([rid, floor_node_id, 1])

        print(f"[FloorTracker] Finalized {floor_node_id}. Prepare to switch.")

        # --- 切换状态 ---
        # 如果 update 中检测到了 next_floor_id，则应用它
        if hasattr(self, 'next_floor_id'):
            self.current_floor_id = self.next_floor_id
            # 清除标记
            del self.next_floor_id

            # 清空缓冲区，避免切换后立即又触发
            self.height_buffer.clear()
            self.id_buffer.clear()
            self.pending_data_cache = [] # 新楼层开始为空
