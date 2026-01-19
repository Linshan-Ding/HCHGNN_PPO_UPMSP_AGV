"""
graph_state.py - 异质图状态（4类节点）
特征不做归一化，由HGNN进行标准化处理
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from config import config
from utils import euclidean_distance


def one_hot(index: int, num_classes: int) -> np.ndarray:
    vec = np.zeros(num_classes, dtype=np.float32)
    if 0 <= index < num_classes:
        vec[index] = 1.0
    return vec


@dataclass
class Order:
    """订单"""
    order_id: int
    workpiece_type: int
    total_workpieces: int
    remaining_workpieces: int
    due_date: float
    position: Tuple[float, float]
    num_types: int = 10
    
    def get_features(self) -> np.ndarray:
        return np.concatenate([
            one_hot(self.workpiece_type, self.num_types),
            [float(self.remaining_workpieces)],
            [self.due_date],
            [self.position[0]],
            [self.position[1]]
        ]).astype(np.float32)


@dataclass
class AGV:
    """AGV"""
    agv_id: int
    position: Tuple[float, float]
    speed: float = 1.0
    state: int = 0  # 0:Idle, 1:ToWaiting, 2:Waiting, 3:ToBuffer, 4:ToWarehouse
    carrying_type: Optional[int] = None
    carrying_order_id: Optional[int] = None
    target_machine_id: Optional[int] = None
    busy_until: float = 0.0
    num_types: int = 10
    
    def get_features(self) -> np.ndarray:
        carrying_onehot = one_hot(self.carrying_type if self.carrying_type is not None else -1, self.num_types)
        return np.concatenate([
            one_hot(self.state, 5),
            [self.position[0]],
            [self.position[1]],
            [self.speed],
            [self.busy_until],
            carrying_onehot
        ]).astype(np.float32)
    
    def is_idle(self) -> bool:
        return self.state == 0


@dataclass
class WaitingArea:
    """机器等待区"""
    machine_id: int
    position: Tuple[float, float]
    queue: List[int] = field(default_factory=list)
    
    def get_features(self) -> np.ndarray:
        return np.array([
            self.position[0],
            self.position[1],
            float(len(self.queue)),
            float(len(self.queue) > 0)
        ], dtype=np.float32)
    
    def has_waiting(self) -> bool:
        return len(self.queue) > 0


@dataclass
class BufferArea:
    """机器缓存区"""
    machine_id: int
    position: Tuple[float, float]
    capacity: int = 2
    finished_count: int = 0
    
    def get_features(self) -> np.ndarray:
        return np.array([
            self.position[0],
            self.position[1],
            float(self.finished_count),
            float(self.capacity),
            float(self.finished_count > 0),
            float(self.finished_count < self.capacity)
        ], dtype=np.float32)
    
    def has_finished(self) -> bool:
        return self.finished_count > 0
    
    def can_accept(self) -> bool:
        return self.finished_count < self.capacity


@dataclass
class Machine:
    """机器"""
    machine_id: int
    position: Tuple[float, float]
    process_times: Dict[int, float]
    switch_time: float
    is_processing: bool = False
    current_type: Optional[int] = None
    last_type: Optional[int] = None
    busy_until: float = 0.0
    
    def can_start(self, buffer: BufferArea, waiting: WaitingArea) -> bool:
        return (not self.is_processing and 
                buffer.finished_count < buffer.capacity and 
                waiting.has_waiting())


class GraphData:
    """异质图数据"""
    def __init__(self):
        self.order_features: torch.Tensor = None
        self.agv_features: torch.Tensor = None
        self.waiting_features: torch.Tensor = None
        self.buffer_features: torch.Tensor = None
        
        self.order_to_agv_edge_index: torch.Tensor = None
        self.agv_to_order_edge_index: torch.Tensor = None
        self.order_agv_edge_feat: torch.Tensor = None
        
        self.agv_to_waiting_edge_index: torch.Tensor = None
        self.waiting_to_agv_edge_index: torch.Tensor = None
        self.agv_waiting_edge_feat: torch.Tensor = None
        
        self.agv_to_buffer_edge_index: torch.Tensor = None
        self.buffer_to_agv_edge_index: torch.Tensor = None
        self.agv_buffer_edge_feat: torch.Tensor = None
        
        # 注：删除了 WaitingArea <-> BufferArea 边
        
        self.action_mask: torch.Tensor = None


class StateBuilder:
    """状态构建器"""
    
    def __init__(self, orders: List[Order], agvs: List[AGV], 
                 waitings: List[WaitingArea], buffers: List[BufferArea],
                 machines: List[Machine]):
        self.orders = orders
        self.agvs = agvs
        self.waitings = waitings
        self.buffers = buffers
        self.machines = machines
        
        self.num_orders = len(orders)
        self.num_agvs = len(agvs)
        self.num_machines = len(machines)
        
        self.order_feat_dim = len(orders[0].get_features())
        self.agv_feat_dim = len(agvs[0].get_features())
        self.waiting_feat_dim = len(waitings[0].get_features())
        self.buffer_feat_dim = len(buffers[0].get_features())
    
    def build_graph(self, device=config.DEVICE) -> GraphData:
        """构建异质图（原始特征，不归一化）"""
        g = GraphData()
        
        g.order_features = torch.tensor(
            np.stack([o.get_features() for o in self.orders]),
            dtype=torch.float32, device=device)
        g.agv_features = torch.tensor(
            np.stack([a.get_features() for a in self.agvs]),
            dtype=torch.float32, device=device)
        g.waiting_features = torch.tensor(
            np.stack([w.get_features() for w in self.waitings]),
            dtype=torch.float32, device=device)
        g.buffer_features = torch.tensor(
            np.stack([b.get_features() for b in self.buffers]),
            dtype=torch.float32, device=device)
        
        # Order <-> AGV 边（只保留距离）
        o2a_src, o2a_dst, o2a_feat = [], [], []
        for i, o in enumerate(self.orders):
            for j, a in enumerate(self.agvs):
                o2a_src.append(i)
                o2a_dst.append(j)
                dist = euclidean_distance(o.position, a.position)
                o2a_feat.append([dist])
        g.order_to_agv_edge_index = torch.tensor([o2a_src, o2a_dst], dtype=torch.long, device=device)
        g.agv_to_order_edge_index = torch.tensor([o2a_dst, o2a_src], dtype=torch.long, device=device)
        g.order_agv_edge_feat = torch.tensor(o2a_feat, dtype=torch.float32, device=device)
        
        # AGV <-> WaitingArea 边（距离 + 切换时间）
        a2w_src, a2w_dst, a2w_feat = [], [], []
        for i, a in enumerate(self.agvs):
            for j, w in enumerate(self.waitings):
                a2w_src.append(i)
                a2w_dst.append(j)
                dist = euclidean_distance(a.position, w.position)
                # 计算切换时间：AGV携带的工件类型与机器上次加工类型不同则需要切换
                machine = self.machines[j]
                if a.carrying_type is not None and machine.last_type is not None:
                    switch_time = machine.switch_time if a.carrying_type != machine.last_type else 0.0
                else:
                    switch_time = 0.0
                a2w_feat.append([dist, switch_time])
        g.agv_to_waiting_edge_index = torch.tensor([a2w_src, a2w_dst], dtype=torch.long, device=device)
        g.waiting_to_agv_edge_index = torch.tensor([a2w_dst, a2w_src], dtype=torch.long, device=device)
        g.agv_waiting_edge_feat = torch.tensor(a2w_feat, dtype=torch.float32, device=device)
        
        # AGV <-> BufferArea 边（只保留距离）
        a2b_src, a2b_dst, a2b_feat = [], [], []
        for i, a in enumerate(self.agvs):
            for j, b in enumerate(self.buffers):
                a2b_src.append(i)
                a2b_dst.append(j)
                dist = euclidean_distance(a.position, b.position)
                a2b_feat.append([dist])
        g.agv_to_buffer_edge_index = torch.tensor([a2b_src, a2b_dst], dtype=torch.long, device=device)
        g.buffer_to_agv_edge_index = torch.tensor([a2b_dst, a2b_src], dtype=torch.long, device=device)
        g.agv_buffer_edge_feat = torch.tensor(a2b_feat, dtype=torch.float32, device=device)
        
        # 注：删除了 WaitingArea <-> BufferArea 边（通过AGV节点间接相连）
        
        # 动作掩码（防死锁）
        num_dispatch = self.num_orders * self.num_machines
        num_pickup = self.num_machines
        num_targets = num_dispatch + num_pickup
        
        mask = np.zeros(self.num_agvs * num_targets, dtype=np.float32)
        
        has_pickup_task = any(b.has_finished() for b in self.buffers)
        buffer_full_machines = {mi for mi, b in enumerate(self.buffers) if b.finished_count >= b.capacity}
        has_dispatch_task = any(o.remaining_workpieces > 0 for o in self.orders)
        
        for ai, a in enumerate(self.agvs):
            if not a.is_idle():
                continue
            base = ai * num_targets
            
            if has_pickup_task:
                for mi, b in enumerate(self.buffers):
                    if b.has_finished():
                        mask[base + num_dispatch + mi] = 1.0
            elif has_dispatch_task:
                for oi, o in enumerate(self.orders):
                    if o.remaining_workpieces > 0:
                        for mi in range(self.num_machines):
                            if mi not in buffer_full_machines:
                                mask[base + oi * self.num_machines + mi] = 1.0
        
        g.action_mask = torch.tensor(mask, dtype=torch.float32, device=device)
        return g
    
    def decode_action(self, action: int) -> Tuple[int, str, int, int]:
        num_dispatch = self.num_orders * self.num_machines
        num_targets = num_dispatch + self.num_machines
        agv_id = action // num_targets
        target = action % num_targets
        if target < num_dispatch:
            return agv_id, "dispatch", target // self.num_machines, target % self.num_machines
        return agv_id, "pickup", -1, target - num_dispatch
    
    def get_action_space_size(self) -> int:
        return self.num_agvs * (self.num_orders * self.num_machines + self.num_machines)


def create_entities_from_instance(instance_data):
    """从算例创建实体"""
    orders = [Order(oid, wtype, qty, qty, due, instance_data.order_dispatch_point, instance_data.num_workpiece_types)
              for oid, wtype, qty, due in instance_data.orders]
    
    agvs = [AGV(aid, (x, y), spd, num_types=instance_data.num_workpiece_types)
            for aid, x, y, spd in instance_data.agvs]
    
    machines, waitings, buffers = [], [], []
    for mid, x, y, sw, buf in instance_data.machines:
        pos = (x, y)
        machines.append(Machine(mid, pos, instance_data.process_times.get(mid, {}), sw))
        waitings.append(WaitingArea(mid, pos))
        buffers.append(BufferArea(mid, pos, buf))
    
    return orders, agvs, waitings, buffers, machines
