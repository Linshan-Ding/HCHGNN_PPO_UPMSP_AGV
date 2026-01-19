"""
instance_loader.py - CSV算例加载器
"""

import os
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class InstanceData:
    """算例数据"""
    num_workpiece_types: int
    num_orders: int
    num_agvs: int
    num_machines: int
    map_width: float
    map_height: float
    agv_speed: float
    buffer_capacity: int
    order_dispatch_point: Tuple[float, float]
    warehouse_position: Tuple[float, float]
    orders: List[Tuple[int, int, int, float]]  # (id, type, qty, due)
    agvs: List[Tuple[int, float, float, float]]  # (id, x, y, speed)
    machines: List[Tuple[int, float, float, float, int]]  # (id, x, y, switch, buffer)
    process_times: Dict[int, Dict[int, float]]  # machine -> type -> time
    
    @property
    def total_workpieces(self) -> int:
        return sum(o[2] for o in self.orders)


def load_instance(instance_dir: str) -> InstanceData:
    """加载算例"""
    cfg = {}
    with open(os.path.join(instance_dir, "config.csv"), 'r') as f:
        for row in csv.DictReader(f):
            p, v = row['parameter'], row['value']
            cfg[p] = int(v) if p in ['num_workpiece_types', 'num_orders', 'num_agvs', 
                                      'num_machines', 'buffer_capacity'] else float(v)
    
    orders = []
    with open(os.path.join(instance_dir, "orders.csv"), 'r') as f:
        for row in csv.DictReader(f):
            orders.append((int(row['order_id']), int(row['workpiece_type']),
                          int(row['num_workpieces']), float(row['due_date'])))
    
    agvs = []
    with open(os.path.join(instance_dir, "agvs.csv"), 'r') as f:
        for row in csv.DictReader(f):
            agvs.append((int(row['agv_id']), float(row['init_x']),
                        float(row['init_y']), float(row['speed'])))
    
    machines = []
    with open(os.path.join(instance_dir, "machines.csv"), 'r') as f:
        for row in csv.DictReader(f):
            machines.append((int(row['machine_id']), float(row['pos_x']),
                           float(row['pos_y']), float(row['switch_time']),
                           int(row['buffer_capacity'])))
    
    process_times = {}
    with open(os.path.join(instance_dir, "process_times.csv"), 'r') as f:
        for row in csv.DictReader(f):
            mid = int(row['machine_id'])
            process_times[mid] = {}
            for k, v in row.items():
                if k.startswith('type_'):
                    process_times[mid][int(k.split('_')[1])] = float(v)
    
    return InstanceData(
        num_workpiece_types=cfg['num_workpiece_types'],
        num_orders=cfg['num_orders'],
        num_agvs=cfg['num_agvs'],
        num_machines=cfg['num_machines'],
        map_width=cfg['map_width'],
        map_height=cfg['map_height'],
        agv_speed=cfg['agv_speed'],
        buffer_capacity=cfg['buffer_capacity'],
        order_dispatch_point=(cfg['order_dispatch_x'], cfg['order_dispatch_y']),
        warehouse_position=(cfg['warehouse_x'], cfg['warehouse_y']),
        orders=orders, agvs=agvs, machines=machines, process_times=process_times
    )


def list_instances(instances_dir: str) -> List[str]:
    """列出所有算例"""
    if not os.path.exists(instances_dir):
        return []
    return sorted([d for d in os.listdir(instances_dir) 
                   if os.path.isfile(os.path.join(instances_dir, d, "config.csv"))])


def print_instance_info(data: InstanceData):
    """打印算例信息"""
    print(f"Orders: {data.num_orders}, AGVs: {data.num_agvs}, "
          f"Machines: {data.num_machines}, Types: {data.num_workpiece_types}, "
          f"Total Workpieces: {data.total_workpieces}")
