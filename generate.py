"""
generate.py - 算例生成器
"""

import os
import csv
import argparse
import numpy as np


def generate_instance(output_dir: str, name: str, num_types: int, num_orders: int,
                      avg_workpieces: int, num_agvs: int, num_machines: int, seed: int = 42):
    """生成单个算例"""
    np.random.seed(seed)
    
    instance_dir = os.path.join(output_dir, name)
    os.makedirs(instance_dir, exist_ok=True)
    
    with open(os.path.join(instance_dir, "config.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['parameter', 'value'])
        w.writerow(['num_workpiece_types', num_types])
        w.writerow(['num_orders', num_orders])
        w.writerow(['num_agvs', num_agvs])
        w.writerow(['num_machines', num_machines])
        w.writerow(['map_width', 100.0])
        w.writerow(['map_height', 100.0])
        w.writerow(['agv_speed', 1.0])
        w.writerow(['buffer_capacity', 2])
        w.writerow(['order_dispatch_x', 0.0])
        w.writerow(['order_dispatch_y', 50.0])
        w.writerow(['warehouse_x', 100.0])
        w.writerow(['warehouse_y', 50.0])
    
    with open(os.path.join(instance_dir, "orders.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['order_id', 'workpiece_type', 'num_workpieces', 'due_date'])
        for i in range(num_orders):
            wtype = np.random.randint(0, num_types)
            qty = max(1, int(np.random.normal(avg_workpieces, avg_workpieces * 0.3)))
            due = np.random.uniform(100, 500)
            w.writerow([i, wtype, qty, f"{due:.2f}"])
    
    with open(os.path.join(instance_dir, "agvs.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['agv_id', 'init_x', 'init_y', 'speed'])
        for i in range(num_agvs):
            w.writerow([i, 0.0, 50.0, 1.0])
    
    with open(os.path.join(instance_dir, "machines.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['machine_id', 'pos_x', 'pos_y', 'switch_time', 'buffer_capacity'])
        for i in range(num_machines):
            x = np.random.uniform(20, 80)
            y = np.random.uniform(10, 90)
            sw = np.random.uniform(2, 5)
            w.writerow([i, f"{x:.2f}", f"{y:.2f}", f"{sw:.2f}", 2])
    
    with open(os.path.join(instance_dir, "process_times.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        header = ['machine_id'] + [f'type_{j}' for j in range(num_types)]
        w.writerow(header)
        for i in range(num_machines):
            row = [i] + [f"{np.random.uniform(5, 20):.2f}" for _ in range(num_types)]
            w.writerow(row)
    
    print(f"  {name}: Types={num_types}, Orders={num_orders}, AGVs={num_agvs}, Machines={num_machines}")


def generate_instance_set(output_dir: str, seed: int = 42):
    """生成标准算例集"""
    configs = [
        ("small_1", 2, 3, 3, 2, 3),
        ("small_2", 3, 4, 4, 2, 4),
        ("small_3", 3, 5, 5, 3, 4),
        ("medium_1", 4, 8, 6, 3, 5),
        ("medium_2", 5, 10, 8, 4, 6),
        ("medium_3", 5, 12, 10, 5, 8),
        ("large_1", 6, 15, 12, 6, 10),
        ("large_2", 8, 20, 15, 8, 12),
        ("large_3", 10, 25, 20, 10, 15),
    ]
    
    print(f"Generating {len(configs)} instances...")
    os.makedirs(output_dir, exist_ok=True)
    
    for name, nt, no, aw, na, nm in configs:
        generate_instance(output_dir, name, nt, no, aw, na, nm, seed + hash(name) % 10000)
    
    print(f"All instances saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='instances')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--generate_set', action='store_true')
    args = parser.parse_args()
    generate_instance_set(args.output_dir, args.seed)
