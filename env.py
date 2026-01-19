"""
env.py - 事件驱动仿真环境
- AGV卸载后立即空闲，可停在任意位置
- 细粒度事件：到达等待区、开始加工、加工完成、到达缓存区装载、到达仓库卸载
- 即时奖励：上一决策时刻最小空闲时刻 - 当前决策时刻最小空闲时刻
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import heapq

from config import config
from graph_state import (
    Order, AGV, WaitingArea, BufferArea, Machine,
    StateBuilder, GraphData, create_entities_from_instance
)
from utils import euclidean_distance, travel_time


@dataclass
class GanttTask:
    """甘特图任务"""
    entity_type: str  # "machine" or "agv"
    entity_id: int
    task_type: str  # machine: process/switch, agv: move/wait
    start_time: float
    end_time: float
    workpiece_type: Optional[int] = None
    order_id: Optional[int] = None
    from_loc: Optional[str] = None  # AGV移动起点
    to_loc: Optional[str] = None    # AGV移动终点


@dataclass
class Event:
    """事件"""
    time: float
    event_type: str
    data: Dict
    
    def __lt__(self, other):
        return self.time < other.time


class SchedulingEnv:
    """调度环境"""
    
    def __init__(self, instance_data):
        self.instance_data = instance_data
        self.num_orders = instance_data.num_orders
        self.num_agvs = instance_data.num_agvs
        self.num_machines = instance_data.num_machines
        self.warehouse = instance_data.warehouse_position
        self.dispatch_point = instance_data.order_dispatch_point
        
        self.orders: List[Order] = []
        self.agvs: List[AGV] = []
        self.waitings: List[WaitingArea] = []
        self.buffers: List[BufferArea] = []
        self.machines: List[Machine] = []
        
        self.current_time = 0.0
        self.event_queue: List[Event] = []
        self.total_workpieces = 0
        self.completed_workpieces = 0
        self.makespan = 0.0
        self.is_deadlock = False
        
        self.prev_min_idle_time = 0.0  # 上一决策时刻的最小空闲时刻
        
        self.gantt_data: List[GanttTask] = []
        self.state_builder: StateBuilder = None
        self.action_space_size = 0
        
        self.reset()
    
    def reset(self) -> Tuple[GraphData, np.ndarray]:
        """重置环境"""
        self.orders, self.agvs, self.waitings, self.buffers, self.machines = \
            create_entities_from_instance(self.instance_data)
        
        self.total_workpieces = sum(o.total_workpieces for o in self.orders)
        self.completed_workpieces = 0
        self.current_time = 0.0
        self.event_queue = []
        self.makespan = 0.0
        self.gantt_data = []
        self.is_deadlock = False
        self.prev_min_idle_time = 0.0
        
        self.state_builder = StateBuilder(
            self.orders, self.agvs, self.waitings, self.buffers, self.machines)
        self.action_space_size = self.state_builder.get_action_space_size()
        
        graph = self.state_builder.build_graph()
        return graph, graph.action_mask.cpu().numpy()
    
    def _get_min_next_idle_time(self) -> float:
        """获取所有机器和AGV的下一空闲时刻的最小值（需大于当前时刻）"""
        idle_times = []
        
        for agv in self.agvs:
            t = agv.busy_until if agv.busy_until > self.current_time else self.current_time
            idle_times.append(t)
        
        for machine in self.machines:
            t = machine.busy_until if machine.busy_until > self.current_time else self.current_time
            idle_times.append(t)
        
        return min(idle_times) if idle_times else self.current_time
    
    def step(self, action: int) -> Tuple[GraphData, float, bool, Dict]:
        """执行动作"""
        graph = self.state_builder.build_graph()
        mask = graph.action_mask.cpu().numpy()
        
        if mask[action] == 0:
            return graph, -10.0, False, {'makespan': self.makespan}
        
        # 保存上一决策时刻的最小空闲时刻
        self.prev_min_idle_time = self._get_min_next_idle_time()
        
        agv_id, action_type, order_id, machine_id = self.state_builder.decode_action(action)
        
        if action_type == "dispatch":
            self._dispatch_to_waiting(agv_id, order_id, machine_id)
        else:
            self._pickup_from_buffer(agv_id, machine_id)
        
        self._process_events_until_decision_or_done()
        
        self.state_builder = StateBuilder(
            self.orders, self.agvs, self.waitings, self.buffers, self.machines)
        new_graph = self.state_builder.build_graph()
        
        done = self._check_done()
        reward = self._compute_reward(done)
        
        return new_graph, reward, done, {'makespan': self.makespan, 'deadlock': self.is_deadlock}
    
    def _dispatch_to_waiting(self, agv_id: int, order_id: int, machine_id: int):
        """派发工件到机器等待区"""
        agv = self.agvs[agv_id]
        order = self.orders[order_id]
        waiting = self.waitings[machine_id]
        
        order.remaining_workpieces -= 1
        
        # AGV状态：前往等待区
        agv.state = 1  # ToWaiting
        agv.carrying_type = order.workpiece_type
        agv.carrying_order_id = order_id
        agv.target_machine_id = machine_id
        
        # 计算时间
        time_to_order = travel_time(agv.position, order.position, agv.speed)
        time_to_machine = travel_time(order.position, waiting.position, agv.speed)
        arrival_time = self.current_time + time_to_order + time_to_machine
        agv.busy_until = arrival_time
        
        # 记录AGV移动（派发点->订单->等待区）
        if time_to_order > 0:
            self.gantt_data.append(GanttTask(
                "agv", agv_id, "move", self.current_time, self.current_time + time_to_order,
                order.workpiece_type, order_id, "current", "order"))
        self.gantt_data.append(GanttTask(
            "agv", agv_id, "move", self.current_time + time_to_order, arrival_time,
            order.workpiece_type, order_id, "order", f"waiting_{machine_id}"))
        
        heapq.heappush(self.event_queue, Event(arrival_time, "arrive_waiting", {
            'agv_id': agv_id, 'machine_id': machine_id,
            'workpiece_type': order.workpiece_type, 'order_id': order_id
        }))
    
    def _pickup_from_buffer(self, agv_id: int, machine_id: int):
        """从缓存区取回工件"""
        agv = self.agvs[agv_id]
        buffer = self.buffers[machine_id]
        
        # 预定工件
        buffer.finished_count -= 1
        
        agv.state = 3  # ToBuffer
        agv.target_machine_id = machine_id
        
        # 计算到达缓存区时间
        time_to_buffer = travel_time(agv.position, buffer.position, agv.speed)
        arrive_buffer_time = self.current_time + time_to_buffer
        
        if time_to_buffer > 0:
            self.gantt_data.append(GanttTask(
                "agv", agv_id, "move", self.current_time, arrive_buffer_time,
                from_loc="current", to_loc=f"buffer_{machine_id}"))
        
        heapq.heappush(self.event_queue, Event(arrive_buffer_time, "arrive_buffer", {
            'agv_id': agv_id, 'machine_id': machine_id
        }))
        
        # 取走后检查能否开始加工
        self._try_start_processing(machine_id)
    
    def _process_events_until_decision_or_done(self):
        """处理事件直到有有效决策点或完成"""
        max_iterations = 10000
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            if self.completed_workpieces >= self.total_workpieces:
                break
            
            if self._has_valid_decision():
                break
            
            if not self.event_queue:
                if self._is_deadlock():
                    self.is_deadlock = True
                    if self.makespan == 0:
                        self.makespan = self.current_time + 1000.0
                break
            
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            
            if event.event_type == "arrive_waiting":
                self._handle_arrive_waiting(event.data)
            elif event.event_type == "start_process":
                self._handle_start_process(event.data)
            elif event.event_type == "process_complete":
                self._handle_process_complete(event.data)
            elif event.event_type == "arrive_buffer":
                self._handle_arrive_buffer(event.data)
            elif event.event_type == "arrive_warehouse":
                self._handle_arrive_warehouse(event.data)
    
    def _is_deadlock(self) -> bool:
        if self.completed_workpieces >= self.total_workpieces:
            return False
        if self.event_queue:
            return False
        for agv in self.agvs:
            if agv.is_idle():
                for o in self.orders:
                    if o.remaining_workpieces > 0:
                        return False
                for b in self.buffers:
                    if b.has_finished():
                        return False
        for agv in self.agvs:
            if agv.state in [3, 4]:
                return False
        for machine in self.machines:
            if machine.is_processing:
                return False
        return True
    
    def _has_valid_decision(self) -> bool:
        for a in self.agvs:
            if a.is_idle():
                for o in self.orders:
                    if o.remaining_workpieces > 0:
                        return True
                for b in self.buffers:
                    if b.has_finished():
                        return True
        return False
    
    def _handle_arrive_waiting(self, data: Dict):
        """AGV到达等待区"""
        agv_id = data['agv_id']
        machine_id = data['machine_id']
        
        agv = self.agvs[agv_id]
        waiting = self.waitings[machine_id]
        
        agv.position = waiting.position
        agv.state = 2  # Waiting
        waiting.queue.append(agv_id)
        
        # 记录等待开始
        self.gantt_data.append(GanttTask(
            "agv", agv_id, "wait", self.current_time, self.current_time,
            data['workpiece_type'], data['order_id'],
            f"waiting_{machine_id}", f"waiting_{machine_id}"))
        
        self._try_start_processing(machine_id)
    
    def _try_start_processing(self, machine_id: int):
        """尝试开始加工"""
        machine = self.machines[machine_id]
        waiting = self.waitings[machine_id]
        buffer = self.buffers[machine_id]
        
        if not machine.can_start(buffer, waiting):
            return
        
        agv_id = waiting.queue.pop(0)
        agv = self.agvs[agv_id]
        wtype = agv.carrying_type
        order_id = agv.carrying_order_id
        
        # 更新等待任务结束时间
        for task in reversed(self.gantt_data):
            if task.entity_type == "agv" and task.entity_id == agv_id and task.task_type == "wait":
                task.end_time = self.current_time
                break
        
        # AGV卸载，立即变为空闲（不返回派发点）
        agv.state = 0
        agv.carrying_type = None
        agv.carrying_order_id = None
        agv.target_machine_id = None
        agv.busy_until = self.current_time
        # agv.position 保持在等待区位置
        
        # 计算加工时间
        proc_time = machine.process_times.get(wtype, 15.0)
        switch_time = 0.0
        if machine.last_type is not None and machine.last_type != wtype:
            switch_time = machine.switch_time
            self.gantt_data.append(GanttTask(
                "machine", machine_id, "switch",
                self.current_time, self.current_time + switch_time, wtype))
        
        machine.is_processing = True
        machine.current_type = wtype
        machine.last_type = wtype
        process_end_time = self.current_time + switch_time + proc_time
        machine.busy_until = process_end_time
        
        self.gantt_data.append(GanttTask(
            "machine", machine_id, "process",
            self.current_time + switch_time, process_end_time, wtype, order_id))
        
        heapq.heappush(self.event_queue, Event(process_end_time, "process_complete", {
            'machine_id': machine_id, 'workpiece_type': wtype, 'order_id': order_id
        }))
    
    def _handle_process_complete(self, data: Dict):
        """加工完成"""
        machine_id = data['machine_id']
        machine = self.machines[machine_id]
        buffer = self.buffers[machine_id]
        
        machine.is_processing = False
        machine.current_type = None
        buffer.finished_count += 1
        machine.busy_until = self.current_time
        
        self.makespan = max(self.makespan, self.current_time)
        self._try_start_processing(machine_id)
    
    def _handle_arrive_buffer(self, data: Dict):
        """AGV到达缓存区装载工件"""
        agv_id = data['agv_id']
        machine_id = data['machine_id']
        
        agv = self.agvs[agv_id]
        buffer = self.buffers[machine_id]
        
        agv.position = buffer.position
        agv.state = 4  # ToWarehouse
        
        # 计算到仓库时间
        time_to_warehouse = travel_time(buffer.position, self.warehouse, agv.speed)
        arrival_time = self.current_time + time_to_warehouse
        agv.busy_until = arrival_time
        
        self.gantt_data.append(GanttTask(
            "agv", agv_id, "move", self.current_time, arrival_time,
            from_loc=f"buffer_{machine_id}", to_loc="warehouse"))
        
        heapq.heappush(self.event_queue, Event(arrival_time, "arrive_warehouse", {
            'agv_id': agv_id
        }))
        
        # 检查能否继续加工
        self._try_start_processing(machine_id)
    
    def _handle_arrive_warehouse(self, data: Dict):
        """AGV到达仓库卸载工件"""
        agv_id = data['agv_id']
        agv = self.agvs[agv_id]
        
        agv.position = self.warehouse
        self.completed_workpieces += 1
        self.makespan = max(self.makespan, self.current_time)
        
        # AGV立即空闲，停在仓库位置
        agv.state = 0
        agv.carrying_type = None
        agv.target_machine_id = None
        agv.busy_until = self.current_time
    
    def _check_done(self) -> bool:
        if self.completed_workpieces >= self.total_workpieces:
            return True
        if self.is_deadlock:
            return True
        return False
    
    def _compute_reward(self, done: bool) -> float:
        """计算奖励: 上一决策时刻最小空闲时刻 - 当前决策时刻最小空闲时刻"""
        if self.is_deadlock:
            return -100.0
        
        current_min_idle = self._get_min_next_idle_time()
        reward = self.prev_min_idle_time - current_min_idle
        
        if done and not self.is_deadlock:
            reward += 10.0
        
        return reward
    
    def get_gantt_data(self) -> List[GanttTask]:
        return self.gantt_data
