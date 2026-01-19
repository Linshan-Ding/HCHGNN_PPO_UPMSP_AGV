"""
visualizer.py - Visdom可视化 + 甘特图
甘特图：机器行显示加工/切换，AGV行显示移动/等待
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict
from collections import deque


class VisdomVisualizer:
    """Visdom可视化器"""
    
    def __init__(self, env_name: str = "scheduling", server: str = "http://localhost", port: int = 8097):
        self.env_name = env_name
        self.enabled = False
        self.viz = None
        
        try:
            import visdom
            self.viz = visdom.Visdom(server=server, port=port, env=env_name)
            if self.viz.check_connection():
                self.enabled = True
                self.viz.close(env=env_name)
                print(f"[Visdom] Connected: {server}:{port}, env: {env_name}")
            else:
                print("[Warning] Visdom not running. Start: python -m visdom.server")
        except ImportError:
            print("[Warning] visdom not installed. Run: pip install visdom")
        except Exception as e:
            print(f"[Warning] Visdom error: {e}")
        
        self.episodes = []
        self.makespans = []
        self.best_makespan = float('inf')
        self.makespan_window = deque(maxlen=100)
        self.windows = {}
    
    def log_episode(self, episode: int, makespan: float, reward: float):
        self.episodes.append(episode)
        self.makespans.append(makespan)
        self.makespan_window.append(makespan)
        
        if makespan < self.best_makespan:
            self.best_makespan = makespan
        
        if not self.enabled:
            return
        
        avg = np.mean(list(self.makespan_window))
        
        if 'makespan' not in self.windows:
            self.windows['makespan'] = self.viz.line(
                X=np.array([episode]), Y=np.array([makespan]),
                opts=dict(title='Makespan', xlabel='Episode', ylabel='Makespan',
                         legend=['Makespan', 'Moving Avg'], showlegend=True),
                name='Makespan')
        else:
            self.viz.line(X=np.array([episode]), Y=np.array([makespan]),
                         win=self.windows['makespan'], name='Makespan', update='append')
        
        if len(self.episodes) >= 2:
            self.viz.line(X=np.array([episode]), Y=np.array([avg]),
                         win=self.windows['makespan'], name='Moving Avg', update='append')
    
    def log_loss(self, update: int, policy_loss: float, value_loss: float, entropy: float):
        if not self.enabled:
            return
        
        # 策略损失窗口
        if 'policy_loss' not in self.windows:
            self.windows['policy_loss'] = self.viz.line(
                X=np.array([update]), Y=np.array([policy_loss]),
                opts=dict(title='Policy Loss', xlabel='Update', ylabel='Loss',
                         legend=['Policy Loss'], showlegend=True))
        else:
            self.viz.line(X=np.array([update]), Y=np.array([policy_loss]),
                         win=self.windows['policy_loss'], update='append')
        
        # 值损失窗口
        if 'value_loss' not in self.windows:
            self.windows['value_loss'] = self.viz.line(
                X=np.array([update]), Y=np.array([value_loss]),
                opts=dict(title='Value Loss', xlabel='Update', ylabel='Loss',
                         legend=['Value Loss'], showlegend=True))
        else:
            self.viz.line(X=np.array([update]), Y=np.array([value_loss]),
                         win=self.windows['value_loss'], update='append')
        
        # 熵窗口
        if 'entropy' not in self.windows:
            self.windows['entropy'] = self.viz.line(
                X=np.array([update]), Y=np.array([entropy]),
                opts=dict(title='Entropy', xlabel='Update', ylabel='Entropy',
                         legend=['Entropy'], showlegend=True))
        else:
            self.viz.line(X=np.array([update]), Y=np.array([entropy]),
                         win=self.windows['entropy'], update='append')
        
        # 合并损失窗口（方便对比）
        if 'all_loss' not in self.windows:
            self.windows['all_loss'] = self.viz.line(
                X=np.array([[update, update]]),
                Y=np.array([[policy_loss, value_loss]]),
                opts=dict(title='Policy vs Value Loss', xlabel='Update', ylabel='Loss',
                         legend=['Policy', 'Value'], showlegend=True))
        else:
            self.viz.line(X=np.array([[update, update]]),
                         Y=np.array([[policy_loss, value_loss]]),
                         win=self.windows['all_loss'], update='append')
    
    def close(self):
        pass


def create_visualizer(use_visdom: bool = True, env_name: str = "scheduling", port: int = 8097):
    if use_visdom:
        return VisdomVisualizer(env_name, port=port)
    return VisdomVisualizer.__new__(VisdomVisualizer)


def plot_gantt_chart(gantt_data: List, num_machines: int, num_agvs: int,
                     save_path: str, makespan: float):
    """
    绘制甘特图
    - 上半部分：机器行（显示process/switch）
    - 下半部分：AGV行（显示move/wait）
    """
    # 颜色设置
    type_colors = plt.cm.Set3(np.linspace(0, 1, 12))
    move_color = '#4CAF50'    # 绿色-移动
    wait_color = '#FFC107'    # 黄色-等待
    switch_color = '#FF9800'  # 橙色-切换
    
    fig, ax = plt.subplots(figsize=(16, max(8, (num_machines + num_agvs) * 0.6)))
    
    y_labels = []
    y_pos = 0
    
    # ========== 机器部分 ==========
    for mid in range(num_machines):
        y_labels.append(f'Machine {mid}')
        for task in gantt_data:
            if task.entity_type == 'machine' and task.entity_id == mid:
                duration = task.end_time - task.start_time
                if duration < 0.01:
                    continue
                
                if task.task_type == 'process':
                    color = type_colors[task.workpiece_type % 12] if task.workpiece_type is not None else 'steelblue'
                    label = f'T{task.workpiece_type}' if task.workpiece_type is not None else 'P'
                elif task.task_type == 'switch':
                    color = switch_color
                    label = 'SW'
                else:
                    continue
                
                rect = mpatches.FancyBboxPatch(
                    (task.start_time, y_pos - 0.35), duration, 0.7,
                    boxstyle="round,pad=0.02", facecolor=color, edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
                
                if duration > makespan * 0.02:
                    ax.text(task.start_time + duration/2, y_pos, label,
                           ha='center', va='center', fontsize=8, fontweight='bold')
        y_pos += 1
    
    # 分隔线
    ax.axhline(y=y_pos - 0.5, color='black', linewidth=2, linestyle='--')
    
    # ========== AGV部分 ==========
    for aid in range(num_agvs):
        y_labels.append(f'AGV {aid}')
        for task in gantt_data:
            if task.entity_type == 'agv' and task.entity_id == aid:
                duration = task.end_time - task.start_time
                if duration < 0.01:
                    continue
                
                if task.task_type == 'move':
                    color = move_color
                    # 简化标签
                    from_str = task.from_loc if task.from_loc else ''
                    to_str = task.to_loc if task.to_loc else ''
                    if 'waiting' in to_str:
                        label = f'→W{to_str.split("_")[-1]}'
                    elif 'buffer' in to_str:
                        label = f'→B{to_str.split("_")[-1]}'
                    elif to_str == 'warehouse':
                        label = '→WH'
                    elif to_str == 'order':
                        label = '→O'
                    else:
                        label = '→'
                elif task.task_type == 'wait':
                    color = wait_color
                    label = 'Wait'
                else:
                    continue
                
                rect = mpatches.FancyBboxPatch(
                    (task.start_time, y_pos - 0.35), duration, 0.7,
                    boxstyle="round,pad=0.02", facecolor=color, edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
                
                if duration > makespan * 0.03:
                    ax.text(task.start_time + duration/2, y_pos, label,
                           ha='center', va='center', fontsize=7, fontweight='bold')
        y_pos += 1
    
    ax.set_xlim(0, makespan * 1.05)
    ax.set_ylim(-0.5, y_pos - 0.5)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_title(f'Gantt Chart (Makespan: {makespan:.2f})', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # 图例
    legend_elements = [
        mpatches.Patch(facecolor=type_colors[0], label='Process (by type)'),
        mpatches.Patch(facecolor=switch_color, label='Switch'),
        mpatches.Patch(facecolor=move_color, label='AGV Move'),
        mpatches.Patch(facecolor=wait_color, label='AGV Wait'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gantt chart saved: {save_path}")


def plot_training_curves(makespans: List, rewards: List, stats: Dict, save_path: str):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    window = min(100, len(makespans) // 10 + 1)
    
    ax = axes[0, 0]
    ax.plot(makespans, alpha=0.3)
    if len(makespans) >= window:
        ma = np.convolve(makespans, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(makespans)), ma, label=f'MA({window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Makespan')
    ax.set_title('Makespan')
    ax.legend()
    ax.grid(True)
    
    ax = axes[0, 1]
    ax.plot(rewards, alpha=0.3)
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), ma, label=f'MA({window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Reward')
    ax.legend()
    ax.grid(True)
    
    ax = axes[1, 0]
    if stats.get('policy_loss'):
        ax.plot(stats['policy_loss'], label='Policy')
        ax.plot(stats['value_loss'], label='Value')
        ax.set_xlabel('Update')
        ax.set_ylabel('Loss')
        ax.set_title('Loss')
        ax.legend()
        ax.grid(True)
    
    ax = axes[1, 1]
    if stats.get('entropy'):
        ax.plot(stats['entropy'])
        ax.set_xlabel('Update')
        ax.set_ylabel('Entropy')
        ax.set_title('Entropy')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved: {save_path}")
