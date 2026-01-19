"""
rl_agent.py - PPO智能体（分离的策略网络和值网络）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple

from config import config
from graph_state import GraphData
from policy import PolicyNetworkOptimized
from value import ValueNetwork
from utils import compute_gae


class PPOAgent:
    """PPO智能体（分离策略网络和值网络）"""
    
    def __init__(self, order_dim: int, agv_dim: int, waiting_dim: int, buffer_dim: int,
                 num_orders: int, num_machines: int, 
                 policy_lr: float = config.LEARNING_RATE,
                 value_lr: float = config.LEARNING_RATE):
        
        self.device = config.DEVICE
        self.gamma = config.GAMMA
        self.gae_lambda = config.GAE_LAMBDA
        self.clip_epsilon = config.CLIP_EPSILON
        self.entropy_coef = config.ENTROPY_COEF
        self.max_grad_norm = config.MAX_GRAD_NORM
        self.num_epochs = config.NUM_EPOCHS
        
        # 分离的策略网络和值网络
        self.policy_net = PolicyNetworkOptimized(
            order_dim, agv_dim, waiting_dim, buffer_dim,
            num_orders, num_machines
        ).to(self.device)
        
        self.value_net = ValueNetwork(
            order_dim, agv_dim, waiting_dim, buffer_dim
        ).to(self.device)
        
        # 独立的优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        
        # 独立的学习率调度器
        self.policy_scheduler = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=1000, gamma=0.95)
        self.value_scheduler = optim.lr_scheduler.StepLR(self.value_optimizer, step_size=1000, gamma=0.95)
        
        self.episode_data = []
        self.stats = {'policy_loss': [], 'value_loss': [], 'entropy': []}
    
    def select_action(self, graph_data: GraphData) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """选择动作"""
        self.policy_net.eval()
        self.value_net.eval()
        
        with torch.no_grad():
            logits, _ = self.policy_net(graph_data)
            value, _ = self.value_net(graph_data)
            
            # 处理全为-inf的情况
            if torch.all(logits == float('-inf')):
                return -1, torch.tensor(0.0), value
            
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value
    
    def store_transition(self, graph_data: GraphData, action: int, reward: float,
                         done: bool, log_prob: torch.Tensor, value: torch.Tensor):
        """存储转移"""
        self.episode_data.append({
            'graph_data': graph_data,
            'action': action,
            'reward': reward,
            'done': done,
            'log_prob': log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob,
            'value': value.item() if isinstance(value, torch.Tensor) else value
        })
    
    def update(self) -> Dict[str, float]:
        """更新策略和值网络（分开更新）"""
        if not self.episode_data:
            return {}
        
        self.policy_net.train()
        self.value_net.train()
        
        n = len(self.episode_data)
        rewards = np.array([d['reward'] for d in self.episode_data])
        values = np.array([d['value'] for d in self.episode_data])
        dones = np.array([d['done'] for d in self.episode_data])
        
        advantages, returns = compute_gae(rewards, values, dones, self.gamma, self.gae_lambda)
        
        actions = torch.tensor([d['action'] for d in self.episode_data], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([d['log_prob'] for d in self.episode_data], dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # 标准化优势
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        total_policy_loss, total_value_loss, total_entropy = 0.0, 0.0, 0.0
        
        for _ in range(self.num_epochs):
            for i in range(n):
                graph_data = self.episode_data[i]['graph_data']
                
                # ========== 更新策略网络 ==========
                logits, _ = self.policy_net(graph_data)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                new_log_prob = dist.log_prob(actions[i])
                entropy = dist.entropy()
                
                ratio = torch.exp(new_log_prob - old_log_probs[i])
                surr1 = ratio * advantages_tensor[i]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor[i]
                
                policy_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                # ========== 更新值网络 ==========
                value, _ = self.value_net(graph_data)
                value_loss = nn.functional.mse_loss(value, returns_tensor[i])
                
                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        self.policy_scheduler.step()
        self.value_scheduler.step()
        self.episode_data = []
        
        num_updates = n * self.num_epochs
        stats = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
        
        for k, v in stats.items():
            self.stats[k].append(v)
        
        return stats
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'stats': self.stats
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy_net'])
        self.value_net.load_state_dict(ckpt['value_net'])
        self.policy_optimizer.load_state_dict(ckpt['policy_optimizer'])
        self.value_optimizer.load_state_dict(ckpt['value_optimizer'])
        if 'stats' in ckpt:
            self.stats = ckpt['stats']
    
    def get_lr(self) -> Tuple[float, float]:
        """获取学习率"""
        return (self.policy_optimizer.param_groups[0]['lr'],
                self.value_optimizer.param_groups[0]['lr'])
