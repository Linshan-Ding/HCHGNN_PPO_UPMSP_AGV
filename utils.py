"""
utils.py - 工具函数
"""

import numpy as np
import torch
import random


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def euclidean_distance(pos1, pos2) -> float:
    """计算欧氏距离"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def travel_time(pos1, pos2, speed: float) -> float:
    """计算行驶时间"""
    return euclidean_distance(pos1, pos2) / speed


def one_hot(index: int, num_classes: int) -> np.ndarray:
    """One-hot编码"""
    vec = np.zeros(num_classes, dtype=np.float32)
    if 0 <= index < num_classes:
        vec[index] = 1.0
    return vec


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """计算GAE优势和回报"""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)
    
    last_gae = 0
    for t in reversed(range(n)):
        if dones[t]:
            delta = rewards[t] - values[t]
            last_gae = delta
        else:
            next_value = values[t + 1] if t + 1 < n else 0
            delta = rewards[t] + gamma * next_value - values[t]
            last_gae = delta + gamma * gae_lambda * last_gae
        
        advantages[t] = last_gae
        returns[t] = advantages[t] + values[t]
    
    return advantages, returns
