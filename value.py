"""
value.py - 值网络（独立的HGNN编码器）
"""

import torch
import torch.nn as nn
from hgnn import HGNNEncoder
from config import config


class ValueNetwork(nn.Module):
    """值网络"""
    
    def __init__(self, order_dim: int, agv_dim: int, waiting_dim: int, buffer_dim: int):
        super().__init__()
        
        # 独立的HGNN编码器
        self.encoder = HGNNEncoder(order_dim, agv_dim, waiting_dim, buffer_dim)
        embed_dim = self.encoder.embed_dim
        
        # 值函数头
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, graph_data):
        """前向传播，返回状态值"""
        embeddings = self.encoder(graph_data)
        global_state = embeddings['global']
        value = self.value_head(global_state)
        return value.squeeze(-1), embeddings
