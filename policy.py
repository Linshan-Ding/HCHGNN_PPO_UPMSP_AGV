"""
policy.py - 策略网络（独立的HGNN编码器）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from hgnn import HGNNEncoder
from config import config


class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, order_dim: int, agv_dim: int, waiting_dim: int, buffer_dim: int,
                 num_orders: int, num_machines: int):
        super().__init__()
        
        self.num_orders = num_orders
        self.num_machines = num_machines
        
        # 独立的HGNN编码器
        self.encoder = HGNNEncoder(order_dim, agv_dim, waiting_dim, buffer_dim)
        embed_dim = self.encoder.embed_dim
        
        # 派发任务评分网络 (agv + order + waiting -> score)
        self.dispatch_scorer = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        
        # 取回任务评分网络 (agv + buffer -> score)
        self.pickup_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
    
    def forward(self, graph_data):
        """前向传播，返回动作logits"""
        embeddings = self.encoder(graph_data)
        
        h_order = embeddings['order']
        h_agv = embeddings['agv']
        h_waiting = embeddings['waiting']
        h_buffer = embeddings['buffer']
        
        num_agvs = h_agv.size(0)
        num_dispatch = self.num_orders * self.num_machines
        num_pickup = self.num_machines
        
        logits = []
        
        for ai in range(num_agvs):
            agv_emb = h_agv[ai]
            
            # 派发任务评分
            for oi in range(self.num_orders):
                for mi in range(self.num_machines):
                    combined = torch.cat([agv_emb, h_order[oi], h_waiting[mi]])
                    score = self.dispatch_scorer(combined)
                    logits.append(score)
            
            # 取回任务评分
            for mi in range(self.num_machines):
                combined = torch.cat([agv_emb, h_buffer[mi]])
                score = self.pickup_scorer(combined)
                logits.append(score)
        
        logits = torch.cat(logits).view(-1)
        
        # 应用动作掩码
        mask = graph_data.action_mask
        logits = logits.masked_fill(mask == 0, float('-inf'))
        
        return logits, embeddings


class PolicyNetworkOptimized(nn.Module):
    """优化版策略网络（批量计算）"""
    
    def __init__(self, order_dim: int, agv_dim: int, waiting_dim: int, buffer_dim: int,
                 num_orders: int, num_machines: int):
        super().__init__()
        
        self.num_orders = num_orders
        self.num_machines = num_machines
        
        self.encoder = HGNNEncoder(order_dim, agv_dim, waiting_dim, buffer_dim)
        embed_dim = self.encoder.embed_dim
        
        self.dispatch_scorer = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        
        self.pickup_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
    
    def forward(self, graph_data):
        embeddings = self.encoder(graph_data)
        
        h_order = embeddings['order']
        h_agv = embeddings['agv']
        h_waiting = embeddings['waiting']
        h_buffer = embeddings['buffer']
        
        num_agvs = h_agv.size(0)
        num_dispatch = self.num_orders * self.num_machines
        
        all_logits = []
        
        for ai in range(num_agvs):
            agv_emb = h_agv[ai:ai+1].expand(self.num_orders * self.num_machines, -1)
            
            # 构建所有dispatch组合
            order_idx = torch.arange(self.num_orders, device=h_order.device).repeat_interleave(self.num_machines)
            machine_idx = torch.arange(self.num_machines, device=h_order.device).repeat(self.num_orders)
            
            dispatch_input = torch.cat([agv_emb, h_order[order_idx], h_waiting[machine_idx]], dim=1)
            dispatch_scores = self.dispatch_scorer(dispatch_input).squeeze(-1)
            
            # Pickup评分
            agv_emb_pickup = h_agv[ai:ai+1].expand(self.num_machines, -1)
            pickup_input = torch.cat([agv_emb_pickup, h_buffer], dim=1)
            pickup_scores = self.pickup_scorer(pickup_input).squeeze(-1)
            
            all_logits.append(torch.cat([dispatch_scores, pickup_scores]))
        
        logits = torch.cat(all_logits)
        
        mask = graph_data.action_mask
        logits = logits.masked_fill(mask == 0, float('-inf'))
        
        return logits, embeddings
