"""
hgnn.py - 异质图神经网络（带特征标准化）
边特征：Order-AGV(距离), AGV-Waiting(距离+切换时间), AGV-Buffer(距离)
删除了Waiting-Buffer边
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


class FeatureStandardizer(nn.Module):
    """特征标准化层（对同类节点的同类特征进行标准化）"""
    
    def __init__(self, feat_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.feat_dim = feat_dim
        self.gamma = nn.Parameter(torch.ones(feat_dim))
        self.beta = nn.Parameter(torch.zeros(feat_dim))
    
    def forward(self, x):
        if x.size(0) == 1:
            return x * self.gamma + self.beta
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + self.eps
        x_norm = (x - mean) / std
        return x_norm * self.gamma + self.beta


class EdgeGATLayer(nn.Module):
    """带边特征的GAT层"""
    
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        self.W_src = nn.Linear(in_dim, out_dim, bias=False)
        self.W_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.W_edge = nn.Linear(edge_dim, out_dim, bias=False)
        
        self.a_src = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.a_edge = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        nn.init.xavier_uniform_(self.a_edge)
    
    def forward(self, h_src, h_dst, edge_index, edge_feat):
        num_dst = h_dst.size(0)
        
        h_src_t = self.W_src(h_src).view(-1, self.num_heads, self.head_dim)
        h_dst_t = self.W_dst(h_dst).view(-1, self.num_heads, self.head_dim)
        h_edge_t = self.W_edge(edge_feat).view(-1, self.num_heads, self.head_dim)
        
        src_idx, dst_idx = edge_index[0], edge_index[1]
        
        e_src = (h_src_t[src_idx] * self.a_src).sum(-1)
        e_dst = (h_dst_t[dst_idx] * self.a_dst).sum(-1)
        e_edge = (h_edge_t * self.a_edge).sum(-1)
        e = F.leaky_relu(e_src + e_dst + e_edge, 0.2)
        
        e_max = torch.zeros(num_dst, self.num_heads, device=e.device)
        e_max.scatter_reduce_(0, dst_idx.unsqueeze(-1).expand(-1, self.num_heads), e, reduce='amax', include_self=False)
        e_exp = torch.exp(e - e_max[dst_idx])
        
        e_sum = torch.zeros(num_dst, self.num_heads, device=e.device)
        e_sum.scatter_add_(0, dst_idx.unsqueeze(-1).expand(-1, self.num_heads), e_exp)
        alpha = e_exp / (e_sum[dst_idx] + 1e-8)
        
        msg = alpha.unsqueeze(-1) * h_src_t[src_idx]
        out = torch.zeros(num_dst, self.num_heads, self.head_dim, device=msg.device)
        out.scatter_add_(0, dst_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_heads, self.head_dim), msg)
        
        return out.view(num_dst, -1)


class HeterogeneousGNN(nn.Module):
    """异质图神经网络（带标准化）
    边类型：Order-AGV, AGV-Waiting, AGV-Buffer
    """
    
    def __init__(self, order_dim: int, agv_dim: int, waiting_dim: int, buffer_dim: int,
                 embed_dim: int = 64, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # 特征标准化层
        self.order_standardizer = FeatureStandardizer(order_dim)
        self.agv_standardizer = FeatureStandardizer(agv_dim)
        self.waiting_standardizer = FeatureStandardizer(waiting_dim)
        self.buffer_standardizer = FeatureStandardizer(buffer_dim)
        
        # 边特征维度（更新后）
        edge_dim_oa = 1  # Order-AGV: 距离
        edge_dim_aw = 2  # AGV-Waiting: 距离 + 切换时间
        edge_dim_ab = 1  # AGV-Buffer: 距离
        
        # 边特征标准化
        self.edge_oa_standardizer = FeatureStandardizer(edge_dim_oa)
        self.edge_aw_standardizer = FeatureStandardizer(edge_dim_aw)
        self.edge_ab_standardizer = FeatureStandardizer(edge_dim_ab)
        
        # 节点嵌入层
        self.order_embed = nn.Linear(order_dim, embed_dim)
        self.agv_embed = nn.Linear(agv_dim, embed_dim)
        self.waiting_embed = nn.Linear(waiting_dim, embed_dim)
        self.buffer_embed = nn.Linear(buffer_dim, embed_dim)
        
        # 消息传递层（6种边类型：3种边的双向）
        self.o2a_layers = nn.ModuleList([EdgeGATLayer(embed_dim, embed_dim, edge_dim_oa, num_heads) for _ in range(num_layers)])
        self.a2o_layers = nn.ModuleList([EdgeGATLayer(embed_dim, embed_dim, edge_dim_oa, num_heads) for _ in range(num_layers)])
        self.a2w_layers = nn.ModuleList([EdgeGATLayer(embed_dim, embed_dim, edge_dim_aw, num_heads) for _ in range(num_layers)])
        self.w2a_layers = nn.ModuleList([EdgeGATLayer(embed_dim, embed_dim, edge_dim_aw, num_heads) for _ in range(num_layers)])
        self.a2b_layers = nn.ModuleList([EdgeGATLayer(embed_dim, embed_dim, edge_dim_ab, num_heads) for _ in range(num_layers)])
        self.b2a_layers = nn.ModuleList([EdgeGATLayer(embed_dim, embed_dim, edge_dim_ab, num_heads) for _ in range(num_layers)])
        
        # LayerNorm
        self.order_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.agv_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.waiting_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.buffer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        # 全局池化
        self.global_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, graph_data):
        # 标准化节点特征
        order_feat = self.order_standardizer(graph_data.order_features)
        agv_feat = self.agv_standardizer(graph_data.agv_features)
        waiting_feat = self.waiting_standardizer(graph_data.waiting_features)
        buffer_feat = self.buffer_standardizer(graph_data.buffer_features)
        
        # 标准化边特征
        edge_oa = self.edge_oa_standardizer(graph_data.order_agv_edge_feat)
        edge_aw = self.edge_aw_standardizer(graph_data.agv_waiting_edge_feat)
        edge_ab = self.edge_ab_standardizer(graph_data.agv_buffer_edge_feat)
        
        # 初始嵌入
        h_order = self.order_embed(order_feat)
        h_agv = self.agv_embed(agv_feat)
        h_waiting = self.waiting_embed(waiting_feat)
        h_buffer = self.buffer_embed(buffer_feat)
        
        for i in range(len(self.o2a_layers)):
            # Order <-> AGV
            h_agv_new = self.o2a_layers[i](h_order, h_agv, graph_data.order_to_agv_edge_index, edge_oa)
            h_order_new = self.a2o_layers[i](h_agv, h_order, graph_data.agv_to_order_edge_index, edge_oa)
            
            # AGV <-> Waiting
            h_waiting_new = self.a2w_layers[i](h_agv, h_waiting, graph_data.agv_to_waiting_edge_index, edge_aw)
            h_agv_new2 = self.w2a_layers[i](h_waiting, h_agv, graph_data.waiting_to_agv_edge_index, edge_aw)
            
            # AGV <-> Buffer
            h_buffer_new = self.a2b_layers[i](h_agv, h_buffer, graph_data.agv_to_buffer_edge_index, edge_ab)
            h_agv_new3 = self.b2a_layers[i](h_buffer, h_agv, graph_data.buffer_to_agv_edge_index, edge_ab)
            
            # 残差连接 + LayerNorm
            h_order = self.order_norms[i](h_order + h_order_new)
            h_agv = self.agv_norms[i](h_agv + h_agv_new + h_agv_new2 + h_agv_new3)
            h_waiting = self.waiting_norms[i](h_waiting + h_waiting_new)
            h_buffer = self.buffer_norms[i](h_buffer + h_buffer_new)
        
        # 全局状态
        global_feat = torch.cat([h_order.mean(0), h_agv.mean(0), h_waiting.mean(0), h_buffer.mean(0)])
        global_state = self.global_mlp(global_feat)
        
        return {'order': h_order, 'agv': h_agv, 'waiting': h_waiting, 'buffer': h_buffer, 'global': global_state}


class HGNNEncoder(nn.Module):
    """HGNN编码器（独立实例，用于策略网络或值网络）"""
    
    def __init__(self, order_dim: int, agv_dim: int, waiting_dim: int, buffer_dim: int):
        super().__init__()
        self.hgnn = HeterogeneousGNN(
            order_dim, agv_dim, waiting_dim, buffer_dim,
            config.EMBED_DIM, config.NUM_GAT_LAYERS, config.NUM_GAT_HEADS)
        self.embed_dim = config.EMBED_DIM
    
    def forward(self, graph_data):
        return self.hgnn(graph_data)
