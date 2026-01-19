"""
config.py - 配置参数
"""

import torch


class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # HGNN参数
    EMBED_DIM = 64
    HIDDEN_DIM = 64
    NUM_GAT_LAYERS = 2
    NUM_GAT_HEADS = 4
    
    # PPO参数
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    ENTROPY_COEF = 0.01
    VALUE_LOSS_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    NUM_EPOCHS = 4
    
    # 训练参数
    NUM_EPISODES = 5000
    LOG_INTERVAL = 100
    SAVE_INTERVAL = 500
    SEED = 42


config = Config()
