# Multi-AGV Scheduling with HGNN-PPO

多AGV + 不相关并行机调度系统（异质图神经网络 + PPO强化学习）

## 系统模型

- **4类节点**: 订单(Order)、AGV、机器等待区(WaitingArea)、机器缓存区(BufferArea)
- **AGV状态**: Idle(0) / ToWaiting(1) / Waiting(2) / ToBuffer(3) / ToWarehouse(4)
- **机器等待区**: 存放等待加工的工件（由AGV携带排队，FIFO）
- **机器缓存区**: 存放已加工工件，容量=2
- **加工条件**: 机器空闲 + 缓存区未满 + 等待区有工件

## 图结构

### 节点特征
| 节点类型 | 特征 |
|---------|------|
| Order | 工件类型(one-hot) + 剩余数量 + 交期 + 位置(x,y) |
| AGV | 状态(one-hot) + 位置(x,y) + 速度 + 空闲时刻 + 携带类型(one-hot) |
| WaitingArea | 位置(x,y) + 队列长度 + 是否有等待 |
| BufferArea | 位置(x,y) + 完工数量 + 容量 + 是否有完工 + 是否有空位 |

### 边类型与特征
| 边类型 | 特征 |
|-------|------|
| Order ↔ AGV | 距离 |
| AGV ↔ WaitingArea | 距离 + 切换时间（AGV携带类型与机器上次加工类型不同时） |
| AGV ↔ BufferArea | 距离 |

> 注：删除了WaitingArea ↔ BufferArea边（两类节点已通过AGV间接相连）

## 核心特性

### 1. 特征标准化
- 节点和边特征使用**原始值**（不做归一化）
- 在HGNN中使用**可学习的标准化层**
- 对同类节点的同类特征进行标准化（均值=0，方差=1）

### 2. AGV行为
- AGV卸载工件后**立即变为空闲状态**
- AGV可**停在系统任意位置**等待任务分配
- 不需要返回订单派发点

### 3. 事件驱动仿真
| 事件类型 | 描述 |
|---------|------|
| `arrive_waiting` | AGV到达机器等待区，加入队列 |
| `start_process` | 机器开始加工（从队列取出AGV的工件） |
| `process_complete` | 机器加工完成，工件放入缓存区 |
| `arrive_buffer` | AGV到达缓存区装载完工工件 |
| `arrive_warehouse` | AGV到达仓库卸载工件 |

### 4. 即时奖励
```
reward = prev_min_idle_time - current_min_idle_time
```
- `prev_min_idle_time`: 上一决策时刻所有机器和AGV的下一空闲时刻最小值
- `current_min_idle_time`: 当前决策时刻的相应最小值
- 完成任务额外+10奖励

### 5. 防死锁机制
- 缓存区有工件时，优先执行**pickup**动作
- 禁止向缓存区已满的机器派发新工件

### 6. 分离网络架构
- **策略网络**和**值网络**各自拥有**独立的HGNN编码器**
- 各自拥有**独立的优化器**
- 消除两者损失值差异带来的训练震荡

## 甘特图

- **机器行**: 加工时间条（按工件类型着色）+ 切换时间条
- **AGV行**: 移动时间条（订单点/等待区/缓存区/仓库之间）+ 等待时间条

## 文件结构

```
scheduling_project/
├── config.py           # 超参数配置
├── env.py              # 事件驱动仿真环境
├── graph_state.py      # 异质图状态构建
├── hgnn.py             # 异质图神经网络（含标准化层）
├── policy.py           # 策略网络（独立HGNN编码器）
├── value.py            # 值网络（独立HGNN编码器）
├── rl_agent.py         # PPO智能体（分离优化器）
├── train.py            # 训练入口
├── generate.py         # 算例生成
├── instance_loader.py  # 算例加载
├── visualizer.py       # Visdom可视化 + 甘特图
└── requirements.txt    # 依赖
```

## 使用方法

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成算例
python generate.py --generate_set

# 3. 启动Visdom（可选）
python -m visdom.server

# 4. 训练
python train.py --instance_name medium_1 --num_episodes 5000

# 5. 查看结果
# - Visdom: http://localhost:8097
# - 输出目录: checkpoints/{instance}_{timestamp}/
```

## Visdom监控

训练过程中显示以下曲线：
- **Makespan**: 每个episode的完工时间
- **Policy Loss**: 策略网络损失
- **Value Loss**: 值网络损失  
- **Entropy**: 策略熵
- **Policy vs Value Loss**: 两个损失对比

## 输出文件

| 文件 | 描述 |
|-----|------|
| `best_model.pt` | 最佳模型参数 |
| `best_gantt.png` | 最优解甘特图 |
| `training_curves.png` | 训练曲线 |
| `config.json` | 训练配置 |
| `history.json` | 训练历史数据 |

## 依赖

- Python >= 3.8
- PyTorch >= 1.9
- NumPy
- Matplotlib
- Visdom（可选）
