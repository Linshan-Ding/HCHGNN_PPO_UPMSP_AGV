"""
train.py - 训练入口
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import os
import json
import numpy as np
from datetime import datetime

from config import config
from env import SchedulingEnv
from rl_agent import PPOAgent
from utils import set_seed
from instance_loader import load_instance, list_instances, print_instance_info
from visualizer import create_visualizer, plot_gantt_chart, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description='HGNN-PPO Scheduling')
    parser.add_argument('--instance_path', type=str, default=None)
    parser.add_argument('--instances_dir', type=str, default='instances')
    parser.add_argument('--instance_name', type=str, default='test_2')
    parser.add_argument('--num_episodes', type=int, default=config.NUM_EPISODES)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--use_visdom', action='store_true', default=True)
    parser.add_argument('--no_visdom', action='store_true')
    parser.add_argument('--visdom_port', type=int, default=8097)
    parser.add_argument('--seed', type=int, default=config.SEED)
    parser.add_argument('--log_interval', type=int, default=config.LOG_INTERVAL)
    parser.add_argument('--save_interval', type=int, default=config.SAVE_INTERVAL)
    parser.add_argument('--save_path', type=str, default='checkpoints')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--generate_instances', action='store_true')
    return parser.parse_args()


def train(args):
    set_seed(args.seed)
    
    if args.generate_instances:
        from generate import generate_instance_set
        generate_instance_set(args.instances_dir, args.seed)
    
    instance_path = args.instance_path
    instance_name = args.instance_name
    if instance_name:
        instance_path = os.path.join(args.instances_dir, instance_name)
    
    if not instance_path or not os.path.exists(instance_path):
        print("Error: Instance not found!")
        instances = list_instances(args.instances_dir)
        if instances:
            print("Available:", instances)
        else:
            print("Run: python generate.py --generate_set")
        return
    
    print(f"Loading: {instance_path}")
    instance_data = load_instance(instance_path)
    print_instance_info(instance_data)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{args.save_path}/{instance_name or 'instance'}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    use_visdom = args.use_visdom and not args.no_visdom
    visualizer = create_visualizer(use_visdom, f"sched_{instance_name}_{timestamp}", args.visdom_port)
    
    print("=" * 50)
    print(f"Device: {config.DEVICE}")
    print(f"Episodes: {args.num_episodes}, LR: {args.lr}")
    print(f"Save: {save_dir}")
    print("=" * 50)
    
    env = SchedulingEnv(instance_data)
    graph, _ = env.reset()
    
    order_dim = len(env.orders[0].get_features())
    agv_dim = len(env.agvs[0].get_features())
    waiting_dim = len(env.waitings[0].get_features())
    buffer_dim = len(env.buffers[0].get_features())
    
    print(f"Dims: Order={order_dim}, AGV={agv_dim}, Waiting={waiting_dim}, Buffer={buffer_dim}")
    print(f"Action space: {env.action_space_size}")
    print(f"Total workpieces: {env.total_workpieces}")
    
    agent = PPOAgent(
        order_dim, agv_dim, waiting_dim, buffer_dim,
        instance_data.num_orders, instance_data.num_machines,
        policy_lr=args.lr, value_lr=args.lr
    )
    
    if args.load_model:
        agent.load(args.load_model)
    
    makespans, rewards = [], []
    best_makespan = float('inf')
    best_gantt_data = None
    update_count = 0
    
    print("\nTraining...")
    
    for ep in range(1, args.num_episodes + 1):
        graph, _ = env.reset()
        ep_reward = 0.0
        done = False
        step_count = 0
        
        while not done:
            action, log_prob, value = agent.select_action(graph)
            
            # 无有效动作时，强制处理事件直到有动作或结束
            if action == -1:
                # 如果没有任何步骤执行过，说明初始状态有问题
                if step_count == 0:
                    print(f"[Warning] Ep {ep}: No valid action at start!")
                    done = True
                    ep_reward = -100.0
                break
            
            next_graph, reward, done, info = env.step(action)
            agent.store_transition(graph, action, reward, done, log_prob, value)
            ep_reward += reward
            graph = next_graph
            step_count += 1
        
        # 只有当有经验时才更新
        if agent.episode_data:
            stats = agent.update()
            update_count += 1
            
            if stats:
                visualizer.log_loss(update_count, stats['policy_loss'], stats['value_loss'], stats['entropy'])
        
        makespan = env.makespan
        makespans.append(makespan)
        rewards.append(ep_reward)
        visualizer.log_episode(ep, makespan, ep_reward)
        
        if makespan < best_makespan:
            best_makespan = makespan
            best_gantt_data = env.get_gantt_data().copy()
            agent.save(f"{save_dir}/best_model.pt")

            plot_gantt_chart(best_gantt_data, instance_data.num_machines, instance_data.num_agvs,
                             f"{save_dir}/best_gantt.png", best_makespan)
        
        if ep % args.log_interval == 0:
            avg_ms = np.mean(makespans[-args.log_interval:])
            print(f"Ep {ep:5d} | Makespan: {makespan:.2f} | Avg: {avg_ms:.2f} | Best: {best_makespan:.2f}")
        
        if ep % args.save_interval == 0:
            agent.save(f"{save_dir}/checkpoint_{ep}.pt")
    
    agent.save(f"{save_dir}/final_model.pt")
    
    if best_gantt_data:
        plot_gantt_chart(best_gantt_data, instance_data.num_machines, instance_data.num_agvs,
                        f"{save_dir}/best_gantt.png", best_makespan)
    
    plot_training_curves(makespans, rewards, agent.stats, f"{save_dir}/training_curves.png")
    
    with open(f"{save_dir}/config.json", 'w') as f:
        json.dump({
            'instance_name': instance_name,
            'num_episodes': args.num_episodes,
            'lr': args.lr,
            'seed': args.seed,
            'best_makespan': best_makespan
        }, f, indent=2)
    
    with open(f"{save_dir}/history.json", 'w') as f:
        json.dump({'makespans': makespans, 'rewards': rewards, 'best_makespan': best_makespan}, f)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Best Makespan: {best_makespan:.2f}")
    print(f"Results: {save_dir}")
    print("=" * 50)
    
    visualizer.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
