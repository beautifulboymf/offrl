import torch
import minari
import numpy as np
from tqdm import tqdm

from utils.buffer import OfflineReplayBuffer
from agents.critic import IQL_Q_V
from agents.bc_ensemble import BC_Ensemble
from agents.abppo import AdaptiveBehaviorPPO

# 假设你已经把官方的 dynamics_eval.py 和 transition_model 拷过来了
# from dynamics_eval import train_dynamics, dynamics_eval
# 如果不想立刻跑长达数小时的 dynamics，可以先 mock 一个，验证 ABPO 运行逻辑
def mock_dynamics_eval(bppo, iql_q, buffer):
    """一个假的 OPE 评估，随机返回一个分数，仅用于测试 ABPO 的替换逻辑是否无 Bug"""
    return np.random.uniform(50, 100)

def test_abppo_pipeline():
    dataset_name = "mujoco/walker2d/expert-v0"
    dataset = minari.load_dataset(dataset_name)
    env = dataset.recover_environment()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 准备数据
    buffer = OfflineReplayBuffer(device, state_dim, action_dim, max_size=dataset.total_steps)
    buffer.load_dataset(dataset_name)
    buffer.normalize_state()
    buffer.compute_return(gamma=0.99)
    buffer.reward_normalize(gamma=0.99, scale_strategy="dynamic")

    # 2. IQL 预训练 (Mock，假设已经训练好，我们只训练 500 步)
    print("--- 1. Training IQL (Mock 500 steps) ---")
    iql = IQL_Q_V(device, state_dim, action_dim, 1024, 2, 1e-4, 2, 0.005, 0.99, 256, 256, 3, 1e-4, 0.7, True)
    for _ in tqdm(range(500)): iql.update(buffer)

    # 3. BC Ensemble 预训练 (Mock 500 步)
    print("--- 2. Training BC Ensemble (Mock 500 steps) ---")
    bc_ensemble = BC_Ensemble(num_policies=4, state_dim=state_dim, action_dim=action_dim, device=device)
    for _ in tqdm(range(500)): bc_ensemble.joint_train(buffer, 256, alpha=0.1)

    # 4. ABPO 离线微调 (P4 核心)
    print("--- 3. Running ABPO Offline Optimization ---")
    # 对齐超参：clip_ratio=0.25, lr=1e-4, omega=0.7
    abppo = AdaptiveBehaviorPPO(num_policies=4, state_dim=state_dim, action_dim=action_dim, 
                                device=device, lr=1e-4, clip_ratio=0.25, omega=0.7)
    
    # 将 BC 的权重加载进 ABPO 作为起点
    abppo.load_bc_weights(bc_ensemble)

    total_bppo_steps = 2000 # 原文是 10000 步
    eval_step = 100         # 每 100 步 OPE 评估一次
    best_mean_qs = np.zeros(4)

    pbar = tqdm(range(total_bppo_steps), desc="ABPO Training")
    for step in pbar:
        # Clip Ratio 线性衰减
        current_clip = 0.25 * (1 - step / total_bppo_steps)
        
        losses = abppo.joint_train(buffer, 256, iql, current_clip)
        
        # 触发 AM-Q OPE 策略评估替换逻辑
        if (step + 1) % eval_step == 0:
            current_qs = np.zeros(4)
            for i in range(4):
                # 真实情况下这里应该调用: 
                # current_qs[i] = dynamics_eval(args, abppo.ensemble[i], iql.minQ, dynamics, eval_buffer, ...)
                current_qs[i] = mock_dynamics_eval(abppo.ensemble[i], iql.minQ, buffer) 
                
            # 找到得分超越历史最佳的策略索引
            improve_indices = np.where(current_qs > best_mean_qs)[0]
            if len(improve_indices) > 0:
                # 更新最佳得分，并替换旧策略 (old_policy <- policy)
                best_mean_qs[improve_indices] = current_qs[improve_indices]
                abppo.replace_policies(improve_indices)
                pbar.write(f"Step {step+1}: Policies {improve_indices} improved and replaced! Best Qs: {best_mean_qs}")

    print("P4 Test Finished! ABPO multi-step optimization logic runs successfully.")

if __name__ == "__main__":
    test_abppo_pipeline()