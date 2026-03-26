import torch
import minari
import numpy as np
import os
from tqdm import tqdm

from utils.buffer import OfflineReplayBuffer
from agents.critic import IQL_Q_V
from agents.bc_ensemble import BC_Ensemble
from agents.abppo import AdaptiveBehaviorPPO
from dynamics_eval import train_dynamics, dynamics_eval

class Stage4Args:
    """
    完善参数类，补全动力学模型初始化所需的全部超参数
    """
    def __init__(self, full_env_name, device):
        # 基础环境信息
        self.env = full_env_name.split('/')[1] if '/' in full_env_name else full_env_name
        self.algo_name = "mobile"
        self.seed = 8
        self.device = str(device)
        self.is_state_norm = True          
        self.is_eval_state_norm = False    
        
        # OPE 评估超参 (对齐 Uni-O4 论文: 每 100步评 1次，Rollout 长度 1000)
        self.eval_step = 100
        self.rollout_length = 1000
        self.rollout_batch_size = 512
        
        # 动力学模型架构与训练参数 (必须补全，否则初始化会报错)
        self.n_ensemble = 7
        self.n_elites = 5
        self.dynamics_hidden_dims = [200, 200, 200, 200]
        self.dynamics_weight_decay = [2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4] # 核心修复
        self.dynamics_lr = 1e-3                                          # 核心修复
        self.dynamics_max_epochs = 1000
        self.max_epochs_since_update = 5
        self.model_type = 'probabilistic'
        
        # 其他冗余参数适配
        self.penalty_coef = 1.0 
        self.gamma = 0.99
        self.scale_strategy = 'dynamic'
        self.real_ratio = 0.05

def main():
    print("====== Starting Stage 4: ABPO Offline Optimization ======")
    dataset_name = "mujoco/walker2d/medium-v0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Stage4Args(dataset_name, device)

    print(f"-> Loading Minari dataset '{dataset_name}'...")
    dataset = minari.load_dataset(dataset_name)
    env = dataset.recover_environment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print("--- 1. Setting up Buffers ---")
    replay_buffer = OfflineReplayBuffer(device, state_dim, action_dim, max_size=dataset.total_steps)
    replay_buffer.load_dataset(dataset_name)
    mean, std = replay_buffer.normalize_state()
    replay_buffer.compute_return(gamma=0.99)
    replay_buffer.reward_normalize(gamma=0.99, scale_strategy="dynamic")

    eval_buffer = OfflineReplayBuffer(device, state_dim, action_dim, max_size=dataset.total_steps)
    eval_buffer.load_dataset(dataset_name)
    
    print("--- 2. Loading Pre-trained Models ---")
    # 1. 加载 Stage 3 的动力学模型
    # 因为 args 补全了，现在 train_dynamics 会正确初始化并加载本地模型
    dynamics = train_dynamics(args, env, eval_buffer)

    # 2. 加载 Stage 1 的 IQL 
    iql = IQL_Q_V(device, state_dim, action_dim, 1024, 2, 1e-4, 2, 0.005, 0.99, 256, 256, 3, 1e-4, 0.7, True)
    try:
        iql.load("checkpoints/stage1/iql_q.pt", "checkpoints/stage1/iql_v.pt") 
        print("   -> Stage 1 IQL Weights Loaded.")
    except Exception as e:
        print(f"   -> Error: Failed to load IQL weights from checkpoints/stage1/! {e}")
        return
    
    # 3. 加载 Stage 2 的 BC Ensemble
    bc_ensemble = BC_Ensemble(num_policies=4, state_dim=state_dim, action_dim=action_dim, device=device)
    try:
        for i in range(4):
            path = f"checkpoints/stage2/bc_policy_{i}.pt"
            bc_ensemble.ensemble[i].load_state_dict(torch.load(path, map_location=device))
        print("   -> Stage 2 BC Ensemble Weights Loaded.")
    except Exception as e:
        print(f"   -> Error: Failed to load BC weights from checkpoints/stage2/! {e}")
        return

    print("--- 3. Starting Stage 4: ABPO Offline Multi-Step Optimization ---")
    abppo = AdaptiveBehaviorPPO(num_policies=4, state_dim=state_dim, action_dim=action_dim, 
                                device=device, lr=1e-4, clip_ratio=0.25, omega=0.7)
    # 将 BC 权重同步给 ABPO 的 policy 和 old_policy
    abppo.load_bc_weights(bc_ensemble)

    total_steps = 100000
    best_mean_qs = np.full(4, -np.inf) 

    pbar = tqdm(range(total_steps), desc="ABPO Training")
    for step in pbar:
        # 严格对齐官方线性衰减
        current_lr = 1e-4 * (1 - step / total_steps)
        current_clip = 0.25 * (1 - step / total_steps)

        losses = abppo.joint_train(replay_buffer, batch_size=512, iql_net=iql, 
                                   current_clip_ratio=current_clip, current_lr=current_lr)

        # 每 100 步触发一次 AM-Q 评估
        if (step + 1) % args.eval_step == 0:
            current_qs = np.zeros(4)
            for i in range(4):
                # 动力学 Rollout 核心调用
                Q_mean, _ = dynamics_eval(
                    args=args, 
                    policy=abppo.ensemble[i].policy, 
                    Q=iql.minQ, 
                    dynamics=dynamics, 
                    replay_buffer=eval_buffer, 
                    env=env, 
                    mean=mean, 
                    std=std
                )
                current_qs[i] = Q_mean
            
            # 安全替换逻辑：只有在虚拟环境得分提升时才更新行为基准
            improve_indices = np.where(current_qs > best_mean_qs)[0]
            if len(improve_indices) > 0:
                best_mean_qs[improve_indices] = current_qs[improve_indices]
                abppo.replace_policies(improve_indices)
                pbar.write(f"Step {step+1}: Policies {improve_indices} improved! Best AM-Q Qs: {best_mean_qs}")

    # 保存最终优化后的最佳策略
    os.makedirs("checkpoints/stage4", exist_ok=True)
    best_policy_idx = np.argmax(best_mean_qs)
    torch.save(abppo.ensemble[best_policy_idx].policy.state_dict(), "checkpoints/stage4/best_offline_policy.pt")
    print(f"\nStage 4 completed! Best policy (idx:{best_policy_idx}) saved to checkpoints/stage4/")

if __name__ == "__main__":
    main()