import torch
import minari
import os
import numpy as np
from utils.buffer import OfflineReplayBuffer

# 导入官方的动力学训练接口 (确保 transition_model 文件夹和 dynamics_eval.py 已在根目录)
from dynamics_eval import train_dynamics

class DynamicsArgs:
    """
    对齐 Uni-O4 官方超参数
    解决 Logger 系统的 KeyError 和 JSON 序列化问题
    """
    def __init__(self, full_env_name, device):
        # --- 核心环境适配 ---
        # 提取简称以匹配 transition_model/utils/termination_fns.py
        # 例如: "mujoco/walker2d/medium-v0" -> "walker2d"
        self.env = full_env_name.split('/')[1] if '/' in full_env_name else full_env_name
        self.algo_name = "mobile"       # 影响日志和模型保存路径
        self.seed = 8                   # 建议与之前的实验对齐
        self.device = str(device)       # 核心修复：转为字符串以支持 JSON 序列化
        
        # --- 状态归一化逻辑对齐 ---
        # 动力学模型训练时 eval_buffer 通常不进行外部归一化，
        # 因为模型内部会调用 StandardScaler 且终止函数依赖原始坐标
        self.is_state_norm = False
        self.is_eval_state_norm = False 
        
        # --- 动力学模型架构对齐 ---
        # MuJoCo 任务标准配置: 200-200-200-200
        self.dynamics_hidden_dims = [200, 200, 200, 200] 
        self.dynamics_weight_decay = [2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4]
        self.dynamics_lr = 1e-3
        self.n_ensemble = 7             # 7个平行概率网络
        self.n_elites = 5               # 选取预测误差最小的5个
        self.model_type = 'probabilistic'

        # --- 训练控制 ---
        self.dynamics_max_epochs = 1000  # 使用 None 或较大值配合早停机制
        self.max_epochs_since_update = 5 # 5个 epoch 不改进则停止
        self.rollout_batch_size = 512    # 官方默认采样大小

        # --- 冗余参数防范 (对齐 Logger.make_log_dirs) ---
        # 这些参数对于训练逻辑非必须，但如果不提供，Logger 系统会抛出 KeyError
        self.penalty_coef = 1.0
        self.rollout_length = 1000       # 重要：AM-Q 离线评估 H 的默认值
        self.real_ratio = 0.05
        self.gamma = 0.99
        self.scale_strategy = 'dynamic'

def main():
    # 数据集建议使用带有多样性动作的 medium，以更好地训练动力学
    dataset_name = "mujoco/walker2d/medium-v0" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading Minari dataset: {dataset_name}")
    dataset = minari.load_dataset(dataset_name)
    env = dataset.recover_environment()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 1. 初始化纯净的 Buffer（适配动力学模型的 sample_all 接口）
    print("Preparing Clean Buffer for Dynamics (Raw States)...")
    buffer = OfflineReplayBuffer(device, state_dim, action_dim, max_size=dataset.total_steps)
    buffer.load_dataset(dataset_name)
    
    # 注意：绝对不要在此处调用 buffer.normalize_state()
    # 否则 Rollout 期间的高度判定会失效，导致 AM-Q 评估极短。
    
    # 2. 配置参数
    args = DynamicsArgs(dataset_name, device)
    
    print(f"\nStarting Stage 3: Ensemble Dynamics Model Training for '{args.env}'")
    print("Training 7 Probabilistic Neural Networks with Early Stopping.")
    
    # 3. 启动官方训练程序
    # train_dynamics 内部会自动创建 1saved_models_False/ 目录并保存 pt 文件
    dynamics = train_dynamics(args, env, buffer)
    
    print("\nStage 3 completed! Dynamics model is stored and ready for AM-Q evaluation in P4.")

if __name__ == "__main__":
    main()