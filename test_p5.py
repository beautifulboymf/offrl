import numpy as np
import torch
import gymnasium as gym
from utils.buffer import OfflineReplayBuffer
from dynamics_eval import train_dynamics

# 模拟 Uni-O4 main.py 中的参数配置
class SimpleArgs:
    def __init__(self):
        # --- 基础与路径配置 ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.env = "mujoco/walker2d/expert-v0" 
        self.seed = 8
        self.algo_name = "mobile"       # 影响日志文件夹命名
        self.is_eval_state_norm = False # 影响模型路径生成逻辑
        
        # --- 动力学模型架构参数 ---
        self.dynamics_hidden_dims = [200, 200, 200, 200] 
        self.dynamics_weight_decay = [2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4] # 5个值
        self.dynamics_lr = 1e-3
        self.n_ensemble = 7             # 与 dynamics_eval.py 第 91 行对齐
        self.n_elites = 5
        self.model_type = 'probabilistic'

        # --- 核心训练控制参数 (解决最近两次报错) ---
        self.dynamics_max_epochs = 100  # 解决最新报错
        self.max_epochs_since_update = 5 # 解决上一次报错，早停机制阈值
        self.dynamics_steps = 100       # 您的测试步数
        self.rollout_batch_size = 512
        self.dynamics_update_freq = 1
        
        # --- 日志系统所需的冗余参数 (KeyError 防范) ---
        # Logger.make_log_dirs 会遍历这些键来生成实验标识
        self.penalty_coef = 1.0
        self.rollout_length = 1000
        self.real_ratio = 0.05
        self.gamma = 0.99
        self.scale_strategy = 'dynamic'

def test_dynamics_learning():
    args = SimpleArgs()
    env_name = "mujoco/walker2d/expert-v0" # 或者您本地现有的数据集
    gym_env_name = "Walker2d-v5"
    
    print(f"--- 开始测试动力学模型: 环境 {env_name} ---")
    
    # 1. 加载数据集到您的 OfflineReplayBuffer
    env = gym.make(gym_env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 假设您的 Buffer 已经处理好了 Minari 加载
    buffer = OfflineReplayBuffer(args.device, state_dim, action_dim, max_size=1000000)
    # 注意：这里需要确保您的 OfflineReplayBuffer 有 load_dataset 方法
    # 并且已经根据 Uni-O4 的要求进行了 reward_normalize
    buffer.load_dataset(env_name) 
    
    # 2. 状态归一化 (动力学模型对这个非常敏感)
    # Uni-O4 必须在训练动力学模型前计算离线数据的均值和方差
    mean, std = buffer.normalize_state()
    print(f"数据加载完成。状态均值样例: {mean[:3]}, 标准差样例: {std[:3]}")

    # 3. 训练动力学模型
    # 调用移植过来的 dynamics_eval.py 中的封装函数
    print("正在训练动力学集成模型 (Ensemble)...")
    dynamics_model = train_dynamics(args, env, buffer)
    
    # 4. 验证预测效果 (One-step Validation)
    print("\n--- 验证预测准确率 ---")
    # 从 Buffer 中抽样一组真实转换 (s, a, r, s_p, d)
    s_tensor, a_tensor, r_tensor, s_p_tensor, d_tensor = buffer.sample(batch_size=5)
    
    # 核心修复：将 CUDA Tensor 转换为 NumPy 数组
    s = s_tensor.cpu().numpy()
    a = a_tensor.cpu().numpy()
    r = r_tensor.cpu().numpy()
    s_p = s_p_tensor.cpu().numpy()

    with torch.no_grad():
        # 现在传入的是 NumPy 数组，符合接口要求
        pred_next_obs, pred_reward, _, _ = dynamics_model.step(s, a)
        
    # 计算误差 (注意：如果返回的是 NumPy，直接减法即可)
    obs_error = np.mean((pred_next_obs - s_p)**2)
    rew_error = np.mean((pred_reward - r)**2)
    
    print(f"单步预测 状态均方误差 (MSE): {obs_error:.6f}")
    print(f"单步预测 奖励均方误差 (MSE): {rew_error:.6f}")
    
    if obs_error < 1e-3:
        print("✅ 结果判定: 动力学模型学习效果良好，可以用于策略评估。")
    else:
        print("❌ 结果判定: 误差偏大，请检查数据归一化或模型配置。")

if __name__ == "__main__":
    test_dynamics_learning()