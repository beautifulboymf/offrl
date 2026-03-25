import numpy as np
import minari
import torch


class RolloutBuffer:
    def __init__(self, steps_per_env, num_envs, state_dim, action_dim, device):
        # Shape: (Steps, Envs, Dimension) to maintain time consistency for GAE
        self.s = np.zeros((steps_per_env, num_envs, state_dim))
        self.a = np.zeros((steps_per_env, num_envs, action_dim))
        self.r = np.zeros((steps_per_env, num_envs, 1))
        self.s_next = np.zeros((steps_per_env, num_envs, state_dim))
        self.logprob = np.zeros((steps_per_env, num_envs, action_dim))
        self.done = np.zeros((steps_per_env, num_envs, 1))

        self.ptr = 0
        self.steps_per_env = steps_per_env
        self.num_envs = num_envs
        self.device = device

    def store(self, s, a, r, s_next, logprob, done):
        # Store parallel transitions. ptr will go from 0 to 255.
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r.reshape(-1, 1)
        self.s_next[self.ptr] = s_next
        self.logprob[self.ptr] = logprob
        self.done[self.ptr] = done.reshape(-1, 1)
        self.ptr += 1

    def get_data(self):
        # Convert to tensor and reset pointer for next rollout
        data = (
            torch.FloatTensor(self.s).to(self.device),
            torch.FloatTensor(self.a).to(self.device),
            torch.FloatTensor(self.r).to(self.device),
            torch.FloatTensor(self.s_next).to(self.device),
            torch.FloatTensor(self.logprob).to(self.device),
            torch.FloatTensor(self.done).to(self.device),
        )
        self.reset()  # Critical fix for IndexError
        return data

    def reset(self):
        self.ptr = 0


class OfflineReplayBuffer:
    def __init__(self, device, state_dim, action_dim, max_size, percentage=1.0):
        self.device = device
        self.max_size = int(max_size * percentage)
        self.ptr = 0
        self.size = 0

        # 分配内存
        self.s = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.a = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.r = np.zeros((self.max_size, 1), dtype=np.float32)
        self.s_next = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.float32)
        self.returns = np.zeros((self.max_size, 1), dtype=np.float32)

    def load_dataset(self, dataset_name, clip=False):
        """使用 Minari 加载数据集"""
        print(f"Loading Minari dataset: {dataset_name}")
        dataset = minari.load_dataset(dataset_name)

        # 直接遍历数据集对象本身，或者 dataset.iterate_episodes()
        for episode in dataset:
            # 状态序列长度通常比动作多 1
            obs = episode.observations[:-1]
            next_obs = episode.observations[1:]
            actions = episode.actions
            rewards = episode.rewards.reshape(-1, 1)

            # 合并终止（terminations）和截断（truncations）
            dones = (episode.terminations | episode.truncations).reshape(-1, 1)

            if clip:
                actions = np.clip(actions, -1.0, 1.0)

            ep_len = len(actions)
            # 防止溢出
            if self.ptr + ep_len > self.max_size:
                ep_len = self.max_size - self.ptr
                if ep_len <= 0:
                    break
                obs = obs[:ep_len]
                next_obs = next_obs[:ep_len]
                actions = actions[:ep_len]
                rewards = rewards[:ep_len]
                dones = dones[:ep_len]

            self.s[self.ptr : self.ptr + ep_len] = obs
            self.a[self.ptr : self.ptr + ep_len] = actions
            self.r[self.ptr : self.ptr + ep_len] = rewards
            self.s_next[self.ptr : self.ptr + ep_len] = next_obs
            self.done[self.ptr : self.ptr + ep_len] = dones

            self.ptr += ep_len
            self.size = min(self.size + ep_len, self.max_size)

        print(f"Dataset loaded. Total transitions in buffer: {self.size}")

    def compute_return(self, gamma):
        """计算每一步的 Return-to-go，用于价值函数的初始估计和缩放"""
        print("Computing returns...")
        curr_ret = 0.0
        for i in reversed(range(self.size)):
            curr_ret = self.r[i][0] + gamma * curr_ret * (1.0 - self.done[i][0])
            self.returns[i][0] = curr_ret

    def reward_normalize(self, gamma, scale_strategy="dynamic"):
        """按照 Uni-O4 的设定，根据 Return 的标准差等进行 Reward 缩放"""
        if scale_strategy == "normal":
            std = self.r[: self.size].std() + 1e-5
            self.r[: self.size] = (
                self.r[: self.size] - self.r[: self.size].mean()
            ) / std
        elif scale_strategy == "dynamic":
            # 基于回报的绝对最大值进行缩放 (一种简化的动态缩放策略)
            max_ret = np.abs(self.returns[: self.size]).max()
            if max_ret > 0:
                self.r[: self.size] /= max_ret + 1e-5
                self.returns[: self.size] /= max_ret + 1e-5

    def normalize_state(self):
        """状态归一化：提取均值和方差，并应用到已有数据。返回均值和标准差供在线环境使用"""
        print("Normalizing states...")
        mean = self.s[: self.size].mean(axis=0, keepdims=True)
        std = self.s[: self.size].std(axis=0, keepdims=True) + 1e-5

        self.s[: self.size] = (self.s[: self.size] - mean) / std
        self.s_next[: self.size] = (self.s_next[: self.size] - mean) / std

        return mean.flatten(), std.flatten()

    def sample(self, batch_size):
        """为离线训练提供采样接口"""
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.s[ind]).to(self.device),
            torch.FloatTensor(self.a[ind]).to(self.device),
            torch.FloatTensor(self.r[ind]).to(self.device),
            torch.FloatTensor(self.s_next[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device),
        )
