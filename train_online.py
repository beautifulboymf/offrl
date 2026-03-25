import gymnasium as gym
import torch
import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from agents.ppo import PPO
from utils.buffer import RolloutBuffer


# Trick 2: state normalization
class Normalization:
    def __init__(self, shape, mean=None, std=None):
        self.mean = mean if mean is not None else np.zeros(shape)
        self.std = std if std is not None else np.ones(shape)

    def __call__(self, x):
        return (x - self.mean) / (self.std + 1e-8)


# Trick 4: reward scaling
class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape=shape)
        self.R = np.zeros(shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)
        return x

    def reset(self):
        self.R = np.zeros(self.shape)


class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if len(x.shape) > 0 else 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean, self.var, self.count = new_mean, new_var, tot_count

    @property
    def std(self):
        return np.sqrt(self.var)


def train():
    env_name = "Walker2d-v5"
    num_envs = 8
    steps_per_env = 128
    total_steps = int(1e7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{env_name}_UniO4_aligned_{timestamp}"
    os.makedirs("logs", exist_ok=True)
    writer = SummaryWriter(log_dir=f"logs/{exp_name}")

    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(env_name) for _ in range(num_envs)]
    )
    eval_env = gym.make(env_name)

    # offline_data_stats = np.load(f"stats/{env_name}_offline_stats.npz")
    # mean, std = offline_data_stats['mean'], offline_data_stats['std']
    state_norm = Normalization(envs.single_observation_space.shape[0])
    reward_scaling = RewardScaling(shape=num_envs, gamma=0.99)

    agent = PPO(
        envs.single_observation_space.shape[0],
        envs.single_action_space.shape[0],
        device,
    )
    buffer = RolloutBuffer(
        steps_per_env,
        num_envs,
        envs.single_observation_space.shape[0],
        envs.single_action_space.shape[0],
        device,
    )

    obs, _ = envs.reset()
    current_step = 0
    start_time = time.time()

    while current_step < total_steps:
        reward_scaling.reset()
        for _ in range(steps_per_env):
            # state normalization
            norm_obs = state_norm(obs)
            with torch.no_grad():
                obs_t = torch.as_tensor(norm_obs, dtype=torch.float32).to(device)
                dist = agent.actor.get_dist(obs_t)
                action = dist.sample()
                log_p = dist.log_prob(action)

            next_obs, reward, term, trun, _ = envs.step(action.cpu().numpy())

            # reward scaling
            scaled_reward = reward_scaling(reward)
            # scaled_reward = reward

            buffer.store(
                norm_obs,
                action.cpu().numpy(),
                scaled_reward,
                state_norm(next_obs),
                log_p.cpu().numpy(),
                term | trun,
            )
            obs = next_obs
            current_step += num_envs

        aloss, closs, entropy = agent.update(buffer, current_step, total_steps)

        if (current_step % 20480) < num_envs:
            avg_reward = evaluate_policy(eval_env, agent, state_norm, device)
            fps = int(current_step / (time.time() - start_time))
            print(
                f"[{current_step}/{total_steps}] Reward: {avg_reward:.1f} | ALoss: {aloss:.4f} | CLoss: {closs:.4f} | FPS: {fps}"
            )
            writer.add_scalar("Eval/Reward", avg_reward, current_step)
            writer.add_scalar("Loss/Actor", aloss, current_step)
            writer.add_scalar("Loss/Critic", closs, current_step)
            agent.save(f"checkpoints/{exp_name}_latest.pt")


def evaluate_policy(env, agent, state_norm, device, episodes=3):
    total_reward = 0
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        while not done:
            s_norm = state_norm(s)
            with torch.no_grad():
                s_t = (
                    torch.as_tensor(s_norm, dtype=torch.float32).to(device).unsqueeze(0)
                )
                a = agent.actor.get_dist(s_t).mean.cpu().numpy().flatten()
            s, r, term, trun, _ = env.step(a)
            done = term or trun
            total_reward += r
    return total_reward / episodes


if __name__ == "__main__":
    train()
