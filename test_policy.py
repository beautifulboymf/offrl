import gymnasium as gym
import torch
import os
from agents.ppo import PPO


def test():
    # --- Configuration ---
    env_name = "Walker2d-v5"
    # model_path = f"checkpoints/{env_name}_ppo_parallel_latest.pt"
    model_path = "checkpoints\Walker2d-v5_UniO4_aligned_20260324-161151_latest.pt"
    device = torch.device("cpu")  # Testing on CPU is usually enough

    # Initialize environment with visualization
    env = gym.make(env_name, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPO(state_dim, action_dim, device)

    # Load model
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Successfully loaded model from {model_path}")
    else:
        print("Warning: No model found, running with random weights.")

    while True:
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                # Use policy mean for visualization
                action = agent.actor.get_dist(obs_t).mean.numpy().flatten()

            obs, r, term, trun, _ = env.step(action)
            done = term or trun
            total_reward += r

        # print(f"Episode finished | Total Reward: {total_reward:.2f}")


if __name__ == "__main__":
    test()
