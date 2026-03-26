import gymnasium as gym
import minari
import torch
import numpy as np
import os
from utils.net import Actor

def evaluate():
    # 1. Base settings
    dataset_name = "mujoco/walker2d/medium-v0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_path = "checkpoints/stage4/best_offline_policy.pt"
    stats_path = "stats/mujoco_walker2d_medium-v0_stats.npz"
    
    if not os.path.exists(policy_path):
        print(f"fatal: can't find policy file {policy_path}")
        return

    # 2. load data and recover environment (align with Minari env)
    print(f"-> Loading dataset and recovering environment: {dataset_name}")
    dataset = minari.load_dataset(dataset_name)
    env = dataset.recover_environment()
    
    # 3. load normalized stat coefficients (same as Stage 1)
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        mean, std = stats['mean'], stats['std']
        print(f"-> Normalization stats loaded from {stats_path}")
    else:
        print(f"Warning: can't find stat file {stats_path}, evaluation may fail.")
        mean, std = 0.0, 1.0

    # 4. init model
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    actor = Actor(state_dim, action_dim).to(device)
    actor.load_state_dict(torch.load(policy_path, map_location=device))
    actor.eval()
    
    print(f"\n--- evaluating Stage 4 best policy ---")
    
    total_rewards = []
    num_episodes = 10
    
    for i in range(num_episodes):
        # Gymnasium API: reset return (obs, info)
        state, _ = env.reset()
        terminated, truncated = False, False
        episode_reward = 0
        
        while not (terminated or truncated):
            # use state norm
            norm_state = (state - mean) / (std + 1e-8)
            norm_state_tensor = torch.FloatTensor(norm_state).to(device)
            
            # use mean action to eval
            with torch.no_grad():
                action = actor.select_action(norm_state_tensor, is_sample=False)
            
            state, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            episode_reward += reward
            
        total_rewards.append(episode_reward)
        print(f"Episode {i+1}: Reward = {episode_reward:.2f}")
        
    print("-" * 30)
    print(f"evaluate success (Episodes: {num_episodes})")
    print(f"Average Return: {np.mean(total_rewards):.2f}")
    print(f"Max Return: {np.max(total_rewards):.2f}")
    print(f"Std Dev: {np.std(total_rewards):.2f}")

if __name__ == "__main__":
    evaluate()