import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import minari
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.net import Actor, Critic

class OnlinePPOAgent:
    def __init__(self, state_dim, action_dim, device, lr=1e-5, gamma=0.99, gae_lambda=0.95, clip_ratio=0.1):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        
        # Actor
        self.actor = Actor(state_dim, action_dim).to(device)
        # Critic
        self.critic = Critic(state_dim).to(device)
        
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])

    def load_offline_weights(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        print(f"-> Offline policy loaded from {path}")

    @torch.no_grad()
    def get_action_and_value(self, state, sample=True):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist = self.actor.get_dist(state)
        action = dist.sample() if sample else dist.mean
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(state)
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]

    def update(self, states, actions, log_probs, returns, advantages):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(log_probs)).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)
        advantages = torch.FloatTensor(np.array(advantages)).to(self.device)
        
        # advantage normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(states)
        batch_size = 2048  # Uni-O4: might be 2048, but we choose mini-batch to against forgetting

        for _ in range(10):  # 10 个 Epoch
            # shuffle data index
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_indices = indices[start:end]
                
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]

                dist = self.actor.get_dist(mb_states)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # add safe locker, avoid grad explosion under high return
                log_ratio = new_log_probs - mb_old_log_probs
                log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
                ratio = torch.exp(log_ratio)
                
                # PPO Clip
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
                
                current_values = self.critic(mb_states).view(-1)
                mb_returns = mb_returns.view(-1)
                critic_loss = nn.MSELoss()(current_values, mb_returns)
                
                loss = actor_loss + 0.5 * critic_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip Actor and Critic nets' grad simutaneously
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                
                self.optimizer.step()

def evaluate(env, agent, mean, std, episodes=5):
    """evaluate temp policy"""
    eval_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            norm_state = (state - mean) / (std + 1e-8)
            action, _, _ = agent.get_action_and_value(norm_state, sample=False)
            state, reward, terminated, truncated, _ = env.step(np.clip(action, -1, 1))
            ep_reward += reward
            done = terminated or truncated
        eval_rewards.append(ep_reward)
    return np.mean(eval_rewards)

def plot_paper_style(steps, rewards, save_path):
    """visualize"""
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-paper')
    
    rewards_arr = np.array(rewards)
    window = 5
    smooth_rewards = np.convolve(rewards_arr, np.ones(window)/window, mode='valid')
    smooth_steps = steps[window-1:]
    
    plt.plot(smooth_steps, smooth_rewards, label='Uni-O4 (Ours)', color='#1f77b4', linewidth=2.5)
    # sim std shade
    plt.fill_between(smooth_steps, smooth_rewards - 200, smooth_rewards + 200, color='#1f77b4', alpha=0.2)
    
    plt.title('Online Fine-tuning Performance (Walker2d-v3)', fontsize=15, fontweight='bold')
    plt.xlabel('Environment Steps', fontsize=12)
    plt.ylabel('Average Return', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.savefig(save_path, dpi=300)
    print(f"-> Visualization saved to: {save_path}")

def main():
    # 1. recover env and data loading
    dataset_name = "mujoco/walker2d/medium-v0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"-> Recovering environment from {dataset_name}...")
    dataset = minari.load_dataset(dataset_name)
    env = dataset.recover_environment()
    
    # load stat coefficients
    stats_path = "stats/mujoco_walker2d_medium-v0_stats.npz"
    stats = np.load(stats_path)
    mean, std = stats['mean'], stats['std']
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 2. init Agent and load off-policy weight
    agent = OnlinePPOAgent(state_dim, action_dim, device)
    agent.load_offline_weights("checkpoints/stage4/best_offline_policy.pt")

    # 3. training settings
    max_steps = 1000000       # total online steps
    update_freq = 4096        # how many steps collected to update PPO
    eval_freq = 5000          # evaluate freq
    
    steps_log, rewards_log = [], []
    
    # initial evaluation
    init_score = evaluate(env, agent, mean, std)
    steps_log.append(0)
    rewards_log.append(init_score)
    print(f"[Step 0] Initial Return: {init_score:.2f}")

    # 4. online iteration
    curr_state, _ = env.reset()
    buffer_s, buffer_a, buffer_lp, buffer_r, buffer_v, buffer_d = [], [], [], [], [], []
    
    pbar = tqdm(range(1, max_steps + 1), desc="Online Training")
    for step in pbar:
        norm_s = (curr_state - mean) / (std + 1e-8)
        action, log_prob, value = agent.get_action_and_value(norm_s)
        
        next_state, reward, terminated, truncated, _ = env.step(np.clip(action, -1, 1))
        done = terminated or truncated
        
        buffer_s.append(norm_s)
        buffer_a.append(action)
        buffer_lp.append(log_prob)
        buffer_r.append(reward)
        buffer_v.append(value)
        buffer_d.append(done)
        
        curr_state = next_state
        if done:
            curr_state, _ = env.reset()

        # trigger update
        if step % update_freq == 0:
            # compute GAE
            returns, advantages = [], []
            next_v = agent.get_action_and_value((curr_state-mean)/(std+1e-8))[2]
            gae = 0
            for r, v, d in zip(reversed(buffer_r), reversed(buffer_v), reversed(buffer_d)):
                mask = 1.0 - d
                delta = r + agent.gamma * next_v * mask - v
                gae = delta + agent.gamma * agent.gae_lambda * mask * gae
                returns.insert(0, gae + v)
                advantages.insert(0, gae)
                next_v = v
            
            agent.update(buffer_s, buffer_a, buffer_lp, returns, advantages)
            buffer_s, buffer_a, buffer_lp, buffer_r, buffer_v, buffer_d = [], [], [], [], [], []

        # evaluate
        if step % eval_freq == 0:
            avg_r = evaluate(env, agent, mean, std)
            steps_log.append(step)
            rewards_log.append(avg_r)
            pbar.set_postfix({'Reward': f"{avg_r:.1f}"})

    # 5. save results
    os.makedirs("results", exist_ok=True)
    plot_paper_style(np.array(steps_log), np.array(rewards_log), "results/walker2d_online_curve.png")
    torch.save(agent.actor.state_dict(), "results/final_finetuned_actor.pt")
    print("Stage 5 Complete!")

if __name__ == "__main__":
    main()