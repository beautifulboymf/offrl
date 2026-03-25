import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils.net import Actor, Critic


class PPO:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)

        self.lr = 3e-4  # 3e-5
        self.gamma = 0.99
        self.lamda = 0.95
        self.epsilon = 0.2  # 0.1
        self.omega = 0.5  # 0.7
        self.K_epochs = 10  # 30
        self.entropy_coef = 0.01

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)
        self.optimizer_critic = optim.Adam(
            self.critic.parameters(), lr=self.lr, eps=1e-5
        )

    def update(self, buffer, current_step, total_steps):
        s, a, r, s_next, old_logprob, done = buffer.get_data()

        with torch.no_grad():
            # 1. compute advantages and value targets (with GAE) and prepare for Value Clipping
            vs = self.critic(s)
            vs_next = self.critic(s_next)
            deltas = r + self.gamma * vs_next * (1 - done) - vs

            adv = torch.zeros_like(deltas).to(self.device)
            gae = 0
            for t in reversed(range(s.size(0))):
                gae = deltas[t] + self.gamma * self.lamda * (1 - done[t]) * gae
                adv[t] = gae
            v_target = adv + vs

            # old value for clipping
            old_vs = vs.clone().reshape(-1, 1)

            # Flatten
            s, a, old_logprob, adv, v_target = [
                x.reshape(-1, x.size(-1)) for x in [s, a, old_logprob, adv, v_target]
            ]

            # 2. Asymmetric advantage weighting (Uni-O4 Core)
            weight = torch.where(adv > 0, self.omega, (1 - self.omega))
            adv = weight * adv
            adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        a_losses, c_losses, ents = [], [], []
        for _ in range(self.K_epochs):
            sampler = BatchSampler(SubsetRandomSampler(range(s.size(0))), 512, False)
            for idx in sampler:
                # Actor update
                dist = self.actor.get_dist(s[idx])
                logprob_now = dist.log_prob(a[idx])
                entropy = dist.entropy().sum(-1, keepdim=True)
                ratio = torch.exp(
                    logprob_now.sum(-1, keepdim=True) - old_logprob[idx].sum(-1, keepdim=True)
                )

                surr1 = ratio * adv[idx]
                surr2 = (
                    torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv[idx]
                )
                loss_actor = (
                    -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                )

                self.optimizer_actor.zero_grad()
                loss_actor.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                # Critic update with Value Clipping
                v_curr = self.critic(s[idx])
                v_loss_unclipped = (v_curr - v_target[idx]).pow(2)
                v_clipped = old_vs[idx] + torch.clamp(
                    v_curr - old_vs[idx], -self.epsilon, self.epsilon
                )
                v_loss_clipped = (v_clipped - v_target[idx]).pow(2)
                loss_critic = torch.max(v_loss_unclipped, v_loss_clipped).mean()

                self.optimizer_critic.zero_grad()
                loss_critic.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

                a_losses.append(loss_actor.item())
                c_losses.append(loss_critic.item())
                ents.append(entropy.mean().item())

        self.lr_decay(current_step, total_steps)
        return np.mean(a_losses), np.mean(c_losses), np.mean(ents)

    def lr_decay(self, step, total_steps):
        lr_now = self.lr * (1 - step / total_steps)
        for p in self.optimizer_actor.param_groups:
            p["lr"] = lr_now
        for p in self.optimizer_critic.param_groups:
            p["lr"] = lr_now

    def save(self, path):
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
