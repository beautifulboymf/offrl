import torch
import numpy as np
import copy
from utils.net import Actor

class BehaviorPPO:
    """Single Behavior PPO Optimizer"""
    def __init__(self, state_dim, action_dim, device, lr, clip_ratio):
        self.device = device
        self.clip_ratio = clip_ratio
        
        self.policy = Actor(state_dim, action_dim).to(device)
        self.old_policy = Actor(state_dim, action_dim).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def get_log_prob(self, policy_net, s, a):
        dist = policy_net.get_dist(s)
        return dist.log_prob(a).sum(dim=-1, keepdim=True)

    def update(self, s, a, adv, current_clip_ratio, current_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

        with torch.no_grad():
            old_log_prob = self.get_log_prob(self.old_policy, s, a)

        new_log_prob = self.get_log_prob(self.policy, s, a)
        
        # Prevent exp() overflow to inf caused by large Log Prob differences
        log_ratio = new_log_prob - old_log_prob
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0) # Safety boundary
        ratio = torch.exp(log_ratio)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - current_clip_ratio, 1.0 + current_clip_ratio) * adv
        
        loss = -torch.min(surr1, surr2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        return loss.item()

    def sync_old_policy(self):
        """Replaces the behavior policy baseline when AM-Q evaluation score improves"""
        self.old_policy.load_state_dict(self.policy.state_dict())


class AdaptiveBehaviorPPO:
    """ABPO Ensemble Manager"""
    def __init__(self, num_policies, state_dim, action_dim, device, lr=1e-4, clip_ratio=0.25, omega=0.7):
        self.num_policies = num_policies
        self.device = device
        self.omega = omega
        
        self.ensemble = [
            BehaviorPPO(state_dim, action_dim, device, lr, clip_ratio) 
            for _ in range(num_policies)
        ]

    def load_bc_weights(self, bc_ensemble_net):
        """Loads BC weights trained in Stage 3 as the initial policy"""
        for i in range(self.num_policies):
            self.ensemble[i].policy.load_state_dict(bc_ensemble_net.ensemble[i].state_dict())
            self.ensemble[i].sync_old_policy()

    def weighted_advantage(self, advantage):
        """Advantage function weighting (corresponds to Uni-O4 hyperparameter omega=0.7)"""
        if self.omega == 0.5:
            return advantage
        weight = torch.where(advantage > 0, self.omega, 1.0 - self.omega)
        return weight * advantage

    def joint_train(self, buffer, batch_size, iql_net, current_clip_ratio, current_lr):
        s, _, _, _, _ = buffer.sample(batch_size)
        losses = []

        for i, bppo in enumerate(self.ensemble):
            with torch.no_grad():
                dist = bppo.old_policy.get_dist(s)
                a = dist.sample()
                a = a.clamp(-1.0, 1.0)
                
                adv = iql_net.get_advantage(s, a)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                adv = self.weighted_advantage(adv)

            # --- Update: Passing current_lr ---
            loss = bppo.update(s, a, adv, current_clip_ratio, current_lr)
            losses.append(loss)

        return np.array(losses)

    def replace_policies(self, indices):
        """Triggers multi-step optimization: only updates policies that outperformed their previous evaluation scores"""
        for idx in indices:
            self.ensemble[idx].sync_old_policy()
            
    def get_best_policy(self, best_idx):
        """Extracts the best performing policy after offline training is complete"""
        return self.ensemble[best_idx].policy