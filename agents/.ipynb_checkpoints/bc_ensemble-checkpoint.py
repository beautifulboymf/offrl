import torch
import numpy as np
from utils.net import Actor

class BC_Ensemble:
    def __init__(self, num_policies, state_dim, action_dim, device, lr=1e-4):
        self.num_policies = num_policies
        self.device = device
        # Instantiate multiple Actors in the ensemble
        self.ensemble = [Actor(state_dim, action_dim).to(device) for _ in range(num_policies)]
        self.optimizers = [torch.optim.Adam(actor.parameters(), lr=lr) for actor in self.ensemble]

    def joint_train(self, buffer, batch_size, alpha):
        # Adapts to your OfflineReplayBuffer.sample output
        s, a, _, _, _ = buffer.sample(batch_size)
        losses = []

        # 1. Pre-calculate log_prob for all Actors on current data (used for diversity penalty)
        all_log_probs = []
        with torch.no_grad():
            for actor in self.ensemble:
                dist = actor.get_dist(s)
                # Calculate log_prob: [batch_size, 1]
                all_log_probs.append(dist.log_prob(a).sum(dim=-1, keepdim=True))
        
        # Concatenate and find the maximum log_prob for each sample
        all_log_probs_t = torch.cat(all_log_probs, dim=1) # [batch_size, num_policies]
        max_log_prob, _ = all_log_probs_t.max(dim=-1, keepdim=True)

        # 2. Update each Actor in sequence
        for i, actor in enumerate(self.ensemble):
            dist = actor.get_dist(s)
            log_prob = dist.log_prob(a).sum(dim=-1, keepdim=True)
            
            # Standard BC loss
            bc_loss = -log_prob.mean()
            
            # Paper Core: Diversity Penalty (Disagreement Penalty)
            # Formula: loss = bc_loss - alpha * (log_prob - max_log_prob)
            penalty = (log_prob - max_log_prob.detach()).mean()
            loss = bc_loss - alpha * penalty
            
            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()
            
            losses.append(loss.item())

        return np.array(losses)

    @torch.no_grad()
    def get_ensemble_actions(self, s):
        """Input a single state and retrieve prediction means from all Actors"""
        actions = []
        for actor in self.ensemble:
            actor.eval()
            dist = actor.get_dist(s)
            actions.append(dist.mean.cpu().numpy())
        return np.array(actions) # [num_policies, action_dim]