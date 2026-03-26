import torch
import numpy as np
import copy
from utils.net import Actor

class BehaviorPPO:
    """单个行为 PPO 优化器"""
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
        
        # 防止 Log Prob 差异过大导致 exp() 溢出为 inf
        log_ratio = new_log_prob - old_log_prob
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0) # 安全边界
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
        """当 AM-Q 评估分数提高时，用此方法替换行为策略基准"""
        self.old_policy.load_state_dict(self.policy.state_dict())


class AdaptiveBehaviorPPO:
    """ABPO 集成管理器"""
    def __init__(self, num_policies, state_dim, action_dim, device, lr=1e-4, clip_ratio=0.25, omega=0.7):
        self.num_policies = num_policies
        self.device = device
        self.omega = omega
        
        self.ensemble = [
            BehaviorPPO(state_dim, action_dim, device, lr, clip_ratio) 
            for _ in range(num_policies)
        ]

    def load_bc_weights(self, bc_ensemble_net):
        """将 P3 阶段训练好的 BC 权重加载进来作为初始策略"""
        for i in range(self.num_policies):
            self.ensemble[i].policy.load_state_dict(bc_ensemble_net.ensemble[i].state_dict())
            self.ensemble[i].sync_old_policy()

    def weighted_advantage(self, advantage):
        """优势函数加权 (对应 Uni-O4 的超参 omega=0.7)"""
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

            # --- 修改：传入 current_lr ---
            loss = bppo.update(s, a, adv, current_clip_ratio, current_lr)
            losses.append(loss)

        return np.array(losses)

    def replace_policies(self, indices):
        """触发多步优化：只更新评估得分超越过去的策略"""
        for idx in indices:
            self.ensemble[idx].sync_old_policy()
            
    def get_best_policy(self, best_idx):
        """在离线结束后，提取表现最好的策略"""
        return self.ensemble[best_idx].policy