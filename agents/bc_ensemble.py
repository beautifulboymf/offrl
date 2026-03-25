import torch
import numpy as np
from utils.net import Actor

class BC_Ensemble:
    def __init__(self, num_policies, state_dim, action_dim, device, lr=1e-4):
        self.num_policies = num_policies
        self.device = device
        # 实例化集成中的多个 Actor
        self.ensemble = [Actor(state_dim, action_dim).to(device) for _ in range(num_policies)]
        self.optimizers = [torch.optim.Adam(actor.parameters(), lr=lr) for actor in self.ensemble]

    def joint_train(self, buffer, batch_size, alpha):
        # 适配你的 OfflineReplayBuffer.sample 输出
        s, a, _, _, _ = buffer.sample(batch_size)
        losses = []

        # 1. 预计算所有 Actor 对当前数据的 log_prob (用于多样性惩罚)
        all_log_probs = []
        with torch.no_grad():
            for actor in self.ensemble:
                dist = actor.get_dist(s)
                # 计算 log_prob: [batch_size, 1]
                all_log_probs.append(dist.log_prob(a).sum(dim=-1, keepdim=True))
        
        # 拼接并找到每个样本的最大 log_prob
        all_log_probs_t = torch.cat(all_log_probs, dim=1) # [batch_size, num_policies]
        max_log_prob, _ = all_log_probs_t.max(dim=-1, keepdim=True)

        # 2. 依次更新每个 Actor
        for i, actor in enumerate(self.ensemble):
            dist = actor.get_dist(s)
            log_prob = dist.log_prob(a).sum(dim=-1, keepdim=True)
            
            # 标准 BC 损失
            bc_loss = -log_prob.mean()
            
            # 论文核心：多样性惩罚项 (Disagreement Penalty)
            # 公式: loss = bc_loss - alpha * (log_prob - max_log_prob)
            penalty = (log_prob - max_log_prob.detach()).mean()
            loss = bc_loss - alpha * penalty
            
            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()
            
            losses.append(loss.item())

        return np.array(losses)

    @torch.no_grad()
    def get_ensemble_actions(self, s):
        """输入单个状态，获取所有 Actor 的预测均值"""
        actions = []
        for actor in self.ensemble:
            actor.eval()
            dist = actor.get_dist(s)
            actions.append(dist.mean.cpu().numpy())
        return np.array(actions) # [num_policies, action_dim]