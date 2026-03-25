import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.net import ValueMLP, QMLP, DoubleQMLP


class IQL_Q_V(nn.Module):
    def __init__(
        self,
        device,
        state_dim,
        action_dim,
        q_hidden_dim,
        q_depth,
        Q_lr,
        target_update_freq,
        tau,
        gamma,
        batch_size,
        v_hidden_dim,
        v_depth,
        v_lr,
        omega,
        is_double_q,
    ):
        super().__init__()
        self._device = device
        self._omega = omega
        self._is_double_q = is_double_q
        self._gamma = gamma
        self._tau = tau
        self._batch_size = batch_size
        self._target_update_freq = target_update_freq
        self._total_update_step = 0

        # 初始化 Q 网络
        if is_double_q:
            self._Q = DoubleQMLP(state_dim, action_dim, q_hidden_dim, q_depth).to(
                device
            )
            self._target_Q = DoubleQMLP(
                state_dim, action_dim, q_hidden_dim, q_depth
            ).to(device)
        else:
            self._Q = QMLP(state_dim, action_dim, q_hidden_dim, q_depth).to(device)
            self._target_Q = QMLP(state_dim, action_dim, q_hidden_dim, q_depth).to(
                device
            )

        self._target_Q.load_state_dict(self._Q.state_dict())
        self._q_optimizer = torch.optim.Adam(self._Q.parameters(), lr=Q_lr)

        # 初始化 V 网络
        self._value = ValueMLP(state_dim, v_hidden_dim, v_depth).to(device)
        self._v_optimizer = torch.optim.Adam(self._value.parameters(), lr=v_lr)

    def minQ(self, s, a):
        Q1, Q2 = self._Q(s, a)
        return torch.min(Q1, Q2)

    def target_minQ(self, s, a):
        Q1, Q2 = self._target_Q(s, a)
        return torch.min(Q1, Q2)

    def expectile_loss(self, diff):
        weight = torch.where(diff > 0, self._omega, (1 - self._omega))
        return weight * (diff**2)

    def update(self, replay_buffer):
        # 完美适配你的 OfflineReplayBuffer.sample 输出
        s, a, r, s_p, done = replay_buffer.sample(self._batch_size)
        not_done = 1.0 - done

        # 1. 更新 V 网络 (Expectile Regression)
        with torch.no_grad():
            self._target_Q.eval()
            if self._is_double_q:
                target_q = self.target_minQ(s, a)
            else:
                target_q = self._target_Q(s, a)

        value = self._value(s)
        # 注意：这里的 target_q - value 即优势函数 A(s,a)
        value_loss = self.expectile_loss(target_q - value).mean()

        self._v_optimizer.zero_grad()
        value_loss.backward()
        self._v_optimizer.step()

        # 2. 更新 Q 网络 (MSE Loss)
        with torch.no_grad():
            self._value.eval()
            next_v = self._value(s_p)

        target_q_val = r + not_done * self._gamma * next_v

        if self._is_double_q:
            current_q1, current_q2 = self._Q(s, a)
            q_loss = (
                (current_q1 - target_q_val) ** 2 + (current_q2 - target_q_val) ** 2
            ).mean()
        else:
            Q = self._Q(s, a)
            q_loss = F.mse_loss(Q, target_q_val)

        self._q_optimizer.zero_grad()
        q_loss.backward()
        self._q_optimizer.step()

        # 3. 目标网络软更新
        self._total_update_step += 1
        if self._total_update_step % self._target_update_freq == 0:
            for param, target_param in zip(
                self._Q.parameters(), self._target_Q.parameters()
            ):
                target_param.data.copy_(
                    self._tau * param.data + (1 - self._tau) * target_param.data
                )

        return q_loss.item(), value_loss.item()

    def get_advantage(self, s, a):
        # 在离线 PPO 阶段用来替代 GAE
        if self._is_double_q:
            return self.minQ(s, a) - self._value(s)
        else:
            return self._Q(s, a) - self._value(s)
