import torch
import torch.nn as nn
from torch.distributions import Normal


def soft_clamp(x: torch.Tensor, bound: tuple) -> torch.Tensor:
    # Uni-O4 specific soft clamp using Tanh to map log_std
    low, high = bound
    x = torch.tanh(x)
    return low + 0.5 * (high - low) * (x + 1)


def init_weights(m):
    # Orthogonal initialization with bias=0
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=1.0)
        nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Aligned with Table 5: 256-256-256 depth
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 2 * action_dim),
        )
        self.log_std_bound = (-5.0, 0.0)  # Aligned with source
        self.apply(init_weights)

    def get_dist(self, s):
        mu, log_std = self.trunk(s).chunk(2, dim=-1)
        log_std = soft_clamp(log_std, self.log_std_bound)
        return Normal(mu, log_std.exp())


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        # Critic typically uses ReLU in Uni-O4 online finetune
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.apply(init_weights)

    def forward(self, s):
        return self.trunk(s)


def build_mlp(input_dim, output_dim, hidden_dim, depth, activation=nn.ReLU):
    layers = [nn.Linear(input_dim, hidden_dim), activation()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), activation()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class ValueMLP(nn.Module):
    def __init__(self, state_dim, hidden_dim, depth):
        super().__init__()
        self.net = build_mlp(state_dim, 1, hidden_dim, depth)

    def forward(self, s):
        return self.net(s)


class QMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, depth):
        super().__init__()
        self.net = build_mlp(state_dim + action_dim, 1, hidden_dim, depth)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)


class DoubleQMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, depth):
        super().__init__()
        self.q1 = build_mlp(state_dim + action_dim, 1, hidden_dim, depth)
        self.q2 = build_mlp(state_dim + action_dim, 1, hidden_dim, depth)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)
