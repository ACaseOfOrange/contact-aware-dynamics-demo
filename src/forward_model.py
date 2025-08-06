import torch
import torch.nn as nn

class ForwardModel(nn.Module):
    def __init__(self, obs_dim=36, action_dim=8, q_dim=8, output_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim + action_dim + q_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, obs, q, action):
        x = torch.cat([obs, q, action], dim=-1)
        return self.fc(x)
