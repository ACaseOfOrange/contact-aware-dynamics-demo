import torch
import torch.nn as nn

class GraspClassifier(nn.Module):
    def __init__(self, tactile_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(tactile_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, tactile):
        return self.fc(tactile)