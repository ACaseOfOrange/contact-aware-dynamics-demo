import torch
import torch.nn as nn

class ObjectClassifier(nn.Module):
    def __init__(self, input_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, tactile, keypoints):
        x = torch.cat([tactile, keypoints], dim=-1)
        return self.fc(x)