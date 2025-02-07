import torch.nn as nn
import torch

class MexicanHat(nn.Module):
    def __init__(self, alpha_init=1):
        super(MexicanHat, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32), requires_grad=True)
    def forward(self, x):
        x = (1-x**2*self.alpha**2)*torch.exp(-x**2*self.alpha**2)
        return x

class VoltageNN(nn.Module):
    """Neural network to predict terminal voltage from SOC, current, and temperature."""

    def __init__(self,
                 in_features: int,
                 hidden_feature: int,
                 out_features: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_feature),  # Input: [SOC, current, temperature]
            MexicanHat(),
            nn.Linear(hidden_feature, hidden_feature),
            MexicanHat(),
            nn.Linear(hidden_feature, hidden_feature),
            MexicanHat(),
            nn.Linear(hidden_feature, out_features)  # Output: Voltage
        )

    def forward(self, x):
        return self.fc(x)