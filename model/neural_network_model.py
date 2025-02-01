import torch.nn as nn

class VoltageNN(nn.Module):
    """Neural network to predict terminal voltage from SOC, current, and temperature."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 16),  # Input: [SOC, current, temperature]
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Output: Voltage
        )

    def forward(self, x):
        return self.fc(x)