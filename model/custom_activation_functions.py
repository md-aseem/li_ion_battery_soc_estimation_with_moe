import torch
import torch.nn as nn

class MexicanHat(nn.Module):
    def __init__(self, alpha_init=1):
        super(MexicanHat, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32), requires_grad=True)
    def forward(self, x):
        x = (1-x**2*self.alpha**2)*torch.exp(-x**2*self.alpha**2)
        return x