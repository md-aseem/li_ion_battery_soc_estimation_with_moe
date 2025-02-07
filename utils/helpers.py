import numpy as np
import torch
import torch.nn as nn

def parameter_count(model):
    return torch.sum(torch.tensor([p.numel() for name, p in model.named_parameters()]))