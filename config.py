from dataclasses import dataclass, field
import numpy as np
from typing import List
import torch

@dataclass
class MultiStageDataParams:
    temps: List[int] = field(default_factory=lambda: [45])
    aging_types: List[str] = field(default_factory=lambda: ['z'])
    testpoints: np.ndarray = field(default_factory=lambda: np.arange(0, 2, 1))
    reference_performance_test: List[str] = field(default_factory= lambda: ['ET'])
    cells: List[int] = field(default_factory= lambda: list(np.arange(0, 2, 1)))
    stages: List[int] = field(default_factory=lambda: [1, 2])

    time_res: float = 1
    history_length: int = 200
    add_feature_cols: bool = True

@dataclass
class CalceDataParams:
    temps: List[int] = field(default_factory=lambda: [0, 10, 20, 25, 30, 40, 50])
    history_length: int = 100
    time_res: float = 1

@dataclass
class TrainEvalParams:
    learning_rate : float = 1e-3
    n_epochs: int = 50
    batch_size: int = 512
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class VanillaNNParams:
    hidden_dim: int = 128

@dataclass
class MoENNParams:

    n_experts: int = 8
    top_k: int = 1
    n_layers_per_expert: int = 1
    hidden_dim: int = 64
    gating_dim: int = 32
    gating_noise_std: float = 1e-4
    balancing_coef: float= 1e-3