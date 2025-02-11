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
    history_length: int = 50
    time_res: float = 1

@dataclass
class TrainEvalParams:
    learning_rate : float = 3e-3
    n_epochs: int = 200
    batch_size: int = 256
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class VanillaNNParams:
    hidden_dim: int = 128
    n_hidden_layers: int = 3

@dataclass
class MoENNParams:

    n_experts: int = 8
    top_k: int = 1
    n_layers_per_expert: int = 1
    hidden_dim: int = 32
    gating_dim: int = 32
    gating_noise_std: float = 1e-4
    balancing_coef: float= 1e-3

@dataclass
class ExperimentalDesignParams:
    n_runs: int = 5