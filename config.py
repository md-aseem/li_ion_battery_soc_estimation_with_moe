from dataclasses import dataclass, field
import numpy as np
from typing import List
import torch

@dataclass
class DataParams:
    temps: List[int] = field(default_factory=lambda: [45])
    aging_types: List[str] = field(default_factory=lambda: ['z'])
    testpoints: np.ndarray = field(default_factory=lambda: np.arange(0, 5, 1))
    reference_performance_test: List[str] = field(default_factory= lambda: ['ET'])
    cells: List[int] = field(default_factory= lambda: list(np.arange(0, 2, 1)))
    stages: List[int] = field(default_factory=lambda: [1, 2])

    time_res: float = 0.5
    num_points: int = 10
    add_feature_cols: bool = True

@dataclass
class NeuralNetworkParams:
    learning_rate : float = 1e-3
    n_epochs: int = 5000
    batch_size: int = 64
    hidden_dim: int = 32
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
