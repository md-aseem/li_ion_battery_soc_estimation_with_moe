from dataclasses import dataclass, field
import numpy as np
from typing import List

@dataclass
class DataParams:
    temps: List[int] = field(default_factory=lambda: [45])
    aging_types: List[str] = field(default_factory=lambda: ['z'])
    testpoints: np.ndarray = field(default_factory=lambda: np.arange(0, 2, 1))
    reference_performance_test: List[str] = field(default_factory= lambda: ['ET'])
    cells: List[int] = field(default_factory= lambda: list(np.arange(0, 2, 1)))
    stages: List[int] = field(default_factory=lambda: [1, 2])

    time_res: float = 0.1
    num_points: int = 10

