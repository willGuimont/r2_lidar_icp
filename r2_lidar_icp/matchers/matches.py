from dataclasses import dataclass

import numpy as np


@dataclass
class Matches:
    def __init__(self, distances: np.ndarray, indices: np.ndarray):
        self.distances = distances
        self.indices = indices

    def apply_mask(self, mask: np.ndarray):
        self.distances = self.distances[mask]
        self.indices = self.indices[mask]
