from dataclasses import dataclass

import numpy as np


@dataclass
class Matches:
    def __init__(self, distances: np.ndarray, indices: np.ndarray):
        """
        Matches between two point clouds.
        :param distances: distances[i] is the distance between the ith point in the query point cloud and its nearest
            neighbor in the reference point cloud.
        :param indices: indices[i] is the index of the nearest neighbor of the ith point in the query point cloud in
            the reference point cloud.
        """
        self.distances = distances
        self.indices = indices

    def apply_mask(self, mask: np.ndarray):
        """
        Apply a mask to the matches.
        :param mask: Mask to apply.
        :return: None
        """
        self.distances = self.distances[mask]
        self.indices = self.indices[mask]
