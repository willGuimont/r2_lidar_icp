from dataclasses import dataclass

import numpy as np


@dataclass
class Matches:
    def __init__(self, distances: np.ndarray, from_indices: np.ndarray, indices: np.ndarray):
        """
        Matches between two point clouds.
        :param distances: array of shape (num_points, num_neighbors), of distances between each point in the query
        :param from_indices: array of shape (num_points, 1), of indices of the points in the query point cloud
        :param indices: array of shape (num_points, num_neighbors), of indices of the neighbors in the reference point
        """
        self.distances = distances
        self.from_indices = from_indices
        self.indices = indices

    def apply_mask(self, mask: np.ndarray):
        """
        Apply a mask to the matches.
        :param mask: Mask to apply.
        :return: None
        """
        m = mask[:, 0]
        self.distances = self.distances[m]
        self.from_indices = self.from_indices[m]
        self.indices = self.indices[m]

    @property
    def num_matches(self):
        return self.from_indices.shape[0]

    @property
    def best_distances(self):
        return self.distances[:, [0]]

    @property
    def best_indices(self):
        return self.indices[:, [0]]
