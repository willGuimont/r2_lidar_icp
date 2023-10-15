from typing import Dict

import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.filters.filter import Filter
from r2_lidar_icp.point_cloud import PointCloud


class BoxFilter(Filter):
    def __init__(self, min_corner: np.ndarray, max_corner: np.ndarray):
        """
        Filter out points outside a box defined by two points.
        :param min_corner: Minimum point of the box of shape (dim,).
        :param max_corner: Maximum point of the box of shape (dim,).
        """
        self.min_corner = min_corner.reshape((-1, 1))
        self.max_corner = max_corner.reshape((-1, 1))

    def _compute_mask(self, pc: PointCloud, descriptors: Dict[str, Descriptor]) -> np.ndarray:
        position = pc.features[:3, :]
        in_range = np.bitwise_and(self.min_corner < position, position < self.max_corner).all(axis=0)
        return in_range
