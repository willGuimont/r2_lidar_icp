from typing import Dict

import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.filters.filter import Filter
from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class BoxFilter(Filter):
    def __init__(self, min_x: float, min_y: float, max_x: float, max_y: float):
        """
        Filters out points outside of a box
        """
        super().__init__()
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    def compute_mask(self, pc: PointCloud, descriptors: Dict[str, Descriptor]) -> np.ndarray:
        xs = pc.features[0, :]
        ys = pc.features[1, :]
        in_x_range = np.bitwise_and(self.min_x < xs, xs < self.max_x)
        in_y_range = np.bitwise_and(self.min_y < ys, ys < self.max_y)
        return np.bitwise_and(in_x_range, in_y_range)
