from typing import Dict

import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.filters.filter import Filter
from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class InvertFilter(Filter):
    def __init__(self, filter: Filter):
        """
        Inverts a filter
        :param filter: filter to invert
        """
        super().__init__()
        self.base_filter = filter

    def compute_mask(self, pc: PointCloud, descriptors: Dict[str, Descriptor]) -> np.ndarray:
        return np.bitwise_not(self.base_filter.compute_mask(pc, descriptors))
