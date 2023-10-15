from typing import Dict

import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.filters.filter import Filter
from r2_lidar_icp.point_cloud import PointCloud


class InvertFilter(Filter):
    def __init__(self, base_filter: Filter):
        """
        Invert the mask of a filter.
        :param base_filter: Filter to invert.
        """
        self.base_filter = base_filter

    def _compute_mask(self, pc: PointCloud, descriptors: Dict[str, Descriptor]) -> np.ndarray:
        return np.bitwise_not(self.base_filter._compute_mask(pc, descriptors))
