from typing import Dict

import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.filters.filter import Filter
from r2_lidar_icp.point_cloud import PointCloud


class ComposedFilter(Filter):
    def __init__(self, filters: [Filter]):
        """
        Compose multiple filters into one.
        :param filters: List of filters to compose.
        """
        self.filters = filters

    def _compute_mask(self, pc: PointCloud, descriptors: Dict[str, Descriptor]) -> np.ndarray:
        ...

    def filter(self, pc: PointCloud, descriptors: Dict[str, Descriptor]):
        for f in self.filters:
            f.filter(pc, descriptors)
