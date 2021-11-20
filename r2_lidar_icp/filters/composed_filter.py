from typing import List, Dict

import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.filters.filter import Filter
from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class ComposedFilter(Filter):
    def __init__(self, filters: List[Filter]):
        self.filters = filters

    def compute_mask(self, pc: PointCloud, descriptors: Dict[str, Descriptor]) -> np.ndarray:
        ...

    def filter(self, pc: PointCloud, descriptors: Dict[str, Descriptor]):
        for f in self.filters:
            f.filter(pc, descriptors)
