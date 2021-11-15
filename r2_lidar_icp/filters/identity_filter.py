from typing import Dict

import numpy as np

from backup.point_cloud import PointCloud
from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.filters.filter import Filter


class IdentityFilter(Filter):
    def compute_mask(self, pc: PointCloud, descriptors: Dict[str, Descriptor]) -> np.ndarray:
        return np.full(pc.features.shape[1], True)
