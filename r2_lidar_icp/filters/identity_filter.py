from typing import Dict

import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.filters.filter import Filter
from r2_lidar_icp.point_cloud import PointCloud


class IdentityFilter(Filter):
    """
    Identity filter. Does not filter any points.
    """

    def _compute_mask(self, pc: PointCloud, descriptors: Dict[str, Descriptor]) -> np.ndarray:
        return np.full(pc.features.shape[1], True)
