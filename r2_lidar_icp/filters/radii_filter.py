from typing import Dict

import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.descriptors.polar_descriptor import PolarDescriptor
from r2_lidar_icp.filters.filter import Filter
from r2_lidar_icp.point_cloud import PointCloud


class RadiiFilter(Filter):
    def __init__(self, min_r: float, max_r: float):
        """
        Filter points inside a range of radii in the xy plane.
        :param min_r: Minimum distance to keep
        :param max_r: Maximum distance to keep
        """
        self.min_r = min_r
        self.max_r = max_r

    def _compute_mask(self, pc: PointCloud, descriptors: Dict[str, Descriptor], ) -> np.ndarray:
        polar = pc.get_descriptor(PolarDescriptor.name, descriptors)
        r = polar[PolarDescriptor.RadiusIndex, :]
        return np.bitwise_and(self.min_r < r, r < self.max_r)
