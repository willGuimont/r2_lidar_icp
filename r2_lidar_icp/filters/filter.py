from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.point_cloud import PointCloud


class Filter(ABC):
    """
    Base class for filters. Filters will remove points from the point cloud based on some criteria.
    """

    @abstractmethod
    def _compute_mask(self, pc: PointCloud, descriptors: Dict[str, Descriptor]) -> np.ndarray:
        """
        Compute the mask to apply to the point cloud.
        The mask should be a boolean array with the same length as the point cloud.
        :param pc: Point cloud to compute the mask for.
        :param descriptors: Dictionary of descriptors for the point cloud.
        :return: Mask to apply to the point cloud (boolean array with same length as point cloud).
        """

    def filter(self, pc: PointCloud, descriptors: Dict[str, Descriptor]):
        """
        Apply the filter to the point cloud.
        :param pc: Point cloud to apply the filter to.
        :param descriptors: Dictionary of descriptors for the point cloud.
        :return: None
        """
        mask = self._compute_mask(pc, descriptors)
        pc.apply_mask(mask)

    def __and__(self, other):
        from r2_lidar_icp.filters.composed_filter import ComposedFilter
        return ComposedFilter([self, other])

    def __invert__(self):
        from r2_lidar_icp.filters.invert_filter import InvertFilter
        return InvertFilter(self)
