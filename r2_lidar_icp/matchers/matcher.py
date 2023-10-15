from abc import ABC, abstractmethod

import numpy as np

from r2_lidar_icp.matchers.matches import Matches
from r2_lidar_icp.point_cloud import PointCloud


class Matcher(ABC):
    """
    Base class for matchers. Matchers will find matches between a point cloud and a reference point cloud.
    """

    @staticmethod
    @abstractmethod
    def make_matcher(reference: PointCloud) -> 'Matcher':
        """
        Make a matcher for the given reference point cloud.
        :param reference: Reference point cloud.
        :return: Matcher.
        """
        ...

    @abstractmethod
    def query(self, pc: PointCloud, knn: int) -> (np.ndarray, np.ndarray):
        """
        Query the matcher for the nearest neighbors of each point in a reference point cloud.
        :param pc: Point cloud.
        :param knn: Number of nearest neighbors to query.
        :return: Tuple of distances and indices of the nearest neighbors.
        """
        ...

    def match(self, pc: PointCloud) -> Matches:
        """
        Find matches between the point cloud and the reference point cloud.
        :param pc:
        :return:
        """
        dist, indices = self.query(pc, knn=1)
        return Matches(dist, indices)
