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

    @abstractmethod
    def query(self, pc: PointCloud, knn: int) -> (np.ndarray, np.ndarray):
        """
        Query the matcher for the k nearest neighbors of each point in a reference point cloud.
        :param pc: Point cloud.
        :param knn: Number of nearest neighbors to query.
        :return: (distances, indices)
                 distances is of shape (num_points, knn), where distances[i, j] is the distance between the ith point
                    in the query point cloud and its jth nearest neighbor in the reference point cloud.
                 indices is of shape (num_points, knn), where indices[i, j] is the index of the jth nearest neighbor
                    of the ith point in the query point cloud in the reference point cloud.
        """

    def match(self, pc: PointCloud, knn: int = 1) -> Matches:
        """
        Find matches between the point cloud and the reference point cloud.
        :param pc: Point cloud.
        :param knn: Number of nearest neighbors to query.
        :return: Matches between the point cloud and the reference point cloud.
        """
        dist, indices = self.query(pc, knn=knn)
        from_indices = np.arange(pc.num_points).reshape(-1, 1)
        return Matches(dist, from_indices, indices)
