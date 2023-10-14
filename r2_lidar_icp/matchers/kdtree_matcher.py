import numpy as np
from scipy.spatial import KDTree

from r2_lidar_icp.matchers.matcher import Matcher, MatcherType
from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class KDTreeMatcher(Matcher):
    def __init__(self, tree: KDTree):
        """
        Matcher based on a KDTree.
        :param tree:
        """
        self.tree = tree

    def _query(self, pc: PointCloud, knn: int) -> (np.ndarray, np.ndarray):
        """
        Query the matcher for the k nearest neighbors of each point in a reference point cloud.
        :param pc: Point cloud.
        :param knn: Number of nearest neighbors to query.
        :return: Tuple of distances and indices of the nearest neighbors.
        """
        return self.tree.query(pc.features.T, k=knn)


class KDTreeMatcherType(MatcherType):
    def make_matcher(self, reference: PointCloud) -> Matcher:
        return KDTreeMatcher(KDTree(reference.features.T))
