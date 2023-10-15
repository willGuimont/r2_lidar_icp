import numpy as np
from scipy.spatial import KDTree

from r2_lidar_icp.matchers.matcher import Matcher
from r2_lidar_icp.point_cloud import PointCloud


class KDTreeMatcher(Matcher):
    def __init__(self, tree: KDTree):
        """
        Matcher based on a KDTree.
        :param tree:
        """
        self.tree = tree

    @staticmethod
    def make_matcher(reference: PointCloud) -> Matcher:
        return KDTreeMatcher(KDTree(reference.features.T))

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
        distances, indices = self.tree.query(pc.features.T, k=knn)
        distances = distances.reshape(-1, knn)
        indices = indices.reshape(-1, knn)
        return distances, indices
