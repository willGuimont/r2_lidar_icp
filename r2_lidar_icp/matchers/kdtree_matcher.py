import numpy as np
from scipy.spatial import KDTree

from r2_lidar_icp.matchers.matcher import Matcher, MatcherType
from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class KDTreeMatcher(Matcher):
    def __init__(self, tree: KDTree):
        self.tree = tree

    def query(self, pc: PointCloud, knn: int) -> (np.ndarray, np.ndarray):
        return self.tree.query(pc.features.T, k=knn)


class KDTreeMatcherType(MatcherType):
    def make_matcher(self, reference: PointCloud) -> Matcher:
        return KDTreeMatcher(KDTree(reference.features.T))
