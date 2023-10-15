import unittest

import numpy as np

from r2_lidar_icp.matchers.kdtree_matcher import KDTreeMatcher
from r2_lidar_icp.point_cloud import PointCloud


class MatcherTest(unittest.TestCase):
    def test_kdtree_matcher(self):
        points = np.array([[1, 2, 3, 4, 5], [3, 4, 5, 6, 7]])
        reference = PointCloud(points)
        matcher = KDTreeMatcher.make_matcher(reference)

        matches = matcher.query(reference, knn=2)
        distances, indices = matches

        self.assertEqual((5, 2), distances.shape)
        self.assertEqual((5, 2), indices.shape)
