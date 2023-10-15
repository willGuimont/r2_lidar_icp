import unittest

import numpy as np

from r2_lidar_icp.matchers.kdtree_matcher import KDTreeMatcher
from r2_lidar_icp.point_cloud import PointCloud


class MatcherTest(unittest.TestCase):
    def test_kdtree_matcher(self):
        ref_points = np.array([[1, 2, 3, 4, 5], [3, 4, 5, 6, 7]])
        points = ref_points[:, [0, 2, 4]]

        pc = PointCloud(points)
        reference = PointCloud(ref_points)
        matcher = KDTreeMatcher.make_matcher(reference)

        matches = matcher.match(pc, knn=2)
        distances = matches.distances
        from_indices = matches.from_indices
        indices = matches.indices

        self.assertEqual((3, 2), distances.shape)
        self.assertEqual((3, 1), from_indices.shape)
        self.assertEqual((3, 2), indices.shape)
        self.assertEqual((3, 1), matches.best_indices.shape)
        self.assertEqual((3, 1), matches.best_distances.shape)
