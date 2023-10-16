import unittest

import numpy as np

from r2_lidar_icp.match_filters.identity_match_filter import IdentityMatchFilter
from r2_lidar_icp.match_filters.max_distance_match_filter import MaxDistanceMatchFilter
from r2_lidar_icp.matchers.matches import Matches
from r2_lidar_icp.point_cloud import PointCloud


class MatchFilterTest(unittest.TestCase):
    def test_match_filter(self):
        points = np.random.randint(0, 100, (2, 100))
        pc = PointCloud(points)
        distances = np.arange(100).reshape((100, 1))
        from_indices = np.arange(100).reshape((100, 1))
        indices = np.arange(100).reshape((100, 1))
        matches = Matches(distances, from_indices, indices)

        match_filter = MaxDistanceMatchFilter(50)
        match_filter.filter_matches(pc, matches)

        self.assertEqual(50, matches.distances.shape[0])
        self.assertEqual(50, matches.from_indices.shape[0])
        self.assertEqual(50, matches.indices.shape[0])

    def test_identity_match_filter(self):
        points = np.random.randint(0, 100, (2, 100))
        pc = PointCloud(points)
        distances = np.arange(100).reshape((100, 1))
        from_indices = np.arange(100).reshape((100, 1))
        indices = np.arange(100).reshape((100, 1))
        matches = Matches(distances, from_indices, indices)

        match_filter = IdentityMatchFilter()
        match_filter.filter_matches(pc, matches)

        self.assertEqual(100, matches.distances.shape[0])
        self.assertEqual(100, matches.indices.shape[0])

    def test_max_distance_match_filter(self):
        points = np.random.randint(0, 100, (2, 100))
        pc = PointCloud(points)
        distances = np.arange(100).reshape((100, 1))
        from_indices = np.arange(100).reshape((100, 1))
        indices = np.arange(100).reshape((100, 1))
        matches = Matches(distances, from_indices, indices)

        match_filter = MaxDistanceMatchFilter(50)
        match_filter.filter_matches(pc, matches)

        self.assertEqual(50, matches.distances.shape[0])
        self.assertEqual(50, matches.indices.shape[0])
