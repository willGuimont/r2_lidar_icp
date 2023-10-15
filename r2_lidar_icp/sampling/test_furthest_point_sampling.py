import unittest

import numpy as np

from r2_lidar_icp.sampling.furthest_point_sampling import furthest_point_sampling


class FurthestPointSamplingTest(unittest.TestCase):
    def test_furthest_point_sampling(self):
        points = np.random.randint(0, 100, size=(3, 50))

        distance, indices = furthest_point_sampling(points, 10)
        reduced = points[:, indices]

        self.assertEqual(10, reduced.shape[1])
