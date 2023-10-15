import unittest

import numpy as np

from r2_lidar_icp.descriptors.normal_descriptor import NormalDescriptor
from r2_lidar_icp.descriptors.polar_descriptor import PolarDescriptor
from r2_lidar_icp.matchers.kdtree_matcher import KDTreeMatcher
from r2_lidar_icp.point_cloud import PointCloud


class DescriptorTest(unittest.TestCase):
    def test_normal_descriptor(self):
        # grid on the plane x + y + z = 0, with normal (1, 1, 1)
        x = np.linspace(-1, 1, 5)
        y = np.linspace(-1, 1, 5)
        xx, yy = np.meshgrid(x, y)
        zz = -xx - yy
        scan = np.stack((xx.flatten(), yy.flatten(), zz.flatten()))

        descriptor = NormalDescriptor(knn=5, matcher_cls=KDTreeMatcher)
        point_cloud = PointCloud.from_cartesian_scan(scan)
        descriptor.compute_descriptor(point_cloud)

        self.assertEqual('NormalDescriptor', descriptor.name)
        self.assertEqual(5, descriptor.knn)

        normals = point_cloud.descriptors[descriptor.name]
        expected_normal = np.array([[1, 1, 1]]).T.repeat(25, 1) / np.sqrt(3)

        self.assertEqual((3, 25), normals.shape)
        self.assertTrue(np.allclose(expected_normal, abs(normals), atol=1e-5))

    def test_polar_descriptor(self):
        descriptor = PolarDescriptor()
        scan = np.array([[1, 2, 3, 4, 5], [3, 4, 5, 6, 7]])

        point_cloud = PointCloud.from_cartesian_scan(scan)
        descriptor.compute_descriptor(point_cloud)

        self.assertEqual('PolarDescriptor', descriptor.name)

        polar = point_cloud.descriptors[descriptor.name]
        expected_angle = np.arctan2(scan[1, :], scan[0, :])
        expected_radius = np.sqrt(scan[0, :] ** 2 + scan[1, :] ** 2)

        self.assertEqual((2, 5), polar.shape)
        self.assertTrue(np.allclose(expected_angle, polar[1, :], atol=1e-5))
        self.assertTrue(np.allclose(expected_radius, polar[0, :], atol=1e-5))
