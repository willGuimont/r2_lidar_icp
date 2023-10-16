import unittest

import numpy as np

from r2_lidar_icp.descriptors.polar_descriptor import PolarDescriptor
from r2_lidar_icp.filters.box_filter import BoxFilter
from r2_lidar_icp.filters.composed_filter import ComposedFilter
from r2_lidar_icp.filters.furthest_point_sampling_filter import FurthestPointSamplingFilter
from r2_lidar_icp.filters.identity_filter import IdentityFilter
from r2_lidar_icp.filters.invert_filter import InvertFilter
from r2_lidar_icp.filters.radii_filter import RadiiFilter
from r2_lidar_icp.point_cloud import PointCloud


class FilterTest(unittest.TestCase):
    def test_box_filter(self):
        points = np.array(
            [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [-0.5, 0, 0.5], [0, -10, 0], [0, 0, 5], [12, 0, -1], ]).T
        pc = PointCloud(points)

        min_corner = np.array([-1, -1, -1])
        max_corner = np.array([1, 1, 1])

        box_filter = BoxFilter(min_corner, max_corner)
        box_filter.filter(pc, {})

        self.assertEqual(4, pc.features.shape[1])
        self.assertTrue(np.allclose(points[:, :4], pc.features))

    def test_composed_filter(self):
        points = np.array(
            [[0.5, 0.5, 0.5], [0.5, 0.75, 0.1], [0, 0.5, 0], [-0.5, 0, 0.5], [0, -10, 0], [0, 0, 5], [12, 0, -1], ]).T
        pc = PointCloud(points)

        min_corner_1 = np.array([-1, -1, -1])
        max_corner_1 = np.array([1, 1, 1])
        min_corner_2 = np.array([0, 0, 0])
        max_corner_2 = np.array([2, 2, 2])

        box_filter_1 = BoxFilter(min_corner_1, max_corner_1)
        box_filter_2 = BoxFilter(min_corner_2, max_corner_2)
        composed_filter = ComposedFilter([box_filter_1, box_filter_2])
        composed_filter.filter(pc, {})

        self.assertEqual(2, pc.features.shape[1])
        self.assertTrue(np.allclose(points[:, :2], pc.features))

    def test_furthest_point_sampling_filter(self):
        points = np.random.randint(0, 100, size=(3, 50))
        pc = PointCloud(points)

        furthest_point_sampling_filter = FurthestPointSamplingFilter(10)
        furthest_point_sampling_filter.filter(pc, {})

        self.assertEqual(10, pc.features.shape[1])

    def test_identity_filter(self):
        points = np.random.randint(0, 100, size=(3, 50))
        pc = PointCloud(points)

        id_filter = IdentityFilter()
        id_filter.filter(pc, {})

        self.assertTrue(np.allclose(points, pc.features))

    def test_invert_filter(self):
        points = np.array(
            [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [-0.5, 0, 0.5], [0, -10, 0], [0, 0, 5], [12, 0, -1], ]).T
        pc = PointCloud(points)

        min_corner = np.array([-1, -1, -1])
        max_corner = np.array([1, 1, 1])

        box_filter = BoxFilter(min_corner, max_corner)
        invert_filter = InvertFilter(box_filter)
        invert_filter.filter(pc, {})

        self.assertEqual(3, pc.features.shape[1])
        self.assertTrue(np.allclose(points[:, 4:], pc.features))

    def test_radii_filter(self):
        points = np.array(
            [[3, 4, 0], [3, 4, 10], [2, 1, 1], [100, 2, 3]]).T
        pc = PointCloud(points)
        descriptors = dict(PolarDescriptor=PolarDescriptor())

        radii_filter = RadiiFilter(2, 10)
        radii_filter.filter(pc, descriptors)

        self.assertEqual(3, pc.features.shape[1])
        self.assertTrue(np.allclose(points[:, :3], pc.features))
