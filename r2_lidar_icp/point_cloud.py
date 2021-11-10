from copy import copy
from typing import Union

import numpy as np

from r2_lidar_icp.sampling import furthest_point_sampling
from r2_lidar_icp.utils import point_to_homogeneous, homogeneous_to_points


class PointCloud:
    def __init__(self):
        self.features = np.zeros((0, 0))  # DxN (D = dimension, N = nb points)
        self.descriptors = np.zeros((0, 0))  # MxN (M = descriptor length)

    def __copy__(self):
        point_cloud = PointCloud()
        point_cloud.features = np.copy(self.features)
        point_cloud.descriptors = np.copy(self.descriptors)
        return point_cloud

    @property
    def features_as_points(self):
        return homogeneous_to_points(self.features)

    def __add__(self, other: Union['PointCloud', np.ndarray]):
        if type(other) == PointCloud:
            points_to_add = other.features
        elif type(other) == np.ndarray:
            points_to_add = other
        else:
            raise ValueError(f'Cannot add PointCloud and {type(other)}')

        pc = copy(self)
        pc.features = np.concatenate((pc.features, np.copy(points_to_add)), axis=1)
        pc.descriptors = np.zeros((0, 0))  # np.concatenate((pc.descriptors, np.copy(other.descriptors)), axis=1)
        return pc

    def subsample(self, points_to_keep: int, skip_initial: bool = True):
        indices, _ = furthest_point_sampling(self.features_as_points, points_to_keep, skip_initial=skip_initial)
        pc = copy(self)

        pc.features = self.features[:, indices]
        pc.descriptors = self.descriptors[:, indices]

        return pc

    @staticmethod
    def from_scan(scan) -> 'PointCloud':
        scan = np.array(scan)

        qualities, angles, distances = scan[:, 0], scan[:, 1], scan[:, 2]
        angles = np.deg2rad(angles)

        xs = np.cos(angles) * distances
        ys = np.sin(angles) * distances

        features = np.stack([xs, ys])
        features = point_to_homogeneous(features)

        pc = PointCloud()
        pc.features = features

        return pc
