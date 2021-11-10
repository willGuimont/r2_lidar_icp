import numpy as np

from lidar_icp.utils import point_to_homogeneous


class PointCloud:
    def __init__(self):
        self.features = np.zeros((0, 0))  # DxN (D = dimension, N = nb points)
        self.descriptors = np.zeros((0, 0))  # MxN (M = descriptor length)

    def __copy__(self):
        point_cloud = PointCloud()
        point_cloud.features = np.copy(self.features)
        point_cloud.descriptors = np.copy(self.descriptors)
        return point_cloud

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
