import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.point_cloud import PointCloud


class PolarDescriptor(Descriptor):
    """
    Descriptor that adds polar coordinates to the point cloud in the xy plane.
    Adds a descriptor of shape (2, n), for each n points (r, theta).
    """
    name = 'PolarDescriptor'
    RadiusIndex = 0
    AngleIndex = 1

    def compute_descriptor(self, pc: PointCloud):
        xs = pc.features[0, :]
        ys = pc.features[1, :]

        r = np.sqrt(xs ** 2 + ys ** 2)
        angles = np.arctan2(ys, xs)

        pc.add_descriptor(self, np.stack((r, angles)))
