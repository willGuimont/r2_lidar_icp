import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class PolarDescriptor(Descriptor):
    """
    Descriptor that adds polar coordinates to the point cloud.
    """
    name = 'PolarDescriptor'
    RadiusIndex = 0
    AngleIndex = 0

    def compute_descriptor(self, pc: PointCloud):
        xs = pc.features[0, :]
        ys = pc.features[1, :]

        r = np.sqrt(xs ** 2 + ys ** 2)
        angles = np.arctan2(ys, xs)

        pc.add_descriptor(self, np.stack((r, angles)))
