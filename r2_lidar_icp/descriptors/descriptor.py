from abc import ABC, abstractmethod

from r2_lidar_icp.point_cloud import PointCloud


class Descriptor(ABC):
    """
    Base class for descriptors. Descriptors will add features to the point cloud.
    """
    name = 'Descriptor'

    @abstractmethod
    def compute_descriptor(self, pc: PointCloud):
        """
        Compute the descriptor and add it to the point cloud.
        :param pc: Point cloud to compute the descriptor for.
        :return: None
        """
