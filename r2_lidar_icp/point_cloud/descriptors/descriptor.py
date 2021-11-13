from abc import ABC, abstractmethod

from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class Descriptor(ABC):
    def __init__(self):
        """
        A descriptor adds its computed descriptor to the PointCloud object
        """
        ...

    @abstractmethod
    def compute_descriptor(self, pc: PointCloud):
        ...
