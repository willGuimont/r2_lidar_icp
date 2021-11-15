from abc import ABC, abstractmethod

from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class Descriptor(ABC):
    name = 'Descriptor'

    @abstractmethod
    def compute_descriptor(self, pc: PointCloud):
        ...
