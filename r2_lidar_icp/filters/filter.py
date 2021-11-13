from abc import ABC, abstractmethod

from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class Filter(ABC):
    def __init__(self):
        """
        Filters out points
        """
        ...

    @abstractmethod
    def keep_indices(self, pc: PointCloud):
        ...

    def filter(self, pc: PointCloud):
        ...
