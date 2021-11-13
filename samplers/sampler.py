from abc import ABC, abstractmethod

from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class Sampler(ABC):
    def __init__(self):
        """
        Subsample a PointCloud
        """
        ...

    def sample(self, pc: PointCloud):
        ...