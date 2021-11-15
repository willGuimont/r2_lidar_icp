from abc import abstractmethod, ABC

from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class TransformationChecker(ABC):
    def __init__(self):
        ...

    def begin(self):
        ...

    @abstractmethod
    def is_finished(self, reference: PointCloud, reading: PointCloud) -> bool:
        ...
