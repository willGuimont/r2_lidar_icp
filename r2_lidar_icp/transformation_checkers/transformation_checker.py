from abc import abstractmethod, ABC

from r2_lidar_icp.matchers.matches import Matches
from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class TransformationChecker(ABC):
    def begin(self):
        ...

    @abstractmethod
    def is_finished(self, reference: PointCloud, reading: PointCloud, matches: Matches) -> bool:
        ...
