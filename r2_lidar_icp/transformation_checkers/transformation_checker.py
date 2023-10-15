from abc import abstractmethod, ABC

from r2_lidar_icp.matchers.matches import Matches
from r2_lidar_icp.point_cloud import PointCloud


# TODO test this
class TransformationChecker(ABC):
    """
    Base class for transformation checkers.
    Transformation checkers are used to check if the ICP algorithm should stop.
    """

    def begin(self):
        """
        Called before the ICP algorithm starts.
        Should be used to initialize the checker.
        :return: None
        """

    @abstractmethod
    def is_finished(self, point_cloud: PointCloud, reference: PointCloud, matches: Matches) -> bool:
        """
        Called after each iteration of the ICP algorithm.
        Should return True if the ICP algorithm should stop.
        :param point_cloud: Point to register
        :param reference: Reference point cloud
        :param matches: Matches between point_cloud and reference
        :return: True if the ICP algorithm should stop, False otherwise
        """
