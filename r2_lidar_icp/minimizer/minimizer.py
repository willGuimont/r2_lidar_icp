from abc import abstractmethod, ABC
from typing import Dict

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.matchers.matches import Matches
from r2_lidar_icp.point_cloud import PointCloud


# TODO test
class Minimizer(ABC):
    """
    Base class for minimizers. Minimizers will find the transformation between a point
    cloud and a reference point cloud.
    """

    @abstractmethod
    def find_transformation(self, point_cloud: PointCloud, reference: PointCloud, matches: Matches,
                            descriptors: Dict[str, Descriptor]):
        """
        Find the transformation between the point cloud and the reference point cloud.
        :param point_cloud: New point cloud to register to the reference point cloud.
        :param reference: Reference point cloud
        :param matches: Matches between the point cloud and the reference point cloud.
        :param descriptors: Descriptors of the point cloud.
        :return: Transformation between the point cloud and the reference point cloud in homogeneous coordinates.
        """
