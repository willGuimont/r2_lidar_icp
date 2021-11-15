from abc import abstractmethod, ABC
from typing import Dict

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.matchers.matches import Matches
from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class Minimizer(ABC):
    @abstractmethod
    def find_transformation(self,
                            reference: PointCloud,
                            reading: PointCloud,
                            matches: Matches,
                            descriptors: Dict[str, Descriptor]):
        ...
