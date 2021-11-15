from abc import ABC, abstractmethod

import numpy as np

from r2_lidar_icp.matchers.matches import Matches
from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class Matcher(ABC):
    @abstractmethod
    def query(self, pc: PointCloud, knn: int) -> (np.ndarray, np.ndarray):
        ...

    def match(self, pc: PointCloud) -> Matches:
        dist, indices = self.query(pc, knn=1)
        return Matches(dist, indices)


class MatcherType(ABC):
    @abstractmethod
    def make_matcher(self, reference: PointCloud) -> Matcher:
        ...
