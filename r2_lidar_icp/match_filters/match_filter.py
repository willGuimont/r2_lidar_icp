from abc import ABC, abstractmethod

from r2_lidar_icp.matchers.matches import Matches
from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class MatchFilter(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def compute_mask(self, pc: PointCloud, matches: Matches):
        ...

    def filter_matches(self, pc: PointCloud, matches: Matches):
        mask = self.compute_mask(pc, matches)
        matches.apply_mask(mask)
        pc.apply_mask(mask)
