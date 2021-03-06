import numpy as np

from r2_lidar_icp.matchers.matches import Matches
from r2_lidar_icp.point_cloud.point_cloud import PointCloud
from r2_lidar_icp.transformation_checkers.transformation_checker import TransformationChecker


class ErrorDeltaTransformationChecker(TransformationChecker):
    def __init__(self, min_error_delta: float):
        super().__init__()
        self.min_error_delta = min_error_delta
        self.previous_error = 0

    def begin(self):
        self.previous_error = 0

    def is_finished(self, reference: PointCloud, reading: PointCloud, matches: Matches) -> bool:
        mean_error = np.average(matches.distances)
        if abs(self.previous_error - mean_error) < self.min_error_delta:
            return True
        self.previous_error = mean_error
        return False
