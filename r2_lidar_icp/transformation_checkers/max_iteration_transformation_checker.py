from r2_lidar_icp.matchers.matches import Matches
from r2_lidar_icp.point_cloud.point_cloud import PointCloud
from r2_lidar_icp.transformation_checkers.transformation_checker import TransformationChecker


class MaxIterationTransformationChecker(TransformationChecker):
    def __init__(self, max_iter: int):
        self.max_iter = max_iter
        self.num_iter = 0

    def begin(self):
        self.num_iter = 0

    def is_finished(self, reference: PointCloud, reading: PointCloud, matches: Matches) -> bool:
        self.num_iter += 1
        return self.num_iter >= self.max_iter
