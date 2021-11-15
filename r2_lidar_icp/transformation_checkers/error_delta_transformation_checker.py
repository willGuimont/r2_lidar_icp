from r2_lidar_icp.point_cloud.point_cloud import PointCloud
from r2_lidar_icp.transformation_checkers.transformation_checker import TransformationChecker


class ErrorDeltaTransformationChecker(TransformationChecker):
    def __init__(self, min_error_delta: float):
        super().__init__()
        self.min_error_delta = min_error_delta
        self.last_error = 0

    def begin(self):
        self.last_error = 0

    def is_finished(self, reference: PointCloud, reading: PointCloud) -> bool:
        match
