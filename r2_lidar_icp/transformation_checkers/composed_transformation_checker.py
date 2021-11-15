from typing import List

from r2_lidar_icp.point_cloud.point_cloud import PointCloud
from r2_lidar_icp.transformation_checkers.transformation_checker import TransformationChecker


class ComposedTransformationChecker(TransformationChecker):
    def __init__(self, transformation_checkers: List[TransformationChecker]):
        super().__init__()
        self.checkers = transformation_checkers

    def begin(self):
        for c in self.checkers:
            c.begin()

    def is_finished(self, reference: PointCloud, reading: PointCloud) -> bool:
        for c in self.checkers:
            if c.is_finished(reference, reading):
                return True
        return False
