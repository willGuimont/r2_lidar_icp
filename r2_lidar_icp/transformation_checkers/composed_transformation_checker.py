from r2_lidar_icp.transformation_checkers.transformation_checker import TransformationChecker


class ComposedTransformationChecker(TransformationChecker):
    def __init__(self, transformation_checkers: [TransformationChecker]):
        """
        Composed transformation checker. Will stop the ICP algorithm if any of the transformation checkers returns True.
        :param transformation_checkers: List of transformation checkers
        """
        self.checkers = transformation_checkers

    def begin(self):
        for c in self.checkers:
            c.begin()

    def _is_finished_check(self, error: float) -> bool:
        for c in self.checkers:
            if c.is_finished(error):
                return True
        return False
