from r2_lidar_icp.transformation_checkers.transformation_checker import TransformationChecker


class MaxIterationTransformationChecker(TransformationChecker):
    def __init__(self, max_iter: int):
        """
        Transformation checker that stops the ICP algorithm after a given number of iterations.
        :param max_iter: Maximum number of iterations
        """
        self.max_iter = max_iter
        self.num_iter = 0

    def begin(self):
        self.num_iter = 0

    def _is_finished_check(self, error: float) -> bool:
        self.num_iter += 1
        return self.num_iter >= self.max_iter
