import unittest

from r2_lidar_icp.transformation_checkers.composed_transformation_checker import ComposedTransformationChecker
from r2_lidar_icp.transformation_checkers.error_delta_transformation_checker import ErrorDeltaTransformationChecker
from r2_lidar_icp.transformation_checkers.max_iteration_transformation_checker import MaxIterationTransformationChecker
from r2_lidar_icp.transformation_checkers.relative_error_delta_transformation_checker import \
    RelativeErrorDeltaTransformationChecker


class TransformationCheckerTest(unittest.TestCase):
    def test_max_iteration_checker(self):
        checker = MaxIterationTransformationChecker(100)
        for i in range(99):
            self.assertFalse(checker.is_finished(0))
        self.assertTrue(checker.is_finished(0))

    def test_composed_checker(self):
        iter_1 = MaxIterationTransformationChecker(100)
        iter_2 = MaxIterationTransformationChecker(50)
        checker = ComposedTransformationChecker([iter_1, iter_2])

        for i in range(49):
            self.assertFalse(checker.is_finished(0))
        self.assertTrue(checker.is_finished(0))

    def test_error_delta_checker(self):
        checker = ErrorDeltaTransformationChecker(0.1)
        error = 100
        delta = 1
        for i in range(99):
            error -= delta
            self.assertFalse(checker.is_finished(error))

        small_delta = 0.01
        error -= small_delta
        self.assertTrue(checker.is_finished(error))

    def test_rel_error_checker(self):
        checker = RelativeErrorDeltaTransformationChecker(0.1)
        error = 10000
        relative_change = 0.6
        for i in range(99):
            error -= error * relative_change
            self.assertFalse(checker.is_finished(error))

        small_relative_change = 0.01
        error -= error * small_relative_change
        self.assertTrue(checker.is_finished(error))
