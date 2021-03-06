from copy import copy
from typing import Optional, Dict

import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.descriptors.normal_descriptor import NormalDescriptor
from r2_lidar_icp.descriptors.polar_descriptor import PolarDescriptor
from r2_lidar_icp.filters.filter import Filter
from r2_lidar_icp.filters.identity_filter import IdentityFilter
from r2_lidar_icp.match_filters.identity_match_filter import IdentityMatchFilter
from r2_lidar_icp.match_filters.match_filter import MatchFilter
from r2_lidar_icp.matchers.kdtree_matcher import KDTreeMatcherType
from r2_lidar_icp.matchers.matcher import MatcherType
from r2_lidar_icp.minimizer.minimizer import Minimizer
from r2_lidar_icp.minimizer.point_to_plane_minimizer import PointToPlaneMinimizer
from r2_lidar_icp.point_cloud.point_cloud import PointCloud
from r2_lidar_icp.transformation_checkers.max_iteration_transformation_checker import MaxIterationTransformationChecker
from r2_lidar_icp.transformation_checkers.transformation_checker import TransformationChecker


class ICPBuilder:
    def __init__(self):
        self.descriptors = dict(PolarDescriptor=PolarDescriptor(),
                                NormalDescriptor=NormalDescriptor(knn=20, matcher_type=KDTreeMatcherType()))
        self.reference_preprocessing = IdentityFilter()
        self.reading_preprocessing = IdentityFilter()
        self.matcher_type = KDTreeMatcherType()
        self.match_filter = IdentityMatchFilter()
        self.minimizer = PointToPlaneMinimizer()
        self.transformation_checker = MaxIterationTransformationChecker(50)

    def with_reference_preprocessing(self, reference_preprocessing: Filter):
        self.reference_preprocessing = reference_preprocessing
        return self

    def with_reading_preprocessing(self, reading_preprocessing: Filter):
        self.reading_preprocessing = reading_preprocessing
        return self

    def with_matcher_type(self, matcher_type: MatcherType):
        self.matcher_type = matcher_type
        return self

    def with_match_filter(self, match_filter: MatchFilter):
        self.match_filter = match_filter
        return self

    def with_minimizer(self, minimizer: Minimizer):
        self.minimizer = minimizer
        return self

    def with_transformation_checker(self, transformation_checker: TransformationChecker):
        self.transformation_checker = transformation_checker
        return self

    def build(self):
        return ICP(self.descriptors,
                   self.reference_preprocessing,
                   self.reading_preprocessing,
                   self.matcher_type,
                   self.match_filter,
                   self.minimizer,
                   self.transformation_checker)


class ICP:
    def __init__(self,
                 descriptors: Dict[str, Descriptor],
                 reference_preprocessing: Filter,
                 reading_preprocessing: Filter,
                 matcher_type: MatcherType,
                 match_filter: MatchFilter,
                 minimizer: Minimizer,
                 transformation_checker: TransformationChecker):
        """
        ICP algorithm. The basic flow is that we apply corresponding preprocessing pipelines to both the reference and
        reading `PointCloud`. Then we start the iterative loop: Compute matches, filter out outliers, find the
        minimizing transformation, then continue depending on the output of `transformation_checker`
        """
        self.descriptors = descriptors
        self.reference_preprocessing = reference_preprocessing
        self.reading_preprocessing = reading_preprocessing
        self.matcher_type = matcher_type
        self.match_filter = match_filter
        self.minimizer = minimizer
        self.transformation_checker = transformation_checker

    def find_transformation(self,
                            reference: PointCloud,
                            reading: PointCloud,
                            init_pose: Optional[np.ndarray] = None) -> np.ndarray:
        dim = reference.homogeneous_dim
        if init_pose is not None:
            T = np.copy(init_pose)
        else:
            T = np.identity(dim)

        self.reference_preprocessing.filter(reference, self.descriptors)
        self.reading_preprocessing.filter(reading, self.descriptors)

        reading_prime = copy(reading)
        matcher = self.matcher_type.make_matcher(reference)

        self.transformation_checker.begin()
        while True:
            reading_prime.features = T @ reading.features

            matches = matcher.match(reading_prime)
            self.match_filter.filter_matches(reading_prime, matches)
            T_iter = self.minimizer.find_transformation(reference, reading_prime, matches, self.descriptors)

            T = T_iter @ T

            if self.transformation_checker.is_finished(reference, reading, matches):
                break

        return T


if __name__ == '__main__':
    import pickle
    from matplotlib import pyplot as plt
    from r2_lidar_icp.match_filters.outlier_match_filter import OutlierMatchFilter
    from r2_lidar_icp.utils.draw_utils import draw_point_clouds

    reading = PointCloud.from_scan(pickle.load(open('data/pi/test1/00000.pkl', 'rb')))
    reference = PointCloud.from_scan(pickle.load(open('data/pi/test1/00050.pkl', 'rb')))

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Pre-ICP
    ax = axs[0]
    ax.set_title("Before registration")
    draw_point_clouds(ax, P=reading.features, Q=reference.features)

    # ICP
    icp_builder = ICPBuilder().with_minimizer(PointToPlaneMinimizer()).with_match_filter(OutlierMatchFilter(100))
    icp = icp_builder.build()
    T = icp.find_transformation(reference, reading)
    print(T)

    # Post-ICP
    ax = axs[1]
    ax.set_title("After registration")
    draw_point_clouds(ax,
                      P=T @ reading.features,
                      Q=reference.features,
                      normals_Q=reference.get_descriptor(NormalDescriptor.name, icp_builder.descriptors),
                      T=T)

    fig.tight_layout()
    fig.show()
