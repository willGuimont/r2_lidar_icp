from copy import copy
from typing import Optional, Dict, Type

import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.descriptors.normal_descriptor import NormalDescriptor
from r2_lidar_icp.descriptors.polar_descriptor import PolarDescriptor
from r2_lidar_icp.filters.filter import Filter
from r2_lidar_icp.filters.identity_filter import IdentityFilter
from r2_lidar_icp.match_filters.identity_match_filter import IdentityMatchFilter
from r2_lidar_icp.match_filters.match_filter import MatchFilter
from r2_lidar_icp.matchers.kdtree_matcher import KDTreeMatcher
from r2_lidar_icp.matchers.matcher import Matcher
from r2_lidar_icp.minimizer.minimizer import Minimizer
from r2_lidar_icp.minimizer.point_to_plane_minimizer import PointToPlaneMinimizer
from r2_lidar_icp.point_cloud import PointCloud
from r2_lidar_icp.transformation_checkers.max_iteration_transformation_checker import MaxIterationTransformationChecker
from r2_lidar_icp.transformation_checkers.transformation_checker import TransformationChecker


class ICPBuilder:
    def __init__(self):
        """
        Builder for ICP algorithm.
        Allows to configure the different components of the algorithm.
        """
        self.descriptors = dict(PolarDescriptor=PolarDescriptor(),
                                NormalDescriptor=NormalDescriptor(knn=20, matcher_cls=KDTreeMatcher))
        self.reference_preprocessing = IdentityFilter()
        self.reading_preprocessing = IdentityFilter()
        self.matcher_cls = KDTreeMatcher
        self.match_filter = IdentityMatchFilter()
        self.minimizer = PointToPlaneMinimizer()
        self.transformation_checker = MaxIterationTransformationChecker(50)

    def with_reference_preprocessing(self, reference_preprocessing: Filter):
        """
        Set the reference preprocessing pipeline
        :param reference_preprocessing: reference preprocessing pipeline
        :return: self
        """
        self.reference_preprocessing = reference_preprocessing
        return self

    def with_reading_preprocessing(self, reading_preprocessing: Filter):
        """
        Set the reading preprocessing pipeline
        :param reading_preprocessing: reading preprocessing pipeline
        :return: self
        """
        self.reading_preprocessing = reading_preprocessing
        return self

    def with_matcher(self, matcher_cls: Type[Matcher]):
        """
        Set the matcher type. Will be used to build the matcher
        :param matcher_cls: matcher type
        :return: self
        """
        self.matcher_cls = matcher_cls
        return self

    def with_match_filter(self, match_filter: MatchFilter):
        """
        Set the match filter.
        :param match_filter: match filter
        :return: self
        """
        self.match_filter = match_filter
        return self

    def with_minimizer(self, minimizer: Minimizer):
        """
        Set the minimizer.
        :param minimizer: minimizer
        :return: self
        """
        self.minimizer = minimizer
        return self

    def with_transformation_checker(self, transformation_checker: TransformationChecker):
        """
        Set the transformation checker.
        :param transformation_checker: transformation checker
        :return: self
        """
        self.transformation_checker = transformation_checker
        return self

    def build(self):
        """
        Build the ICP algorithm
        :return: ICP algorithm
        """
        return ICP(self.descriptors,
                   self.reference_preprocessing,
                   self.reading_preprocessing,
                   self.matcher_cls,
                   self.match_filter,
                   self.minimizer,
                   self.transformation_checker)


# TODO test this
class ICP:
    def __init__(self,
                 descriptors: Dict[str, Descriptor],
                 reference_preprocessing: Filter,
                 reading_preprocessing: Filter,
                 matcher_cls: Type[Matcher],
                 match_filter: MatchFilter,
                 minimizer: Minimizer,
                 transformation_checker: TransformationChecker):
        """
        ICP algorithm. The basic flow is that we apply corresponding preprocessing pipelines to both the reference and
        reading `PointCloud`. Then we start the iterative loop: Compute matches, filter out outliers, find the
        minimizing transformation, then continue depending on the output of `transformation_checker`
        :param descriptors: descriptors to use
        :param reference_preprocessing: reference preprocessing pipeline
        :param reading_preprocessing: reading preprocessing pipeline
        :param matcher_cls: matcher type
        :param match_filter: match filter
        :param minimizer: minimizer
        :param transformation_checker: transformation checker
        """
        self.descriptors = descriptors
        self.reference_preprocessing = reference_preprocessing
        self.reading_preprocessing = reading_preprocessing
        self.matcher_cls = matcher_cls
        self.match_filter = match_filter
        self.minimizer = minimizer
        self.transformation_checker = transformation_checker

    def find_transformation(self, point_cloud: PointCloud, reference: PointCloud,
                            init_pose: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Find the transformation between the reference and the reading.
        :param point_cloud: PointCloud to register
        :param reference: Reference PointCloud
        :param init_pose: Initial pose
        :return: transformation matrix
        """
        dim = reference.homogeneous_dim
        if init_pose is not None:
            T = np.copy(init_pose)
        else:
            T = np.identity(dim)

        self.reference_preprocessing.filter(reference, self.descriptors)
        self.reading_preprocessing.filter(point_cloud, self.descriptors)

        reading_prime = copy(point_cloud)
        matcher = self.matcher_cls.make_matcher(reference)

        self.transformation_checker.begin()
        while True:
            reading_prime.features = T @ point_cloud.features

            # TODO use knn>1 to allow for more match filters
            matches = matcher.match(reading_prime)
            self.match_filter.filter_matches(reading_prime, matches)
            T_iter = self.minimizer.find_transformation(reading_prime, reference, matches, self.descriptors)

            T = T_iter @ T

            if self.transformation_checker.is_finished(point_cloud, reference, matches):
                break

        return T


if __name__ == '__main__':
    import pickle
    from matplotlib import pyplot as plt
    from r2_lidar_icp.match_filters.max_distance_match_filter import MaxDistanceMatchFilter
    from r2_lidar_icp.utils.draw_utils import draw_point_clouds

    reading = PointCloud.from_rplidar_scan(pickle.load(open('data/pi/test1/00000.pkl', 'rb')))
    reference = PointCloud.from_rplidar_scan(pickle.load(open('data/pi/test1/00050.pkl', 'rb')))

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Pre-ICP
    ax = axs[0]
    ax.set_title("Before registration")
    draw_point_clouds(ax, P=reading.features, Q=reference.features)

    # ICP
    icp_builder = ICPBuilder().with_minimizer(PointToPlaneMinimizer()).with_match_filter(MaxDistanceMatchFilter(100))
    icp = icp_builder.build()
    T = icp.find_transformation(reading, reference)
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
