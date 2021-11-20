from typing import Optional, Dict

import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.filters.filter import Filter
from r2_lidar_icp.icp import ICP
from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class Mapping:
    def __init__(self,
                 icp: ICP,
                 reference: PointCloud,
                 reference_maintaining: Filter,
                 reference_descriptors: Dict[str, Descriptor],
                 last_position: Optional[np.ndarray] = None):
        self.icp = icp
        self.reference = reference
        self.reference_maintaining = reference_maintaining
        self.reference_descriptors = reference_descriptors
        if last_position is not None:
            self.last_position = last_position
        else:
            self.last_position = np.identity(reference.homogeneous_dim)

    def map(self, reading: PointCloud):
        self.last_position = self.localize(reading)
        reading_in_reference = PointCloud(self.last_position @ reading.features)
        self.reference.add(reading_in_reference, self.icp.descriptors)

    def localize(self, reading) -> np.ndarray:
        return self.icp.find_transformation(
            self.reference,
            reading,
            self.last_position)

    def maintain_reference(self):
        self.reference_maintaining.filter(self.reference, self.icp.descriptors | self.reference_descriptors)


if __name__ == '__main__':
    import argparse
    import pathlib
    import pickle
    import time

    from matplotlib import pyplot as plt

    from r2_lidar_icp.icp import ICPBuilder
    from r2_lidar_icp.match_filters.outlier_match_filter import OutlierMatchFilter
    from r2_lidar_icp.filters.furthest_point_sampling_filter import FurthestPointSamplingFilter
    from r2_lidar_icp.transformation_checkers.composed_transformation_checker import ComposedTransformationChecker
    from r2_lidar_icp.transformation_checkers.max_iteration_transformation_checker import \
        MaxIterationTransformationChecker
    from r2_lidar_icp.transformation_checkers.error_delta_transformation_checker import ErrorDeltaTransformationChecker
    from r2_lidar_icp.utils.draw_utils import draw_point_clouds

    parser = argparse.ArgumentParser(description='Replay a lidar scan sequence')
    parser.add_argument('scans_path', help='Scan folder')
    args = parser.parse_args()

    scans_path = pathlib.Path(args.scans_path)
    scans_paths = sorted(list(scans_path.iterdir()))
    scans = [pickle.load(open(scan_path, 'rb')) for scan_path in scans_paths]

    first_scan = PointCloud.from_scan(scans[0])

    icp_builder = ICPBuilder(). \
        with_match_filter(OutlierMatchFilter(100)). \
        with_transformation_checker(ComposedTransformationChecker([
        MaxIterationTransformationChecker(50),
        ErrorDeltaTransformationChecker(0.001)
    ]))
    icp = icp_builder.build()
    reference_consolidation = FurthestPointSamplingFilter(300, skip_initial=True)
    reference_descriptors = dict()
    mapping = Mapping(icp, first_scan, reference_consolidation, reference_descriptors, last_position=None)

    start_time = time.perf_counter()
    for i, scan in enumerate(scans[1:]):
        reading = PointCloud.from_scan(scan)
        mapping.map(reading)
        if i % 5 == 0:
            mapping.maintain_reference()

    print(f'Duration: {time.perf_counter() - start_time}')

    fig, ax = plt.subplots()
    draw_point_clouds(ax, P=mapping.reference.features)
    fig.show()
