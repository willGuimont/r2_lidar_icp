from copy import copy
from typing import Dict, List, Union

import numpy as np

from r2_lidar_icp.utils.utils import point_to_homogeneous


# TODO simplify descriptors handling
class PointCloud:
    def __init__(self, features: np.ndarray):
        """
        Point cloud with features and descriptors
        :param features: Features of the point cloud
        """
        self.features = features
        self.descriptors = dict()

    @classmethod
    def from_cartesian_scan(cls, scan: Union[np.ndarray, List]) -> 'PointCloud':
        """
        Create a point cloud from a scan and convert it to homogeneous coordinates
        :param scan: a numpy array of shape (2, n) or (3, n) where n is the number of points, the first column is
            the x coordinate and the second column is the y coordinate, the third column is the z coordinate
        :return: Point cloud
        """
        scan = np.array(scan)
        features = point_to_homogeneous(scan)
        return PointCloud(features)

    @classmethod
    def from_polar_scan(cls, scan: Union[np.ndarray, List]) -> 'PointCloud':
        """
        Create a point cloud from a scan and convert it to homogeneous coordinates
        :param scan: a numpy array of shape (2, n) where n is the number of points,
                     the first row is the angle and the second row is the distance
        :return: Point cloud
        """
        scan = np.array(scan)

        angles, distances = scan[0, :], scan[1, :]
        angles = np.deg2rad(angles)

        xs = np.cos(angles) * distances
        ys = np.sin(angles) * distances

        features = np.stack([xs, ys])
        features = point_to_homogeneous(features)

        pc = PointCloud(features)

        return pc

    @classmethod
    def from_rplidar_scan(cls, scan: Union[np.ndarray, List]) -> 'PointCloud':
        """
        Same as from_polar_scan but with a different order of the rows.
        Automatically add the intensity (signal return strength) as a descriptor.
        :param scan: a numpy array of shape (3, n) where n is the number of points, the first row is the intensity,
                     the second row is the angle and the third row is the distance
        :return: Point cloud
        """
        intensity = np.array(scan).T[0][None, :]
        pc = cls.from_polar_scan(np.array(scan).T[1:, :])
        pc.add_descriptor_by_name('IntensityDescriptor', intensity)
        return pc

    def get_descriptor(self, descriptor_name: str, descriptors: Dict[str, 'Descriptor']):
        """
        Get a descriptor from the point cloud
        :param descriptor_name: Name of the descriptor
        :param descriptors: Descriptors of the point cloud
        :return: descriptor
        """
        self.compute_descriptor(descriptor_name, descriptors)
        return self.descriptors[descriptor_name]

    def compute_descriptor(self, descriptor_name: str, descriptors: Dict[str, 'Descriptor']):
        """
        Compute a descriptor from the point cloud if it is not already computed
        :param descriptor_name: Name of the descriptor
        :param descriptors: Descriptors of the point cloud
        :return: None
        """
        if descriptor_name not in self.descriptors:
            if descriptor_name not in descriptors:
                raise RuntimeError(f'Descriptor {descriptor_name} was not computed and is not in descriptors')
            descriptor = descriptors[descriptor_name]
            descriptor.compute_descriptor(self)

    def add_descriptor(self, descriptor: 'Descriptor', value):
        """
        Add a descriptor to the point cloud
        :param descriptor: Descriptor to add
        :param value: Value of the descriptor
        :return: None
        """
        self.descriptors[descriptor.name] = value

    def add_descriptor_by_name(self, name: str, value):
        """
        Add a descriptor to the point cloud
        :param name: Name of the descriptor
        :param value: Value of the descriptor
        :return: None
        """
        self.descriptors[name] = value

    def apply_mask(self, mask: np.ndarray):
        """
        Apply a mask to the point cloud features and descriptors
        :param mask: Mask to apply
        :return: None
        """
        self.features = self.features[:, mask]
        for k, v in self.descriptors.items():
            self.descriptors[k] = v[:, mask]

    def union(self, other: 'PointCloud', descriptors: Dict[str, 'Descriptor']):
        """
        Union of two point clouds
        :param other: Other point cloud
        :param descriptors: Descriptors of the point cloud
        :return:
        """
        self.features = np.concatenate((self.features, other.features), axis=1)
        for desc in self.descriptors.keys():
            other.compute_descriptor(desc, descriptors)
            self.descriptors[desc] = np.concatenate((self.descriptors[desc], other.descriptors[desc]), axis=1)

    @property
    def dim(self):
        """
        Dimension of the point cloud
        :return: Dimension
        """
        return self.features.shape[0] - 1

    @property
    def homogeneous_dim(self):
        """
        Dimension of the point cloud with homogeneous coordinates
        :return: Dimension
        """
        return self.features.shape[0]

    @property
    def num_points(self):
        """
        Number of points in the point cloud
        :return: Number of points
        """
        return self.features.shape[1]

    def __copy__(self):
        """
        Copy the point cloud
        :return: Copy of the point cloud
        """
        pc = PointCloud(copy(self.features))

        for k, v in self.descriptors.items():
            pc.descriptors[k] = copy(v)

        return pc
