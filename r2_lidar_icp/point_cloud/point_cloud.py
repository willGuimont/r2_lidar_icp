import numpy as np


class PointCloud:
    def __init__(self, features: np.ndarray):
        self.features = features
        self.descriptors = dict()

    @classmethod
    def from_scan(cls, scan) -> 'PointCloud':
        scan = np.array(scan)

        qualities, angles, distances = scan[:, 0], scan[:, 1], scan[:, 2]
        angles = np.deg2rad(angles)

        xs = np.cos(angles) * distances
        ys = np.sin(angles) * distances

        features = np.stack([xs, ys])
        features = cls._point_to_homogeneous(features)

        pc = PointCloud(features)

        return pc

    @staticmethod
    def _point_to_homogeneous(pc: np.ndarray) -> np.ndarray:
        if pc.shape[0] == 3:
            return np.copy(pc.T)
        elif pc.shape[0] == 2:
            return np.concatenate((pc, np.ones((1, pc.shape[1]))), axis=0)
        else:
            raise ValueError(f'{pc.shape} is an invalide shape, expected Nx3 or Nx4')
