from typing import Optional

import numpy as np


class TransformationMatrix2D:
    def __init__(self):
        self.transform = np.eye(3)

    def translate(self, x: float, y: float, z: float = 0) -> 'TransformationMatrix2D':
        self.transform = self.transform @ self.make_translation_xy(x, y)
        return self

    def rotate(self, theta: float) -> 'TransformationMatrix2D':
        self.transform = self.transform @ self.make_rotation(theta)
        return self

    def scale(self, sx: float, sy: Optional[float] = None) -> 'TransformationMatrix2D':
        if sy is None:
            sy = sx
        self.transform = self.transform @ self.make_scaling(sx, sy)
        return self

    def build(self) -> np.ndarray:
        return self.transform

    @staticmethod
    def make_translation(point):
        return np.array([[1, 0, point[0]],
                         [0, 1, point[1]],
                         [0, 0, 1]])

    @staticmethod
    def make_translation_xy(x, y):
        return TransformationMatrix2D.make_translation((x, y))

    @staticmethod
    def make_rotation(angle):
        c = np.math.cos(angle)
        s = np.math.sin(angle)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])

    @staticmethod
    def make_scaling(sx, sy):
        return np.array([[sx, 0, 0],
                         [0, sy, 0],
                         [0, 0, 1]])


class TransformationMatrix3D:
    def __init__(self):
        self.transform = np.eye(4)

    def translate(self, x: float, y: float, z: float = 0) -> 'TransformationMatrix3D':
        self.transform = self.transform @ self.make_translation_xyz(x, y, z)
        return self

    def rotate_x(self, theta: float) -> 'TransformationMatrix3D':
        self.transform = self.transform @ self.make_rotation_x(theta)
        return self

    def rotate_y(self, theta: float) -> 'TransformationMatrix3D':
        self.transform = self.transform @ self.make_rotation_y(theta)
        return self

    def rotate_z(self, theta: float) -> 'TransformationMatrix3D':
        self.transform = self.transform @ self.make_rotation_z(theta)
        return self

    def scale(self, sx: float, sy: Optional[float] = None, sz: Optional[float] = None) -> 'TransformationMatrix3D':
        if sy is None and sz is None:
            sy = sx
            sz = sx
        elif sy is None or sz is None:
            raise ValueError('Either give only sx, or all sx, sy and sz')
        self.transform = self.transform @ self.make_scaling(sx, sy, sz)
        return self

    def build(self) -> np.ndarray:
        return self.transform

    @staticmethod
    def make_translation(point):
        return np.array([[1, 0, 0, point[0]],
                         [0, 1, 0, point[1]],
                         [0, 0, 1, point[2]],
                         [0, 0, 0, 1]])

    @staticmethod
    def make_translation_xyz(x, y, z):
        return TransformationMatrix3D.make_translation((x, y, z))

    @staticmethod
    def make_rotation_x(radian_angle_around_x):
        c = np.math.cos(radian_angle_around_x)
        s = np.math.sin(radian_angle_around_x)
        return np.array([[1, 0, 0, 0],
                         [0, c, -s, 0],
                         [0, s, c, 0],
                         [0, 0, 0, 1]])

    @staticmethod
    def make_rotation_y(radian_angle_around_y):
        c = np.math.cos(radian_angle_around_y)
        s = np.math.sin(radian_angle_around_y)
        return np.array(
            [[c, 0, s, 0],
             [0, 1, 0, 0],
             [-s, 0, c, 0],
             [0, 0, 0, 1]])

    @staticmethod
    def make_rotation_z(radian_angle_around_z):
        c = np.math.cos(radian_angle_around_z)
        s = np.math.sin(radian_angle_around_z)
        return np.array(
            [[-c, -s, 0, 0],
             [s, c, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]])

    @staticmethod
    def make_rotation(theta_x, theta_y, theta_z):
        rot_x = TransformationMatrix3D.make_rotation_x(theta_x)
        rot_y = TransformationMatrix3D.make_rotation_y(theta_y)
        rot_z = TransformationMatrix3D.make_rotation_z(theta_z)
        return rot_z @ rot_y @ rot_x

    @staticmethod
    def make_scaling(sx, sy, sz):
        return np.array([[sx, 0, 0, 0],
                         [0, sy, 0, 0],
                         [0, 0, sz, 0],
                         [0, 0, 0, 1]])
