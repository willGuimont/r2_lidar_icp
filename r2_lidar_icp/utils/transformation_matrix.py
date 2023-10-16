from typing import Optional

import numpy as np


# TODO test order of transformations
class TransformationMatrix2D:
    def __init__(self):
        """
        Transformation matrix for 2D transformations.
        Initialize with identity transformation.
        """
        self.transform = np.eye(3)

    def translate(self, x: float, y: float) -> 'TransformationMatrix2D':
        """
        Translate the transformation matrix by x, y.
        :param x: x
        :param y: y
        :return: self
        """
        self.transform = self.transform @ self.make_translation_xy(x, y)
        return self

    def rotate(self, theta: float) -> 'TransformationMatrix2D':
        """
        Rotate the transformation matrix by theta.
        :param theta: theta in radians
        :return: self
        """
        self.transform = self.transform @ self.make_rotation(theta)
        return self

    def scale(self, sx: float, sy: Optional[float] = None) -> 'TransformationMatrix2D':
        """
        Scale the transformation matrix by sx, sy.
        :param sx: Scaling factor in x direction
        :param sy: Scaling factor in y direction. If None, sy = sx
        :return: self
        """
        if sy is None:
            sy = sx
        self.transform = self.transform @ self.make_scaling(sx, sy)
        return self

    def build(self) -> np.ndarray:
        """
        Return the transformation matrix.
        :return: transformation matrix
        """
        return self.transform

    @staticmethod
    def make_translation(point):
        """
        Make a translation matrix for a given point.
        :param point: point to translate to
        :return: translation matrix
        """
        return np.array([[1, 0, point[0]],
                         [0, 1, point[1]],
                         [0, 0, 1]])

    @staticmethod
    def make_translation_xy(x, y):
        """
        Make a translation matrix for a given point.
        :param x: x
        :param y: y
        :return: translation matrix
        """
        return TransformationMatrix2D.make_translation((x, y))

    @staticmethod
    def make_rotation(angle):
        """
        Make a rotation matrix for a given angle.
        :param angle: angle in radians
        :return: rotation matrix
        """
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])

    @staticmethod
    def make_scaling(sx, sy):
        """
        Make a scaling matrix for a given scaling factor.
        :param sx: Scaling factor in x direction
        :param sy: Scaling factor in y direction
        :return: scaling matrix
        """
        return np.array([[sx, 0, 0],
                         [0, sy, 0],
                         [0, 0, 1]])


# TODO test order of transformations
class TransformationMatrix3D:
    def __init__(self):
        """
        Transformation matrix for 3D transformations.
        Initialize with identity transformation.
        """
        self.transform = np.eye(4)

    def translate(self, x: float, y: float, z: float = 0) -> 'TransformationMatrix3D':
        """
        Translate the transformation matrix by x, y, z.
        :param x: x
        :param y: y
        :param z: z
        :return: self
        """
        self.transform = self.transform @ self.make_translation_xyz(x, y, z)
        return self

    def rotate_x(self, theta: float) -> 'TransformationMatrix3D':
        """
        Rotate the transformation matrix by theta around the x-axis.
        :param theta: theta in radians
        :return: self
        """
        self.transform = self.transform @ self.make_rotation_x(theta)
        return self

    def rotate_y(self, theta: float) -> 'TransformationMatrix3D':
        """
        Rotate the transformation matrix by theta around the-y axis.
        :param theta: theta in radians
        :return: self
        """
        self.transform = self.transform @ self.make_rotation_y(theta)
        return self

    def rotate_z(self, theta: float) -> 'TransformationMatrix3D':
        """
        Rotate the transformation matrix by theta around the z axis.
        :param theta: theta in radians
        :return: self
        """
        self.transform = self.transform @ self.make_rotation_z(theta)
        return self

    def scale(self, sx: float, sy: Optional[float] = None, sz: Optional[float] = None) -> 'TransformationMatrix3D':
        """
        Scale the transformation matrix by sx, sy, sz.
        :param sx: Scaling factor in x direction
        :param sy: Scaling factor in y direction. If None, sy = sx
        :param sz: Scaling factor in z direction. If None, sz = sx
        :return: self
        """
        if sy is None and sz is None:
            sy = sx
            sz = sx
        elif sy is None or sz is None:
            raise ValueError('Either give only sx, or all sx, sy and sz')
        self.transform = self.transform @ self.make_scaling(sx, sy, sz)
        return self

    def build(self) -> np.ndarray:
        """
        Return the transformation matrix.
        :return: transformation matrix
        """
        return self.transform

    @staticmethod
    def make_translation(point):
        """
        Make a translation matrix for a given point.
        :param point: point to translate to
        :return: translation matrix
        """
        return np.array([[1, 0, 0, point[0]],
                         [0, 1, 0, point[1]],
                         [0, 0, 1, point[2]],
                         [0, 0, 0, 1]])

    @staticmethod
    def make_translation_xyz(x, y, z):
        """
        Make a translation matrix for a given point.
        :param x: x
        :param y: y
        :param z: z
        :return: translation matrix
        """
        return TransformationMatrix3D.make_translation((x, y, z))

    @staticmethod
    def make_rotation_x(radian_angle_around_x):
        """
        Make a rotation matrix for a given angle.
        :param radian_angle_around_x: angle in radians
        :return: rotation matrix
        """
        c = np.cos(radian_angle_around_x)
        s = np.sin(radian_angle_around_x)
        return np.array([[1, 0, 0, 0],
                         [0, c, -s, 0],
                         [0, s, c, 0],
                         [0, 0, 0, 1]])

    @staticmethod
    def make_rotation_y(radian_angle_around_y):
        """
        Make a rotation matrix for a given angle.
        :param radian_angle_around_y: angle in radians
        :return: rotation matrix
        """
        c = np.cos(radian_angle_around_y)
        s = np.sin(radian_angle_around_y)
        return np.array(
            [[c, 0, s, 0],
             [0, 1, 0, 0],
             [-s, 0, c, 0],
             [0, 0, 0, 1]])

    @staticmethod
    def make_rotation_z(radian_angle_around_z):
        """
        Make a rotation matrix for a given angle.
        :param radian_angle_around_z: angle in radians
        :return: rotation matrix
        """
        c = np.cos(radian_angle_around_z)
        s = np.sin(radian_angle_around_z)
        return np.array(
            [[-c, -s, 0, 0],
             [s, c, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]])

    @staticmethod
    def make_rotation(theta_x, theta_y, theta_z):
        """
        Make a rotation matrix for a given angle.
        TODO validate order of rotations
        Applies rotation around x, then y, then z.
        :param theta_x: Angle in radians around x
        :param theta_y: Angle in radians around y
        :param theta_z: Angle in radians around z
        :return: rotation matrix
        """
        rot_x = TransformationMatrix3D.make_rotation_x(theta_x)
        rot_y = TransformationMatrix3D.make_rotation_y(theta_y)
        rot_z = TransformationMatrix3D.make_rotation_z(theta_z)
        return rot_z @ rot_y @ rot_x

    @staticmethod
    def make_scaling(sx, sy, sz):
        """
        Make a scaling matrix for a given scaling factor.
        :param sx: Scaling factor in x direction
        :param sy: Scaling factor in y direction
        :param sz: Scaling factor in z direction
        :return: scaling matrix
        """
        return np.array([[sx, 0, 0, 0],
                         [0, sy, 0, 0],
                         [0, 0, sz, 0],
                         [0, 0, 0, 1]])
