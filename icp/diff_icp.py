import pickle

import torch


def scan_to_point_cloud(scan):
    scan = torch.array(scan)

    qualities, angles, distances = scan[:, 0], scan[:, 1], scan[:, 2]
    angles = torch.deg2rad(angles)

    xs = torch.cos(angles) * distances
    ys = torch.sin(angles) * distances
    zs = torch.zeros_like(xs)

    stack = torch.stack([xs, ys, zs]).T
    return stack


def point_to_homogeneous(pc):
    """
    :param pc: Nx2, Nx3 or Nx4 numpy array
    :return: deep copy of pc in homogeneous coordinates.
    """
    if pc.shape[1] == 4:
        return torch.copy(pc.T)
    elif pc.shape[1] == 3:
        cp = torch.copy(pc)
        return torch.concatenate((cp.T, torch.ones((1, cp.shape[0]))), axis=0)
    elif pc.shape[1] == 2:
        cp = torch.copy(pc)
        return torch.concatenate((cp.T, torch.zeros((1, cp.shape[0])), torch.ones((1, cp.shape[0]))), axis=0)
    else:
        raise ValueError(f'{pc.shape} is an invalide shape, expected Nx3 or Nx4')


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    :param src: Nx3 array of points
    :param dst: Nx3 array of points
    :return: (distance, indices):
            distance: Euclidean distances (errors) of the nearest neighbor
            indices: dst indices of the nearest neighbor
    """
    indices = torch.zeros(src.shape[0], dtype=torch.int)
    distances = torch.zeros(src.shape[0])
    for i, s in enumerate(src):
        distance_mat = torch.linalg.norm(dst[:] - s, axis=1)
        indices[i] = torch.argmin(distance_mat)
        distances[i] = distance_mat[indices[i]]
    return distances, indices


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform between corresponding 3D points A->B
    :param A: Nx3 numpy array of corresponding 3D points
    :param B: Nx3 numpy array of corresponding 3D points
    :return: (T, R, t)
             T: 4x4 homogeneous transformation matrix
             R: 3x3 rotation matrix
             t: 3x1 column vector
    """
    assert len(A) == len(B)

    # translate points to their centroids
    centroid_A = torch.mean(A, axis=0)
    centroid_B = torch.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = torch.dot(AA.T, BB)
    U, S, Vt = torch.linalg.svd(H)
    R = torch.dot(Vt.T, U.T)

    # special reflection case
    if torch.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = torch.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - torch.dot(R, centroid_A.T)

    # homogeneous transformation
    T = torch.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T, R, t


def icp(A, B, init_pose=None, max_iter=50, tolerance=0.001):
    """
    Fully differentiable Iterative Closest Point method
    :param A: Nx3 or Nx4 numpy array of source 3D points
    :param B: Nx3 or Nx4 numpy array of destination 3D point
    :param init_pose: 4x4 homogeneous transformation
    :param max_iter: exit algorithm after max_iterations
    :param tolerance: convergence criteria
    :return: (T, distances)
            T: final homoheneous transformation
            distances: euclidean distances (errors) of the nearest neighbor
    """
    src = point_to_homogeneous(A)
    dst = point_to_homogeneous(B)

    if init_pose is not None:
        src = torch.dot(init_pose, src)

    raise NotImplementedError()

if __name__ == '__main__':
    scan_0 = pickle.load(open('scans/scan_0.pkl', 'rb'))
    scan_1 = pickle.load(open('scans/scan_1.pkl', 'rb'))

    pc0 = scan_to_point_cloud(scan_0)
    pc1 = scan_to_point_cloud(scan_1)

    T, distances = icp(pc0, pc1)

    print(T)
