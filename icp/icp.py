import numpy as np


def scan_to_point_cloud(scan):
    scan = np.array(scan)

    qualities, angles, distances = scan[:, 0], scan[:, 1], scan[:, 2]
    angles = np.deg2rad(angles)

    xs = np.cos(angles) * distances
    ys = np.sin(angles) * distances
    zs = np.zeros_like(xs)

    stack = np.stack([xs, ys, zs]).T
    return stack


def point_to_homogeneous(pc):
    """
    :param pc: Nx2, Nx3 or Nx4 numpy array
    :return: deep copy of pc in homogeneous coordinates.
    """
    if pc.shape[1] == 4:
        return np.copy(pc.T)
    elif pc.shape[1] == 3:
        cp = np.copy(pc)
        return np.concatenate((cp.T, np.ones((1, cp.shape[0]))), axis=0)
    elif pc.shape[1] == 2:
        cp = np.copy(pc)
        return np.concatenate((cp.T, np.zeros((1, cp.shape[0])), np.ones((1, cp.shape[0]))), axis=0)
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
    indices = np.zeros(src.shape[0], dtype=np.int)
    distances = np.zeros(src.shape[0])
    for i, s in enumerate(src):
        distance_mat = np.linalg.norm(dst[:] - s, axis=1)
        indices[i] = np.argmin(distance_mat)
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
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T, R, t


def icp(A, B, init_pose=None, max_iter=50, tolerance=0.001):
    """
    The Iterative Closest Point method
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
        src = np.dot(init_pose, src)

    prev_error = 0
    distances = None
    for i in range(max_iter):
        distances, indices = nearest_neighbor(src[:3, :].T, dst[:3, :].T)
        T, _, _ = best_fit_transform(src[:3, :].T, dst[:3, indices].T)
        src = np.dot(T, src)
        mean_error = np.average(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    T, _, _ = best_fit_transform(A[:, :3], src[:3, :].T)
    return T, distances


if __name__ == '__main__':
    import pickle
    from cv2 import cv2
    from tools.visualize_point_cloud import draw_point_cloud

    scan_0 = pickle.load(open('data/live/8.pkl', 'rb'))
    scan_1 = pickle.load(open('data/live/85.pkl', 'rb'))

    pc0 = scan_to_point_cloud(scan_0)
    pc1 = scan_to_point_cloud(scan_1)

    T, distances = icp(pc0, pc1)

    window = "icp"
    window_size = 500
    cv2.namedWindow(window)

    src = point_to_homogeneous(pc0)
    dst = point_to_homogeneous(pc1)

    img = np.zeros((window_size, window_size, 3), dtype=np.uint8)
    draw_point_cloud(src, img, window_size, (255, 255, 0))
    draw_point_cloud(dst, img, window_size, (255, 0, 255))
    draw_point_cloud(T @ point_to_homogeneous(pc0), img, window_size, (255, 255, 255))
    cv2.imshow(window, img)
    cv2.waitKey(0)
