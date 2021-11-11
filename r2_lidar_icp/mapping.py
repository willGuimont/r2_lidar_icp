import argparse
import pathlib
import pickle

import cv2.cv2 as cv2
import numpy as np

from r2_lidar_icp.draw_utils import draw_point_cloud_cv2
from r2_lidar_icp.icp import icp
from r2_lidar_icp.point_cloud import PointCloud


def mapping(pc_map: PointCloud, pc: PointCloud, nb_iter: int, tau_filter: float, init_pos: np.ndarray = None) \
        -> (PointCloud, np.ndarray):
    pc_to_map = icp(pc, pc_map, nb_iter, tau_filter=tau_filter, init_pose=init_pos)
    pc_features_in_map = pc_to_map @ pc.features
    return pc_map + pc_features_in_map, pc_to_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay a lidar scan sequence')
    parser.add_argument('scans_path', help='Scan folder')
    args = parser.parse_args()

    scans_path = pathlib.Path(args.scans_path)
    scans_paths = sorted(list(scans_path.iterdir()))

    window_map = "map"
    window_icp = "point cloud"
    window_size = 750
    cv2.namedWindow(window_map)

    pc_map = PointCloud.from_scan(pickle.load(open(scans_paths[0], 'rb')))
    init_pos = np.eye(3)
    for i, scan in enumerate(sorted(scans_paths)):
        scan = pickle.load(open(scan, 'rb'))
        pc = PointCloud.from_scan(scan)

        last_map = pc_map
        pc_map, pc_to_map = mapping(pc_map, pc, nb_iter=100, tau_filter=100, init_pos=init_pos)
        pc_map = pc_map.subsample(250)
        init_pos = pc_to_map

        # draw
        img_map = np.zeros((window_size, window_size, 3), dtype=np.uint8)
        img_pc = np.zeros((window_size, window_size, 3), dtype=np.uint8)

        draw_point_cloud_cv2(pc, img_pc, window_size, (255, 0, 255))
        draw_point_cloud_cv2(pc_map, img_map, window_size, (0, 255, 255), 20_000)

        cv2.imshow(window_icp, img_pc)
        cv2.imshow(window_map, img_map)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()
