import argparse
import pathlib
import pickle
import time

import numpy as np
from cv2 import cv2

from backup.draw_utils import draw_point_cloud_cv2
from backup.icp import icp
from backup.point_cloud import PointCloud


def mapping(pc_map: PointCloud,
            pc: PointCloud,
            nb_iter: int,
            init_pose: np.ndarray,
            tau_filter: float,
            min_error_delta: float) -> (PointCloud, np.ndarray):
    pc_to_map = icp(pc, pc_map, nb_iter, init_pose, tau_filter, min_error_delta)
    pc_features_in_map = pc_to_map @ pc.features
    return pc_map + pc_features_in_map, pc_to_map


def main(scans, show):
    if show:
        window_map = "map"
        window_icp = "point cloud"
        window_size = 750
        cv2.namedWindow(window_map)
        cv2.namedWindow(window_icp)

    nb_iter = 50
    tau_filter = 100
    min_error_delta = 0.001
    points_to_keep = 300
    normal_knn = 20

    start_time = time.time()
    pc_map = PointCloud.from_scan(scans[0])
    init_pose = np.eye(3)
    for i, scan in enumerate(scans):
        pc = PointCloud.from_scan(scan)

        # preprocessing
        pc.compute_normal_descriptor(k_nn=normal_knn)
        pc_map.compute_normal_descriptor(k_nn=normal_knn)

        pc_map, pc_to_map = mapping(pc_map, pc, nb_iter, init_pose, tau_filter, min_error_delta)
        pc_map = pc_map.subsample(points_to_keep)
        init_pose = pc_to_map

        if show:
            img_map = np.zeros((window_size, window_size, 3), dtype=np.uint8)
            img_pc = np.zeros((window_size, window_size, 3), dtype=np.uint8)

            draw_point_cloud_cv2(pc, img_pc, window_size, (255, 0, 255))
            draw_point_cloud_cv2(pc_map, img_map, window_size, (0, 255, 255), 20_000, (-50, 100))

            cv2.imshow(window_icp, img_pc)
            cv2.imshow(window_map, img_map)
            key = cv2.waitKey(1)

            if key == ord('q') or key == 27:
                break

    print(f'Duration: {time.time() - start_time}')
    if show:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay a lidar scan sequence')
    parser.add_argument('scans_path', help='Scan folder')
    parser.add_argument('--no_show', action='store_true', help='Disable display')
    args = parser.parse_args()

    scans_path = pathlib.Path(args.scans_path)
    scans_paths = sorted(list(scans_path.iterdir()))
    scans = [pickle.load(open(scan_path, 'rb')) for scan_path in scans_paths]
    show = not args.no_show

    main(scans, show)