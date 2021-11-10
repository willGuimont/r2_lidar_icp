import argparse
import pathlib
import pickle
from copy import copy

import cv2.cv2 as cv2
import numpy as np

from lidar_icp.point_cloud import PointCloud

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay a lidar scan sequence')
    parser.add_argument('scans_path', help='Scan folder')
    args = parser.parse_args()

    scans_path = pathlib.Path(args.scans_path)
    scans_paths = list(scans_path.iterdir())

    window_map = "map"
    window_icp = "icp"
    window_size = 750

    cv2.namedWindow(window_map)
    last_scan = pickle.load(open(scans_paths[0], 'rb'))
    last_pc = PointCloud.from_scan(last_scan)
    pc_map = copy(last_pc)
    last_T = np.eye(4)
    for i, scan in enumerate(sorted(list(scans_paths))):
        img_map = np.zeros((window_size, window_size, 3), dtype=np.uint8)
        img_icp = np.zeros((window_size, window_size, 3), dtype=np.uint8)
        scan = pickle.load(open(scan, 'rb'))

        pc = PointCloud.from_scan(scan)
        T, _ = icp(pc_map, pc, last_T)  # TODO
        T_inv = np.linalg.inv(T)

        pc_in_last = homogeneous_to_points(T_inv @ point_to_homogeneous(pc))
        pc_map = np.concatenate([pc_map, pc_in_last], axis=0)
        np.random.shuffle(pc_map)
        indices, _ = furthest_point_sampling(pc_map, 400, skip_initial=True)
        pc_map = pc_map[indices]

        draw_point_cloud(pc.T, img_icp, window_size, (255, 255, 0))
        draw_point_cloud(pc_map.T, img_map, window_size, (0, 255, 255), 20_000)

        cv2.imshow(window_map, img_map)
        cv2.imshow(window_icp, img_icp)
        key = cv2.waitKey(100)
        if key == ord(' '):
            pickle.dump(scan, open(f'scans/{i}.pkl', 'wb'))
        if key == ord('q') or key == 27:
            break

        last_scan = scan
        last_pc = pc
        last_T = T

    cv2.waitKey(0)
    cv2.destroyAllWindows()
