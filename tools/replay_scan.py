import argparse
import pathlib
import pickle

import cv2.cv2 as cv2
import numpy as np

from r2_lidar_icp.draw_utils import draw_point_cloud_cv2
from r2_lidar_icp.point_cloud import PointCloud

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay a lidar scan sequence')
    parser.add_argument('scans_path', help='Scan folder')
    args = parser.parse_args()

    scans_path = pathlib.Path(args.scans_path)
    scans_paths = scans_path.iterdir()

    window = "lidar"
    window_size = 750

    cv2.namedWindow(window)
    for i, scan in enumerate(sorted(list(scans_paths))):
        img = np.zeros((window_size, window_size, 3), dtype=np.uint8)
        scan = pickle.load(open(scan, 'rb'))
        pc = PointCloud.from_scan(scan)

        draw_point_cloud_cv2(pc, img, window_size, (255, 255, 0))

        cv2.imshow(window, img)
        key = cv2.waitKey(100)
        if key == ord(' '):
            pickle.dump(scan, open(f'scans/{i}.pkl', 'wb'))
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
