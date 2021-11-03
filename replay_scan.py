import argparse
import pathlib
import pickle

import cv2.cv2 as cv2
import numpy as np

from visualize_point_cloud import draw_point_cloud

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay a lidar scan sequence')
    parser.add_argument('scans_path', help='Scan folder')
    args = parser.parse_args()

    scans_path = pathlib.Path(args.scans_path)
    scans_paths = scans_path.iterdir()

    window = "lidar"
    window_size = 500

    cv2.namedWindow(window)
    for i, scan in enumerate(sorted(list(scans_paths))):
        img = np.zeros((window_size, window_size, 3), dtype=np.uint8)
        scan = pickle.load(open(scan, 'rb'))

        qualities, angles, distances = scan[:, 0], scan[:, 1], scan[:, 2]
        angles = np.deg2rad(angles)

        xs = np.cos(angles) * distances
        ys = np.sin(angles) * distances

        pc = np.stack((xs, ys))
        draw_point_cloud(pc, img, window_size, (255, 255, 0))

        cv2.imshow(window, img)
        key = cv2.waitKey(100)
        if key == ord(' '):
            pickle.dump(scan, open(f'scans/{i}.pkl', 'wb'))
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
