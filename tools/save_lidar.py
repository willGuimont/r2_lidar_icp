import argparse
import pathlib
import pickle
import time

import numpy as np
from rplidar import RPLidar

# TODO refactor with PointCloud
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Path to save to')
    parser.add_argument('scans_path', nargs='?', help='Scan folder')
    args = parser.parse_args()

    save_folder = args.scans_path
    if save_folder is None:
        save_folder = pathlib.Path('../data').joinpath(str(time.time()))
    else:
        save_folder = pathlib.Path(save_folder)

    lidar = RPLidar('/dev/ttyUSB0')
    print('Recording...')

    save_folder.mkdir(parents=True, exist_ok=True)

    for i, scan in enumerate(lidar.iter_scans()):
        scan = np.array(scan)
        pickle.dump(scan, open(f'{str(save_folder)}/{i:05d}.pkl', 'wb'))

    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
