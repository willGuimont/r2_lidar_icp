import argparse
import pathlib
import pickle

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert pkl to npy')
    parser.add_argument('scans_path', help='Scan folder')
    parser.add_argument('out_path', help='Output folder')
    args = parser.parse_args()

    scans_path = pathlib.Path(args.scans_path)
    scans_paths = scans_path.iterdir()
    out_path = pathlib.Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    for i, scan_path in enumerate(sorted(list(scans_paths))):
        scan = pickle.load(open(scan_path, 'rb'))
        file_path = out_path.joinpath(f'{i:05d}.npy')
        np.save(file_path, scan)
