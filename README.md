# r2_lidar_icp

Iterative Closest Point (ICP) algorithm for lidar odometry.
r2_lidar_icp is a simple Python implementation of the ICP algorithm aimed to be easy to understand and modify.

[r2_lidar_icp in action](https://youtu.be/9I7yZk28Vi0?si=otcAcv2YrVtqMob7)

## Pipeline for ICP
- Preprocess (downsample, filter, compute descriptors, etc.)
- While not converged
  - Find matches
  - Reject outlier matches
  - Compute transformation
  - Transform point cloud
- Return transformation

## TODO
- [ ] Add tests
- [ ] Add 3D support for PointToPlaneMinimizer
- [ ] Profile and optimize (numba, cython, etc.)
- [ ] Refactor PointCloud to contain its descriptors
- [ ] Robust kernel
- [ ] De-skewing using IMU
- [ ] Dynamic object detection
- [ ] Diffentiable ICP
- [ ] Differentiable LiDAR simulator
- [ ] More filters
  - [libpointmatcher filters](https://libpointmatcher.readthedocs.io/en/latest/DataFilters/#filter-index)
  - [libpointmatcher config](https://libpointmatcher.readthedocs.io/en/latest/Configuration/)
