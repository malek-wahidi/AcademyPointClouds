import open3d as o3d
import numpy as np
import time

"""
def register(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> np.ndarray:

    This is the function that students will implement.
    It should take two open3d.geometry.PointCloud objects as input
    and return a 4x4 numpy array representing the transformation matrix
    that aligns pcd2 to pcd1.

    # Placeholder implementation: returns an identity matrix
   return np.identity(4)
"""


def preprocess_point_cloud(pcd, voxel_size):
    print("==> Downsampling point cloud...")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print("==> Estimating normals...")
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    print("==> Computing FPFH features...")
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, fpfh

def register(pcd1, pcd2):
    voxel_size = 0.05  # voxel size for downsampling and feature extraction

    # Preprocess both point clouds
    pcd1_down, pcd1_fpfh = preprocess_point_cloud(pcd1, voxel_size)
    pcd2_down, pcd2_fpfh = preprocess_point_cloud(pcd2, voxel_size)

    distance_threshold = voxel_size * 1.5
    print("==> Performing RANSAC global registration...")
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd2_down, pcd1_down, pcd2_fpfh, pcd1_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,  # number of RANSAC correspondence points
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

    print("==> Global registration done. Refining with ICP...")
    distance_threshold_icp = voxel_size * 0.4
    result_icp = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1, distance_threshold_icp,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    print("==> ICP refinement done.")
    return result_icp.transformation
