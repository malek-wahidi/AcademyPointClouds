import open3d as o3d
import numpy as np
from open3d.pipelines.registration import FastGlobalRegistrationOption

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
    """
    Registration using only Fast Global Registration (FGR).
    """
    voxel_size = 0.07

    print(f"Original point clouds: {len(pcd1.points)} and {len(pcd2.points)} points")
    pcd1_down, pcd1_fpfh = preprocess_point_cloud(pcd1, voxel_size)
    pcd2_down, pcd2_fpfh = preprocess_point_cloud(pcd2, voxel_size)
    print(f"Downsampled to: {len(pcd1_down.points)} and {len(pcd2_down.points)} points")

    distance_threshold = voxel_size * 0.5  # Distance threshold for FGR

    print("==> Performing Fast Global Registration...")
    result_fast = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        pcd2_down, pcd1_down, pcd2_fpfh, pcd1_fpfh,
        FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold)
    )

    print(f"Fast Registration fitness: {result_fast.fitness:.4f}")
    print(f"Fast Registration RMSE: {result_fast.inlier_rmse:.6f}")
    print("==> Registration complete.")

    return result_fast.transformation
