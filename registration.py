import open3d as o3d
import numpy as np
import time

voxel_size = 0.05

def randomly_remove_points(pcd, keep_ratio=0.8, seed=None):
    if seed is not None:
        np.random.seed(seed)
    num_points = np.asarray(pcd.points).shape[0]
    num_keep = int(num_points * keep_ratio)
    indices = np.random.choice(num_points, num_keep, replace=False)
    return pcd.select_by_index(indices)

def preprocess_point_cloud(pcd, voxel_size):
    radius_normal = voxel_size * 5
    print(":: Estimate normals (radius %.3f)" % radius_normal)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 7
    print(":: Compute FPFH features (radius %.3f)" % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=50)  # reduced max_nn
    )

    return pcd, pcd_fpfh

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Fast global registration (threshold %.3f)" % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        )
    )
    return result

def register(pcd, pcd_transformed, keep_ratio=0.8, seed=None):

    print(":: Voxel downsample (voxel size %.3f)" % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_transformed_down = pcd_transformed.voxel_down_sample(voxel_size)

    pcd_down = randomly_remove_points(pcd_down, keep_ratio=keep_ratio, seed=seed)
    pcd_transformed_down = randomly_remove_points(pcd_transformed_down, keep_ratio=keep_ratio, seed=seed)

    source_down, source_fpfh = preprocess_point_cloud(pcd_down, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pcd_transformed_down, voxel_size)

    result = execute_fast_global_registration(source_down, target_down,
                                              source_fpfh, target_fpfh,
                                              voxel_size)

    return result.transformation
