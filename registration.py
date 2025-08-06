import open3d as o3d
import numpy as np

def preprocess_point_cloud(pcd, voxel_size):
    print(f":: Downsample with voxel size {voxel_size}")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 3
    print(f":: Estimate normal with search radius {radius_normal:.3f}")
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 6
    print(f":: Compute FPFH feature with search radius {radius_feature:.3f}")
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh


def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(f":: Apply fast global registration with distance threshold {distance_threshold:.3f}")
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        )
    )
    return result


def refine_registration(source, target, initial_transformation, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(f":: Refining registration with ICP and distance threshold {distance_threshold:.3f}")
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold,
        initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result_icp


def register(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> np.ndarray:
    voxel_size = 0.04  # Smaller voxel size improves accuracy

    source_down, source_fpfh = preprocess_point_cloud(pcd1, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pcd2, voxel_size)

    result_fgr = execute_fast_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    print(f"FGR Result - Fitness: {result_fgr.fitness:.4f}, RMSE: {result_fgr.inlier_rmse:.6f}")

    result_icp = refine_registration(pcd1, pcd2, result_fgr.transformation, voxel_size)

    print(f"ICP Refinement - Fitness: {result_icp.fitness:.4f}, RMSE: {result_icp.inlier_rmse:.6f}")

    return result_icp.transformation
