import open3d as o3d
import numpy as np


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000, 0.95)
    )
    return result


def refine_registration(source, target, init_transformation, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)

    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=distance_threshold,
        init=init_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result


def register(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> np.ndarray:
    voxel_size = 0.1

    print(":: Preprocessing point clouds...")
    source_down, source_fpfh = preprocess_point_cloud(pcd1, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pcd2, voxel_size)

    print(":: Performing global RANSAC registration...")
    result_ransac = execute_global_registration(
        source_down, target_down,
        source_fpfh, target_fpfh,
        voxel_size
    )
    print("RANSAC Fitness:", result_ransac.fitness)

    if result_ransac.fitness == 0:
        print("⚠️ Global registration failed — using identity.")
        return np.identity(4)

    print(":: Refining with point-to-plane ICP...")
    result_icp = refine_registration(pcd1, pcd2, result_ransac.transformation, voxel_size)
    print("ICP Fitness:", result_icp.fitness)
    print("ICP RMSE:", result_icp.inlier_rmse)

    return result_icp.transformation