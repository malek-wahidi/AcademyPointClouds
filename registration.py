import open3d as o3d
import numpy as np

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=20)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=50)
    )
    return pcd_down, fpfh


def execute_global_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down,
        src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000, 25)  # ← optimization here
    )
    return result


def refine_registration(src, tgt, initial_transformation, voxel_size):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=distance_threshold,
        init=initial_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result


def register(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> np.ndarray:
    voxel_size = 0.165  # larger = faster, less accurate

    # Step 1: Preprocess both clouds
    pcd1_down, pcd1_fpfh = preprocess_point_cloud(pcd1, voxel_size)
    pcd2_down, pcd2_fpfh = preprocess_point_cloud(pcd2, voxel_size)

    # Debug info
    #print("Source downsampled point count:", np.asarray(pcd1_down.points).shape[0])
    #print("Target downsampled point count:", np.asarray(pcd2_down.points).shape[0])
    #print("FPFH source shape:", pcd1_fpfh.data.shape)
    #print("FPFH target shape:", pcd2_fpfh.data.shape)

    # Step 2: Global alignment with RANSAC
    result_ransac = execute_global_registration(pcd1_down, pcd2_down, pcd1_fpfh, pcd2_fpfh, voxel_size)
    print("Estimated transformation from RANSAC:")
    print(result_ransac.transformation)

    # Step 3: Refine with point-to-plane ICP
    result_icp = refine_registration(pcd1, pcd2, result_ransac.transformation, voxel_size)
    print("Refined transformation after ICP:")
    print(result_icp.transformation)

    return result_icp.transformation
