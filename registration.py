import open3d as o3d
import numpy as np

def preprocess(pcd, voxel_size):
    """
    Downsamples the point cloud, estimates normals, and computes FPFH features.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, fpfh

def register(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Registers source point cloud pcd1 to target point cloud pcd2.
    Returns the 4x4 transformation matrix that aligns pcd1 to pcd2.
    """
    voxel_size = 0.05  # Adjust depending on the scale of your point clouds

    # Step 1: Preprocess both point clouds
    pcd1_down, fpfh1 = preprocess(pcd1, voxel_size)
    pcd2_down, fpfh2 = preprocess(pcd2, voxel_size)

    # Step 2: Global registration using RANSAC and FPFH features
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=pcd1_down,
        target=pcd2_down,
        source_feature=fpfh1,
        target_feature=fpfh2,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(40000, 500)
    )

    # Step 3: Local refinement using ICP (Point-to-Plane)
    icp_result = o3d.pipelines.registration.registration_icp(
        source=pcd1,
        target=pcd2,
        max_correspondence_distance=voxel_size * 1.5,
        init=ransac_result.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    # Optional: Print transformation matrix
    print("Estimated Transformation:\n", icp_result.transformation)

    return icp_result.transformation
