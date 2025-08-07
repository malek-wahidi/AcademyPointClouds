
import open3d as o3d
import numpy as np

def preprocess_point_cloud(pcd, voxel_size):
    """Optimized preprocessing with better parameters."""
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
    Optimized registration function that aligns pcd2 to pcd1.
    
    Key optimizations:
    - Better voxel size for speed/accuracy balance
    - Optimized RANSAC parameters
    - Point-to-plane ICP for better convergence
    - Reduced iterations where appropriate
    """
    # Optimized voxel size - slightly larger for speed but still accurate
    voxel_size = 0.2  # Increased from 0.03 for better speed
    
    # Preprocess both point clouds
    print(f"Original point clouds: {len(pcd1.points)} and {len(pcd2.points)} points")
    pcd1_down, pcd1_fpfh = preprocess_point_cloud(pcd1, voxel_size)
    pcd2_down, pcd2_fpfh = preprocess_point_cloud(pcd2, voxel_size)
    print(f"Downsampled to: {len(pcd1_down.points)} and {len(pcd2_down.points)} points")
    
    # Optimized distance thresholds
    distance_threshold = voxel_size * 1.5
    distance_threshold_icp = voxel_size * 0.4
    
    print("==> Performing RANSAC global registration...")
    # Optimized RANSAC parameters for better speed/accuracy trade-off
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd2_down, pcd1_down, pcd2_fpfh, pcd1_fpfh, 
        mutual_filter=True,  # Use mutual filtering for better correspondences
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,  # Reduced from 4 for speed
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=100000,  # Reduced from 1000000 for speed
            confidence=0.999  # Reduced from 1000 iterations
        )
    )
    
    print(f"Global registration fitness: {result_ransac.fitness:.4f}")
    print(f"Global registration RMSE: {result_ransac.inlier_rmse:.6f}")
    
    # Use Point-to-Plane ICP for better convergence
    print("==> Refining with Point-to-Plane ICP...")
    
    # Ensure normals are computed for full-resolution point clouds
    if not pcd1.has_normals():
        pcd1.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    if not pcd2.has_normals():
        pcd2.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    # Point-to-Plane ICP for better accuracy
    result_icp = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1, distance_threshold_icp,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),  # Changed to Point-to-Plane
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)  # Reduced from 2000
    )
    
    print(f"Final ICP fitness: {result_icp.fitness:.4f}")
    print(f"Final ICP RMSE: {result_icp.inlier_rmse:.6f}")
    print("==> Registration complete.")
    
    return result_icp.transformation
