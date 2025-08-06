import open3d as o3d
import numpy as np

# Step 1: Preprocess the point clouds using Voxel Downsampling, Normal Estimation and FPFH Feature Computation
# This function will be used to prepare the point clouds for registration.

def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float):
    print(f"[INFO] Downsampling point cloud with voxel size: {voxel_size}")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    print(f"[INFO] Estimating normals with radius: {voxel_size * 2}")
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2,
            max_nn=30
        )
    )

    print(f"[INFO] Computing FPFH features with radius: {voxel_size * 5}")
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5,
            max_nn=100
        )
    )

    return pcd_down, fpfh

# Step 2: Implement RANSAC-based registration using FPFH features. This is the global registration step.
# It will find an initial alignment between the two point clouds. 
# I cannot use ICP (local registration) directly here because the point clouds are too misaligned.

def run_ransac_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    print("[INFO] Running global registration")
    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,  
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    return result.transformation

# Step 2.1: Implement Fast Global Registration (FGR) as an alternative to RANSAC.

def run_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    print("[INFO] Running Fast Global Registration")
    
    distance_threshold = voxel_size * 1.5  

    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        )
    )

    return result.transformation


# Step 3: Refine the registration using ICP (Iterative Closest Point) algorithm.
# This step will fine-tune the alignment found by RANSAC.
# I will use point-to-plane ICP for better accuracy.

def refine_registration_with_icp(source_down, target_down, init_transformation, voxel_size):
    print("[INFO] Refining alignment using point-to-plane ICP")
    distance_threshold = voxel_size * 0.4  

    result_icp = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        distance_threshold,
        init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    return result_icp.transformation


def register(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> np.ndarray:
    downpcd1, fpfh1 = preprocess_point_cloud(pcd1, voxel_size=0.05)
    downpcd2, fpfh2 = preprocess_point_cloud(pcd2, voxel_size=0.05)

    ransac_trans = run_fast_global_registration(downpcd1, downpcd2, fpfh1, fpfh2, voxel_size=0.05)

    print("[INFO] Running local refinement")
    icp_trans = refine_registration_with_icp(downpcd1, downpcd2, ransac_trans, voxel_size=0.05)

    return icp_trans
