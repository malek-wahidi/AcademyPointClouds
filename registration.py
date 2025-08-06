import open3d as o3d
import numpy as np

# Step 0: Remove outliers from the point clouds to improve registration quality.
# This hasn't improved accuracy at all and added to the run time
def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    filtered_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return filtered_pcd

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
# This step will fine-tune the alignment found by RANSAC (or FGR as an alternative).
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

# Step 3.2: Implement Multiscale ICP for better convergence by using different voxel sizes.

def multiscale_icp(source, target, init_transformation):
    voxel_levels = [0.05, 0.03, 0.01]
    max_iters = [50, 30, 14]

    current_transformation = init_transformation

    for size in range(len(voxel_levels)):
        voxel_size = voxel_levels[size]
        iter = max_iters[size]

        print(f"\n[Multiscale ICP] Level {size+1} - voxel: {voxel_size}, max_iter: {iter}")
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)

        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )

        threshold = voxel_size * 1.5

        result_icp = o3d.pipelines.registration.registration_icp(
            source_down,
            target_down,
            threshold,
            current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iter)
        )

        current_transformation = result_icp.transformation

        # This is a local evaluation step to check the quality of the registration at this level.
        print(f"  -> Fitness: {result_icp.fitness:.2%}, RMSE: {result_icp.inlier_rmse:.6f}")

    return current_transformation


def register(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> np.ndarray:
    voxel_size = 0.05 

    downpcd1, fpfh1 = preprocess_point_cloud(pcd1, voxel_size)
    downpcd2, fpfh2 = preprocess_point_cloud(pcd2, voxel_size)

    fgr_trans = run_fast_global_registration(downpcd1, downpcd2, fpfh1, fpfh2, voxel_size)

    final_trans = refine_registration_with_icp(downpcd1, downpcd2, fgr_trans, voxel_size)

    #final_trans = multiscale_icp(pcd1, pcd2, fgr_trans)
    
    # I did stick to the FGR + ICP approach since it is faster, with the same accuracy as multiscale ICP in most cases.
    return final_trans


# I explored another alternative, which is to use deep learning based registration methods (like FCGF or PointNetLk) but this would add significant complexity
# and dependencies for just small improvements in registration accuracy. Sticking to traditional methods like FGR and ICP is more practical in this case.
