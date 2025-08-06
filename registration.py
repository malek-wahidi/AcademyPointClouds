import open3d as o3d
import numpy as np


def preprocess_point_cloud(pcd, voxel_size):
    
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5

    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh
# In my first try, I tried the global registration using RANSAC, but it was too slow (7.3 seconds)
# So I switched to Fast Global Registration (FGR) which is much faster and works well (0.32 seconds)

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
   
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def refine_registration(source, target,result_fgr, voxel_size):
    distance_threshold = voxel_size * 0.4

    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_fgr.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def register(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> np.ndarray:
    voxel_size = 0.05  
    
    # First we preprocess the point clouds
    source_down, source_fpfh = preprocess_point_cloud(pcd1, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pcd2, voxel_size)
    
    # We then perform fast global registration
    result_fgr = execute_fast_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    
    # Finally, we refine the registration using ICP
    result_icp = refine_registration(pcd1, pcd2, result_fgr, voxel_size)


    return result_icp.transformation

