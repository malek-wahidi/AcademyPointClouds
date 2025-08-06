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


def preprocess_point_cloud_partial(pcd, voxel_size):
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        return pcd_down


def execute_fast_global_registration(pcd1_down, pcd2_down, pcd1_fpfh,
                                     pcd2_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.6
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        pcd1_down, pcd2_down, pcd1_fpfh, pcd2_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def refine_registration(source, target, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    loss = o3d.pipelines.registration.TukeyLoss(k=1.5*distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(loss))
    return result


def register(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> np.ndarray:

    voxel_size = 0.16
    
    pcd1_down, pcd1_fpfh = preprocess_point_cloud(pcd1, voxel_size)
    pcd2_down, pcd2_fpfh = preprocess_point_cloud(pcd2, voxel_size)


    result_ransac = execute_fast_global_registration(pcd1_down, pcd2_down,
                                            pcd1_fpfh, pcd2_fpfh,
                                            voxel_size)
        
    voxel_size = 0.07
    
    pcd1_down = preprocess_point_cloud_partial(pcd1, voxel_size)
    pcd2_down = preprocess_point_cloud_partial(pcd2, voxel_size)

    result_icp = refine_registration(pcd1_down, pcd2_down, result_ransac, voxel_size)    

    return result_icp.transformation  
