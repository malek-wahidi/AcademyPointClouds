import open3d as o3d
import numpy as np

def execute_fgr_from_correspondance(source_down, target_down,source_fpfh, target_fpfh, voxel_size):

    fgr_option = o3d.pipelines.registration.FastGlobalRegistrationOption()
    fgr_option.maximum_correspondence_distance = voxel_size * 2
    fgr_option.iteration_number= 32
    fgr_option.maximum_tuple_count = 500
# Modified some options for speed
    corres = o3d.pipelines.registration.correspondences_from_features(
            source_fpfh, target_fpfh)
# If we dont customize/use correspondence from another method, fgr based on features is almost identical to 
# fgr based on correspondence, however I found it to be slightly faster
    result = o3d.pipelines.registration.registration_fgr_based_on_correspondence(
        source_down,
        target_down,
        corres,
        fgr_option
    )
    return result


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    _, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
    pcd_down = pcd_down.select_by_index(ind)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh



def icp_registration(source, target, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4

    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


def register(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> np.ndarray:

    voxel_size = 0.05
    
    source_down, source_fpfh = preprocess_point_cloud(pcd1, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pcd2, voxel_size)
   
    result_ransac = execute_fgr_from_correspondance(source_down, target_down, source_fpfh,target_fpfh, voxel_size)
    
    result_icp = icp_registration(source_down, target_down, voxel_size , result_ransac)

    return result_icp.transformation
