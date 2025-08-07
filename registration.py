import open3d as o3d
import numpy as np

def register(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> np.ndarray:
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

    voxel_size = 0.05
    pcd1_down, pcd1_fpfh = preprocess_point_cloud(pcd1, voxel_size)
    pcd2_down, pcd2_fpfh = preprocess_point_cloud(pcd2, voxel_size)
    
    distance_threshold = voxel_size * 0.5

    # fast global registration 
    result_fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        pcd1_down, pcd2_down, pcd1_fpfh, pcd2_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        )
    )

    print("FGR transformation:")
    print(result_fgr.transformation)


    #icp point to point
    trans_init = result_fgr.transformation
    icp_threshold = 0.02

    pcd1_icp = pcd1.voxel_down_sample(0.05)
    pcd2_icp = pcd2.voxel_down_sample(0.05)
    pcd1_icp.estimate_normals()
    pcd2_icp.estimate_normals()


    print("Apply point-to-plane ICP")
    reg_p2l = o3d.pipelines.registration.registration_icp(
        pcd1_icp, pcd2_icp, icp_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    print("ICP result:")
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)


    return reg_p2l.transformation