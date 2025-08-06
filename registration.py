import open3d as o3d
import numpy as np

# Step 1: Preprocess the point clouds using Voxel Downsampling and Normal Estimation
# This function will be used to prepare the point clouds for registration.
def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    print(f"[INFO] Downsampling point cloud with voxel size: {voxel_size}")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    print(f"[INFO] Estimating normals with radius: {voxel_size * 2}")
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30
        )
    )
    
    return pcd_down

def register(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> np.ndarray:
    downpcd1 = preprocess_point_cloud(pcd1, voxel_size=0.05)
    downpcd2 = preprocess_point_cloud(pcd2, voxel_size=0.05)

    

    return np.identity(4)
