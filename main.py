import open3d as o3d
import numpy as np
import time
from registration import register

def main():
    # Load two misaligned point clouds from Open3D's demo dataset
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    pcd = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])  # load source point cloud
    pcd.paint_uniform_color([1, 0.706, 0])  # paint source point cloud red
    pcd_transformed = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])  # load target point cloud
    pcd_transformed.paint_uniform_color([0, 0.651, 0.929])  # paint target point cloud green
    
    # Apply additional transformation to push point clouds further apart
    # Create a more challenging initial misalignment
    additional_rotation = o3d.geometry.get_rotation_matrix_from_xyz([0.3, 0.5, 0.2])  # radians
    additional_translation = np.array([2.0, 1.5, 1.0])  # larger translation
    additional_transform = np.eye(4)
    additional_transform[:3, :3] = additional_rotation
    additional_transform[:3, 3] = additional_translation
    
    pcd_transformed.transform(additional_transform)
    print(f"Applied additional misalignment: rotation={[0.3, 0.5, 0.2]} rad, translation={additional_translation}")

    print("Visualizing source and target point clouds before registration.")
    o3d.visualization.draw_geometries([pcd, pcd_transformed],
                                      window_name="Before Registration")

    # Time the registration function
    start_time = time.time()
    transformation = register(pcd, pcd_transformed)
    print("Transformation matrix:\n", transformation)

    end_time = time.time()

    print(f"Registration took {end_time - start_time:.4f} seconds.")

    # Compute registration accuracy metrics

    # Apply the estimated transformation
    pcd_transformed.transform(transformation)

    # Compute registration accuracy metrics
    distance_threshold = 0.02  # reasonable threshold for correspondence
    evaluation = o3d.pipelines.registration.evaluate_registration(
        pcd, pcd_transformed, distance_threshold)
    
    print(f"Registration accuracy metrics:")
    print(f"  Fitness: {evaluation.fitness*100:.2f} %")  # fraction of target points with correspondences found
    print(f"  Inlier RMSE: {evaluation.inlier_rmse:.4f}")  # RMSE of corresponding points
    print(f"  Correspondences found: {len(evaluation.correspondence_set)}")
    
    # Visualize the alignment
    print("Visualizing source and target point clouds after registration.")
    o3d.visualization.draw_geometries([pcd, pcd_transformed],
                                      window_name="After Registration")

if __name__ == "__main__":
    main()
