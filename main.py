import open3d as o3d
import numpy as np
import time
from registration import register

def main():
    # Load two misaligned point clouds
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    pcd = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    pcd.paint_uniform_color([1, 0.706, 0])
    pcd_transformed = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
    pcd_transformed.paint_uniform_color([0, 0.651, 0.929])

    # Apply extra misalignment to source
    rotation = o3d.geometry.get_rotation_matrix_from_xyz([0.3, 0.5, 0.2])
    translation = np.array([2.0, 1.5, 1.0])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    pcd.transform(transform)

    print(f"Applied misalignment: rotation={[0.3, 0.5, 0.2]}, translation={translation}")
    print("Visualizing BEFORE registration")
    o3d.visualization.draw_geometries([pcd, pcd_transformed],
                                      window_name="Before Registration")

    # Run registration and time it
    start = time.time()
    transformation = register(pcd, pcd_transformed)
    end = time.time()

    print(f"Registration took {end - start:.2f} seconds.")
    print("Transformation matrix (pcd → pcd_transformed):")
    print(transformation)

    # Apply the transformation to align pcd to pcd_transformed
    pcd.transform(transformation)

    # Evaluate the result
    distance_threshold = 0.02
    eval_result = o3d.pipelines.registration.evaluate_registration(
        pcd, pcd_transformed, distance_threshold)

    print("Registration evaluation:")
    print(f"  Fitness: {eval_result.fitness * 100:.2f} %")
    print(f"  Inlier RMSE: {eval_result.inlier_rmse:.4f}")
    print(f"  Correspondences: {len(eval_result.correspondence_set)}")

    print("Visualizing AFTER registration")
    o3d.visualization.draw_geometries([pcd, pcd_transformed],
                                      window_name="After Registration")

if __name__ == "__main__":
    main()
