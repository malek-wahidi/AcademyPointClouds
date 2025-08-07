import open3d as o3d
import numpy as np

voxel_size = 0.02
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5


def load_point_clouds(voxel_size=0.0):
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    pcd = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    pcd.paint_uniform_color([1, 0.706, 0]) 
    pcd_transformed = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
    pcd_transformed.paint_uniform_color([0, 0.651, 0.929])

    return [pcd.voxel_down_sample(voxel_size), pcd_transformed.voxel_down_sample(voxel_size)]


def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=True))
    return pose_graph


if __name__ == "__main__":


    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    # Load point clouds
    pcds_down = load_point_clouds(voxel_size)
    source, target = pcds_down[0], pcds_down[1]

    # Run pairwise ICP registration
    transformation_icp, information_icp = pairwise_registration(source, target)

    # Print transformation matrix
    print("Transformation matrix (from source to target):")
    print(transformation_icp)

    # Apply transformation to source point cloud
    source.transform(transformation_icp)

    # Visualize alignment (optional)
    o3d.visualization.draw_geometries([source, target])



    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    # pcds_down = load_point_clouds(voxel_size)
    # o3d.visualization.draw_geometries(pcds_down)

    # print("Full registration ...")
    # pose_graph = full_registration(pcds_down,
    #                                max_correspondence_distance_coarse,
    #                                max_correspondence_distance_fine)

    # print("Optimizing PoseGraph ...")
    # option = o3d.pipelines.registration.GlobalOptimizationOption(
    #     max_correspondence_distance=max_correspondence_distance_fine,
    #     edge_prune_threshold=0.25,
    #     reference_node=0)
    # o3d.pipelines.registration.global_optimization(
    #     pose_graph, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
    #     o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)

    # print("Transform points and display")
    # for point_id in range(len(pcds_down)):
    #     print(pose_graph.nodes[point_id].pose)
    #     pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    # o3d.visualization.draw_geometries(pcds_down)

    # print("Make a combined point cloud")
    # pcds = load_point_clouds(voxel_size)
    # pcd_combined = o3d.geometry.PointCloud()
    # for point_id in range(len(pcds)):
    #     pcds[point_id].transform(pose_graph.nodes[point_id].pose)
    #     pcd_combined += pcds[point_id]
    # pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    # o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
    # o3d.visualization.draw_geometries([pcd_combined_down])