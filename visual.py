import sys
import math
import torch
import argparse
import numpy as np
import open3d as o3d

colors_map = np.array(
    [
        [128, 128, 128, 255],   # 0  其他，灰色
        # [255, 0, 0, 255],       # 1  障碍物，红色           # 1  障碍物，红色
        [255, 136, 132, 255],       # 1  障碍物，红色           # 1  障碍物，红色
        [0, 255, 0, 255],       # 2  自行车，绿色
        [255, 255, 0, 255],     # 3  公交车，黄色
        [0, 0, 255, 255],       # 4  汽车，蓝色
        [255, 165, 0, 255],     # 5  工程车辆，橙色
        [128, 0, 128, 255],     # 6  摩托车，紫色
        [255, 192, 203, 255],   # 7  行人，粉色
        # [255, 69, 0, 255],      # 8  交通锥，橘红色         # 8  交通锥，橘红色
        [248, 172, 140, 255],      # 8  交通锥，橘红色         # 8  交通锥，橘红色
        [192, 192, 192, 255],   # 9  拖车，银色
        [139 ,69 ,19 ,255],     # 10 卡车，棕色
        # [135 ,206 ,235 ,255],   # 11 可行驶表面，天蓝色      # 11 可行驶表面，天蓝色
        [154, 201, 219 ,255],   # 11 可行驶表面，天蓝色      # 11 可行驶表面，天蓝色
        [160 ,82 ,45 ,255],     # 12 其他平坦表面，褐色      # 12 其他平坦表面，褐色
        [211 ,211 ,211 ,255],   # 13 人行道，浅灰色         # 13 人行道，浅灰色
        [139 ,105 ,20 ,255],    # 14 地形，深黄土色         # 14 地形，深黄土色
        [112 ,128 ,144 ,255],   # 15 人造物，灰石色         # 15 人造物，灰石色
        [34 ,139 ,34 ,255]      # 16 植被，森林绿           # 16 植被，森林绿
    ])
erase_label = np.array([0, 2, 3, 4, 5, 6, 7, 9, 10])

def load_kitti_trajectory(file_path):
    """
    加载KITTI轨迹文件
    """
    poses = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            values = line.split()
            # 转换为4x4齐次矩阵
            pose = np.array([[float(values[0]), float(values[1]), float(values[2]), float(values[3])],
                             [float(values[4]), float(values[5]), float(values[6]), float(values[7])],
                             [float(values[8]), float(values[9]), float(values[10]), float(values[11]) + 1],
                             [0, 0, 0, 1]])
            poses.append(pose)
    return poses

def create_frustum_edges(scale=1.0):
    # 创建视锥体，底面初识朝向x轴正方向
    points = np.array([[0, 0, 0],  # apex
                       [scale*1, scale, scale],  # base corners
                       [scale*1, -scale, scale],
                       [scale*1, scale, -scale],
                       [scale*1, -scale, -scale]])
    
    lines = [[0, 1], [0, 2], [0, 3], [0, 4],  # apex to base corners
             [1, 2], [2, 4], [4, 3], [3, 1]]  # base edges

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for i in range(len(lines))])  # Blue color

    return line_set

def pcd_to_voxel(pcd, voxel_size):
    # 将点云坐标向下取整到最近的栅格点
    voxel = np.floor(pcd[:, 0:3] / voxel_size).astype(int) * 0.4
    voxel = np.c_[voxel, pcd[:, 3:]]

    # 找出唯一的栅格点
    unique_voxel = np.unique(voxel, axis=0)

    return unique_voxel

def convert_o3d(pcd, colors_map):
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd[:, 0:3])

    # lable, round, observe need to be deal with
    colors = np.zeros((pcd.shape[0], 3))
    colors = colors_map[np.int64(pcd[:, 3])][:, 0:3] / 255
    o3d_pcd.colors = o3d.utility.Vector3dVector(colors)

    lro_info = pcd[:, 3:6]

    return o3d_pcd, lro_info

def voxel_profile(voxel, voxel_size):
    centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)
    # centers = voxel
    wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                          torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                          torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]), dim=1)
    yaw = torch.full_like(centers[:, 0:1], 0)
    return torch.cat((centers, wlh, yaw), dim=1)

def my_compute_box_3d(center, size, heading_angle):
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    heading_angle = -heading_angle - math.pi / 2
    center[:, 2] = center[:, 2] + h / 2
    #R = rotz(1 * heading_angle)
    l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
    x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
    z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
    #corners_3d = R @ torch.vstack([x_corners, y_corners, z_corners])
    corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]

    return corners_3d


if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxel_size", default=0.4, type=float, help="voxel size")
    parser.add_argument("--traj", required=True, type=str, help="path to trajectory file")
    parser.add_argument("--map", required=True, type=str, help="path to map file")
    args = parser.parse_args()

    pcd = np.load(args.map)
    pcd = pcd_to_voxel(pcd, args.voxel_size)

    o3d_pcd, _ = convert_o3d(pcd, colors_map)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("map")
    vis.add_geometry(o3d_pcd)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])

    view_control = vis.get_view_control()
    view_control.set_lookat(np.array([0, 0, 0]))

    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=1.6, origin=[0, 0, 0])
    # vis.add_geometry(mesh_frame)

    voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        o3d_pcd, voxel_size=args.voxel_size)
    vis.add_geometry(voxelGrid)

    # voxelize
    bboxes = voxel_profile(torch.tensor(pcd[:, 0:4]), [args.voxel_size, args.voxel_size, args.voxel_size])
    bboxes_corners = my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])
    bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
    bbox_corners = bboxes_corners.numpy()

    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])  # lines along y-axis
    edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)
    edges = edges + bases_[:, None, None]
    linesets = edges.numpy()

    line_sets = o3d.geometry.LineSet()
    line_sets.points = o3d.open3d.utility.Vector3dVector(bbox_corners.reshape((-1, 3)))
    line_sets.lines = o3d.open3d.utility.Vector2iVector(linesets.reshape((-1, 2)))
    line_sets.paint_uniform_color((0, 0, 0))
    vis.add_geometry(line_sets)
    
    # add trajectory
    poses = load_kitti_trajectory(args.traj)

    trajectory = o3d.geometry.LineSet()
    points = [pose[:3, 3] for pose in poses]
    trajectory.points = o3d.utility.Vector3dVector(points)
    lines = [[i, i + 1] for i in range(len(poses) - 1)]
    trajectory.lines = o3d.utility.Vector2iVector(lines)
    trajectory.colors = o3d.utility.Vector3dVector([[0, 0, 1] for i in range(len(lines))])
    vis.add_geometry(trajectory)

    for pose in poses:
        frustum_edges = create_frustum_edges(1)
        frustum_edges.transform(pose)
        vis.add_geometry(frustum_edges)
    
    vis.poll_events()
    vis.update_renderer()

    vis.run()

    vis.destroy_window()
    del vis