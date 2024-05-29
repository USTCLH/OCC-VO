import os 
import re
import copy
import argparse
import numpy as np

from OCC_VO import OCC_VO

colors_map = np.array(
    [
        [128, 128, 128, 255],   # 0  其他，灰色
        [255, 0, 0, 255],       # 1  障碍物，红色           # 1  障碍物，红色
        [0, 255, 0, 255],       # 2  自行车，绿色
        [255, 255, 0, 255],     # 3  公交车，黄色
        [0, 0, 255, 255],       # 4  汽车，蓝色
        [255, 165, 0, 255],     # 5  工程车辆，橙色
        [128, 0, 128, 255],     # 6  摩托车，紫色
        [255, 192, 203, 255],   # 7  行人，粉色
        [255, 69, 0, 255],      # 8  交通锥，橘红色         # 8  交通锥，橘红色
        [192, 192, 192, 255],   # 9  拖车，银色
        [139 ,69 ,19 ,255],     # 10 卡车，棕色
        [135 ,206 ,235 ,255],   # 11 可行驶表面，天蓝色      # 11 可行驶表面，天蓝色
        [160 ,82 ,45 ,255],     # 12 其他平坦表面，褐色      # 12 其他平坦表面，褐色
        [211 ,211 ,211 ,255],   # 13 人行道，浅灰色         # 13 人行道，浅灰色
        [139 ,105 ,20 ,255],    # 14 地形，深黄土色         # 14 地形，深黄土色
        [112 ,128 ,144 ,255],   # 15 人造物，灰石色         # 15 人造物，灰石色
        [34 ,139 ,34 ,255]      # 16 植被，森林绿           # 16 植被，森林绿
    ])
dynamic_label = np.array([0, 2, 3, 4, 5, 6, 7, 9, 10])

def convert_pose(pose_string):
    pose_str = pose_string.split(' ')
    pose = np.zeros((4, 4))

    for i in range(pose.shape[0] - 1):
        for j in range(pose.shape[1]):
            pose[i, j] = float(pose_str[i * pose.shape[0] + j])         # nuscenes
            # pose[i, j] = float(pose_str[i * pose.shape[0] + j + 1])         # kitti
    pose[pose.shape[0] - 1][pose.shape[1] - 1] = 1

    return pose

def get_trans_init(path):
    with open(path, 'r') as fr:
        pose0 = convert_pose(fr.readline())
        pose1 = convert_pose(fr.readline())

    trans_init = np.dot(np.linalg.inv(pose0), pose1)

    return trans_init

def write_pose(f, pose, z_zero=True):
    pose = copy.deepcopy(pose)
    line = ''
    # line = str(time)
    if(z_zero):
        pose[2,3] = 0
    for j in range(3):
        line += ' '.join([str(num) for num in pose[j]])
        line += " "
    f.write(line[:-1] + '\n')

def points2pcd(pcd, path):
    if os.path.exists(path):
        os.remove(path)

    #写文件句柄
    handle = open(path, 'a')

    #得到点云点数
    point_num = pcd.shape[0]

    #pcd头部（重要）
    handle.write('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA ascii')

    # 依次写入点
    for i in range(point_num):
        string = '\n' + str(pcd[i, 0]) + ' ' + str(pcd[i, 1]) + ' ' + str(pcd[i, 2])
        handle.write(string)
    handle.close()


if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help="path to input directory")
    parser.add_argument("--trans_init", default=None, type=str, help="path to trajectory groundtruth file (Kitti format) to get trans_init. Not required, but make it better.")
    parser.add_argument("--output_traj", default=None, type=str, help="path to trajectory output directory")
    parser.add_argument("--output_map", default=None, type=str, help="path to map output directory. Prioritize outputting the presentation map")

    parser.add_argument("--voxel_size", default=0.4, type=float, help="voxel size")
    parser.add_argument("--round_new", default=1, type=float, help="Voxel PFilter param, reserve points at theirs preceding n rounds")
    parser.add_argument("--threshold", default=1, type=float, help="Voxel PFilter param, reserve points with p-Index greater than or equal to n")
    parser.add_argument("--observed_times", default=5, type=float, help="Voxel PFilter param, reserve points that have been observed n or more times")

    # parser.add_argument("--slide_window", default=False, type=bool, help="slide window param, use slide window flag")
    # parser.add_argument("--window_size", default=5, type=int, help="slide window param, each window contain n frames")
    # parser.add_argument("--window_gap", default=3, type=int, help="slide window param, each window gap n frames")

    parser.add_argument("--pres_map", default=True, type=bool, help="generate a denser presentation map. Set false to run faster")
    parser.add_argument("--pres_z0", default=True, type=bool, help="set the z dimension of the presentation map's poses to 0")
    # parser.add_argument("--visual", default=False, type=bool, help="draw the generated map")
    args = parser.parse_args()

    pose_init = np.identity(4)
    trans_init = np.identity(4)

    if args.trans_init is not None:
        # use the first two frames gt to get trans_init in front
        trans_init = get_trans_init(args.trans_init)
        trans_init_x = np.linalg.norm(trans_init[0:3, 3])
        trans_init[0, 3] = trans_init_x
    else:
        # use 0.5 meters in front as init
        trans_init[0, 3] = 0.5

    basename = os.path.basename(args.input)
    if args.output_traj is not None:
        os.makedirs(args.output_traj, exist_ok=True)
        fw = open(os.path.join(args.output_traj, basename + ".txt"), 'w')

    file_list = os.listdir(args.input)
    file_list = sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group()))

    for i in range(len(file_list)):
        print(f"runing frame:{i+1}/{len(file_list)}", end='\r')

        target = np.load(os.path.join(args.input, file_list[i]))

        if i == 0:
            pmap = OCC_VO(args, target, trans_init, pose_init, dynamic_label, colors_map)
        else:
            pmap.registration_icp(target)

        # print(pmap.pose)
        # pmap.draw_pcd(target, with_ax=True)

        if args.output_traj is not None:
            write_pose(fw, copy.deepcopy(pmap.pose))

    if args.output_traj is not None:
        fw.close()

    map_result = pmap.presentation if args.pres_map else pmap.pcd
    map_result = pmap.filter_by_label(map_result, pmap.dynamic_label)

    # if args.visual:
    #     pmap.draw_pcd(map_result)

    if args.output_map is not None:
        os.makedirs(args.output_map, exist_ok=True)
        np.save(os.path.join(args.output_map, basename + '.npy'), map_result)
        points2pcd(map_result, os.path.join(args.output_map, basename + '.pcd'))
