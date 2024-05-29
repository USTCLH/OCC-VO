import copy
import math
import numpy as np
import open3d as o3d

class OCC_VO():
    def __init__(self, args, pcd, trans_init, pose_init, dynamic_label, colors_map=None):
        self.trans = trans_init
        self.pose  = pose_init

        self.dynamic_label = dynamic_label
        self.colors_map    = colors_map

        self.voxel_size     = args.voxel_size
        self.threshold      = args.threshold
        self.round_new      = args.round_new
        self.observed_times = args.observed_times

        self.pres_map = args.pres_map
        self.pres_z0  = args.pres_z0

        pcd = self.init_map(pcd)
        o3d_pcd, pcd_lro = self.convert_o3d(pcd)
        self.pcd = self.convert_np(o3d_pcd, pcd_lro)

        self.presentation = copy.deepcopy(self.pcd)

    # x, y, z, lable, round, observe
    def init_map(self, pcd):
        round = np.ones((pcd.shape[0], 1))
        observe = np.zeros((pcd.shape[0], 1))

        pcd = np.c_[pcd, round]
        pcd = np.c_[pcd, observe]

        return pcd

    def filter_by_label(self, pcd, label_list):
        for label in label_list:
            pcd = pcd[pcd[:, 3] != label]

        return pcd

    def select_by_label(self, pcd, label_list):
        mask = np.zeros(pcd.shape[0], dtype=np.bool)
        for label in label_list:
            mask = np.logical_or(mask, pcd[:, 3] == label)

        return pcd[mask, :], mask
    
    def convert_np(self, o3d_pcd, lro_info):
        pcd = np.asarray(o3d_pcd.points)
        pcd = np.c_[pcd, lro_info]

        return pcd 
    
    def convert_o3d(self, pcd):
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd[:, 0:3])

        if self.colors_map is not None:
            colors = np.zeros((pcd.shape[0], 3))
            colors = self.colors_map[np.int64(pcd[:, 3])][:, 0:3] / 255
            o3d_pcd.colors = o3d.utility.Vector3dVector(colors)

        lro_info = pcd[:, 3:6]

        return o3d_pcd, lro_info

    def rotationMatrixToEulerAngles(self, R) :
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        
        singular = sy < 1e-6
    
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
    
        return np.array([x, y, z])

    def eulerAngles2rotationMat(self, theta):  
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])
    
        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])
    
        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])
        R = np.dot(R_z, np.dot(R_y, R_x))

        return R

    def set_pose_to_z0(self, pose):
        pose_z0 = copy.deepcopy(pose)
        euler_z0 = self.rotationMatrixToEulerAngles(pose_z0)

        euler_z0[0] = 0
        euler_z0[1] = 0
        pose_z0[2: 3] = 0

        pose_z0[0:3, 0:3] = self.eulerAngles2rotationMat(euler_z0)

        return pose_z0

    def filter_dynamic(self, pcd1, pcd2, label_list, max_cluster_dist = 1, 
                        min_cluster_points = 5, max_static_dist = 2):
        pcd1_dynamic, mask_dynamic1 = self.select_by_label(pcd1, label_list)
        pcd2_dynamic, mask_dynamic2 = self.select_by_label(pcd2, label_list)

        # dynamic object not exist
        if mask_dynamic1.sum() == 0 or mask_dynamic2.sum() == 0:
            return self.filter_by_label(pcd1, label_list), self.filter_by_label(pcd2, label_list)

        # use coarse registration result to find dynamic object
        o3d_pcd_dynamic1, _ = self.convert_o3d(pcd1_dynamic)
        o3d_pcd_dynamic2, _ = self.convert_o3d(pcd2_dynamic)

        tmp_pose = self.reg_p2l.transformation @ self.pose
        o3d_pcd_dynamic2 = o3d_pcd_dynamic2.transform(tmp_pose)

        # cluster and find dynamic object
        pcd1_object = np.array(o3d_pcd_dynamic1.cluster_dbscan(eps=max_cluster_dist, min_points=min_cluster_points))
        pcd2_object = np.array(o3d_pcd_dynamic2.cluster_dbscan(eps=max_cluster_dist, min_points=min_cluster_points))

        dists1 = np.array(o3d_pcd_dynamic1.compute_point_cloud_distance(o3d_pcd_dynamic2))
        dists2 = np.array(o3d_pcd_dynamic2.compute_point_cloud_distance(o3d_pcd_dynamic1))
        remove_object1 = np.unique(pcd1_object[dists1 > max_static_dist])
        remove_object2 = np.unique(pcd2_object[dists2 > max_static_dist])

        mask_remove1 = np.zeros(pcd1_object.shape[0])
        for ob in remove_object1:
            mask_remove1 = np.logical_or(mask_remove1, pcd1_object == ob)

        mask_remove2 = np.zeros(pcd2_object.shape[0])
        for ob in remove_object2:
            mask_remove2 = np.logical_or(mask_remove2, pcd2_object == ob)
            
        # remove dynamic object
        index_dynamic1 = np.where(mask_dynamic1 == True)[0]
        index_remove1 = np.where(mask_remove1 == False)[0]
        mask_dynamic1[index_dynamic1[index_remove1]] = False

        index_dynamic2 = np.where(mask_dynamic2 == True)[0]
        index_remove2 = np.where(mask_remove2 == False)[0]
        mask_dynamic2[index_dynamic2[index_remove2]] = False

        return pcd1[np.logical_not(mask_dynamic1)], pcd2[np.logical_not(mask_dynamic2)]
    
    def remove_remote(self, pcd, remote_thresh, pose):
        pcd_o3d, pcd_lro = self.convert_o3d(pcd)
        pcd_o3d.transform(np.linalg.inv(pose))
        pcd = self.convert_np(pcd_o3d, pcd_lro)

        x = pcd[:, 0]
        y = pcd[:, 1]

        # dis_sqaure = x * x + y * y
        # pcd = pcd[dis_sqaure <= remote_thresh * remote_thresh]

        local_flag_x = np.logical_and(x < remote_thresh, x > -remote_thresh)
        local_flag_y = np.logical_and(y < remote_thresh, y > -remote_thresh)
        local_flag= np.logical_and(local_flag_x, local_flag_y)
        pcd = pcd[local_flag]

        return pcd

    # registration module
    def registration_icp(self, target, match_distance_coarse = 1, match_distance_fine = 0.4):
        # we use six dimensions to maintain point data: x y z label round observe
        # 'round' is the rounds it exists, 'observe' is the times it's observed
        source_o3d, _ = self.convert_o3d(self.pcd)
        source_o3d.estimate_normals()

        target = self.init_map(target)
        target_o3d, target_lro = self.convert_o3d(target)

        if self.pres_map and self.pres_z0:
            target_o3d_z0 = copy.deepcopy(target_o3d)
        target_o3d.estimate_normals()
        target_o3d.transform(self.pose)

        # coarse registration, we rewrite this o3d function to implement Semantic Label Filter
        self.reg_p2l = o3d.pipelines.registration.registration_generalized_icp(
            target_o3d, source_o3d, match_distance_coarse, self.trans,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(0.001))

        # Dynamic Object Filter
        self.pcd, target_static = self.filter_dynamic(self.pcd, target, self.dynamic_label)
        source_o3d, _ = self.convert_o3d(self.pcd)
        source_o3d.estimate_normals()

        target_static_o3d, target_static_lro = self.convert_o3d(target_static)
            
        target_static_o3d.estimate_normals()
        target_static_o3d.transform(self.pose)

        # fine registration
        self.reg_p2l = o3d.pipelines.registration.registration_generalized_icp(
            target_static_o3d, source_o3d, match_distance_fine, self.reg_p2l.transformation,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(0.001))

        # registration result
        self.trans = self.reg_p2l.transformation
        self.pose = self.trans @ self.pose 

        # update map
        target_o3d = target_o3d.transform(self.trans)
        target = self.convert_np(target_o3d, target_lro)

        if self.pres_map:
            if self.pres_z0:
                # set pose to z0
                pose_z0 = self.set_pose_to_z0(self.pose)

                # transform target to z0
                target_o3d_z0 = target_o3d_z0.transform(pose_z0)
                target_z0 = self.convert_np(target_o3d_z0, target_lro)

                self.presentation = self.update_map(self.presentation, target_z0, presentation=True)
            else:              
                self.presentation = self.update_map(self.presentation, target, presentation=True)

        # self.pcd = self.remove_remote(self.pcd, 50, self.pose)
        self.pcd = self.update_map(self.pcd, target)

    def update_map(self, source, target, presentation=False):   
        source[:, 4] += 1

        source = np.r_[source, target]
        source = self.filter_voxel(source)              # voxel downsample
        source = self.pfilter(source, presentation)     # Voxel PFilter

        return source

    def pfilter(self, pcd, presentation):
        if presentation:        # for presentation map
            threshold = self.threshold * 0.5
            round_new = self.round_new * 2
            observed_times = self.observed_times * 0.8
        else:                   # for registration map
            threshold = self.threshold
            round_new = self.round_new
            observed_times = self.observed_times

        rate = pcd[:, 5] / pcd[:, 4]
        index1 = (rate >= threshold)                # rate >= threshold
        index2 = (pcd[:, 4] <= round_new)           # round <= round_new
        index3 = (pcd[:, 5] >= observed_times)      # observe >= observed_times

        index = np.logical_or(index1, index2)
        index = np.logical_or(index, index3)

        tmp_threshold = threshold
        while index.sum() / index.shape[0] < 1 / 4:
            tmp_threshold = tmp_threshold - 0.1
            index1 = (rate >= tmp_threshold)
            index = np.logical_or(index, index1)

        return pcd[index]

    def filter_voxel(self, point_cloud):
        filtered_points = []
        # calculate the boundaries of x y z
        x_min, y_min, z_min = np.amin(point_cloud[:, 0:3], axis=0) 
        x_max, y_max, z_max = np.amax(point_cloud[:, 0:3], axis=0)
    
        # calculate voxel num of x y
        Dx = (x_max - x_min)//self.voxel_size + 1
        Dy = (y_max - y_min)//self.voxel_size + 1
    
        # calculate the voxel index for each point
        hx = (point_cloud[:, 0] - x_min)//self.voxel_size
        hy = (point_cloud[:, 1] - y_min)//self.voxel_size
        hz = (point_cloud[:, 2] - z_min)//self.voxel_size
        h = hx + hy*Dx + hz*Dx*Dy
    
        # filter points in each voxel
        h_indice = np.argsort(h)
        h_sorted = h[h_indice]
        begin = 0
        for i in range(len(h_sorted)-1): 
            if h_sorted[i] == h_sorted[i + 1]:
                continue
            else:
                point_idx = h_indice[begin: i + 1]
                if begin == i:
                    filtered_points.append(point_cloud[point_idx][0])
                else:
                    point_tmp = np.zeros(point_cloud.shape[1])
                    xyz_weight = point_cloud[point_idx, 5] + 1
                    max_index_observe = np.argmax(point_cloud[point_idx][:, 5])
                    
                    # point_tmp[0:3] = np.mean(point_cloud[point_idx, 0:3], axis=0)
                    point_tmp[0:3] = np.average(point_cloud[point_idx, 0:3], weights=xyz_weight, axis=0)
                    point_tmp[3:6] = point_cloud[point_idx][max_index_observe, 3:6]

                    point_tmp[5] += (i - begin)

                    filtered_points.append(point_tmp)
                begin = i + 1
    
        filtered_points = np.array(filtered_points, dtype=np.float32)

        return filtered_points 
    
    def draw_pcd(self, pcd, with_ax=False):
        pcd_o3d, _ = self.convert_o3d(pcd)

        draw_result = [pcd_o3d]

        # include axes
        if with_ax:
            ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
            ax.transform(self.pose)
            draw_result.append(ax)

        o3d.visualization.draw_geometries(draw_result,
                                        zoom=0.4459,
                                        front=[0,0,1],
                                        lookat=self.trans[0:3, 3].T,
                                        up=[0,1,0])