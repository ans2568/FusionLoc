import time
import numpy as np
import cv2
import open3d as o3d

# default_path는 경로에 따라 변경 필요

def to2Dcloud(points):
        pcd_arr = np.asarray(points.points)
        pcd_arr[:,2] = 0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_arr)
        return pcd

class LiDART():
    def __init__(self, inputStruct, dataset):
        if dataset == 'gazebo':
            self.default_path = 'data/gazebo_dataset/lidar/'
        elif dataset == 'NIA':
            self.default_path = 'data/NIA/lidar/'
        elif dataset == 'iiclab':
            self.default_path = 'data/iiclab_real/lidar/'

        # inputStruct는 2차원 배열
        # [time, gt_x, gt_y, gt_theta, image_path, lidar_path]
        self.lidar_Query = self.default_path + inputStruct[0][0] + '.pcd'
        self.gt_x = float(inputStruct[0][1]) # 입력 라이다의 groundtruth x
        self.gt_y = float(inputStruct[0][2]) # 입력 라이다의 groundtruth y
        self.gt_theta = float(inputStruct[0][3]) # 입력 라이다의 groundtruth theta
        self.cos = np.cos(self.gt_theta)
        self.sin = np.sin(self.gt_theta)
        self.gt_rotation = np.array([[self.cos, -self.sin, 0],
                                  [self.sin, self.cos, 0],
                                  [0, 0, 1]])
        self.gt_translation = np.array([self.gt_x, self.gt_y, 0])
        self.gt_transformation = np.zeros((4,4))
        self.gt_transformation[:3, :3] = self.gt_rotation
        self.gt_transformation[3, :4] = np.append(self.gt_translation, 1)

    def icp(self, outputStruct):
        # outputStruct는 리스트
        # [time, gt_x, gt_y, gt_theta, image_path, lidar_path]
        start = time.time()
        self.lidar_DB = self.default_path + outputStruct[0] + '.pcd'
        self.gt_x_db = float(outputStruct[1]) # 후보 라이다의 groundtruth x
        self.gt_y_db = float(outputStruct[2]) # 후보 라이다의 groundtruth y
        self.gt_theta_db = float(outputStruct[3]) # 후보 라이다의 groundtruth theta
        cos = np.cos(self.gt_theta_db)
        sin = np.sin(self.gt_theta_db)
        gt_rotation = np.array([[cos, -sin, 0], # 후보 라이다의 groundtruth Rotation
                                [sin, cos, 0],
                                [0, 0, 1]])
        gt_translation_db = np.array([self.gt_x_db, self.gt_y_db, 0]) # 후보 라이다의 groundtruth translation
        gt_transformation_db = np.zeros((4,4))
        gt_transformation_db[:3, :3] = gt_rotation
        gt_transformation_db[:4, 3] = np.append(gt_translation_db, 1) # 후보 라이다의 groundtruth Transformation
        
        query_cloud = o3d.io.read_point_cloud(self.lidar_Query)
        db_cloud = o3d.io.read_point_cloud(self.lidar_DB)
        if not query_cloud.has_points() or not db_cloud.has_points():
             print("Could not find pcd file")
             return
        # query_cloud = pcl.load(self.lidar_Query)
        # db_cloud = pcl.load(self.lidar_DB)
        # if query_cloud is None or db_cloud is None:
        #      print("Could not find pcd file")
        #      return
        cloud_in = to2Dcloud(query_cloud)
        cloud_out = to2Dcloud(db_cloud)
        ret = o3d.pipelines.registration.registration_icp(cloud_in, cloud_out, max_correspondence_distance=0.1, estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
        # icp = cloud_in.make_IterativeClosestPoint()
        # converged, T, _, fitness = icp.icp(cloud_in, cloud_out)
        
        # print("lidar T : ")
        # print(ret.transformation)
        est_T = np.dot(ret.transformation, gt_transformation_db)
        est_R = est_T[:3, :3]
        # est_t = est_T[3, :3]
        est_t = ret.transformation[:3, 3] + gt_transformation_db[:3, 3]
        # print('gt_transformation[translation]')
        # print(gt_transformation_db[:3, 3])
        # print('est_t : ', est_t)
        est_R_vec, _ = cv2.Rodrigues(est_R)
        est_theta = np.linalg.norm(est_R_vec)
        est_x = est_t[0]
        est_y = est_t[1]
        diff_x = abs(abs(self.gt_x) - abs(est_x))
        diff_y = abs(abs(self.gt_y) - abs(est_y))
        diff_theta = abs(abs(self.gt_theta) - abs(est_theta)) # 입력 라이다의 groundtruth theta - 추정 라이다의 estimated theta
        end = time.time()
        elapsed_time = (end-start)*1000 # milliseconds
        print('----------------------------------------------')
        print('Query LiDAR : ' + self.lidar_Query)
        print('DB LiDAR : ' + self.lidar_DB)
        print('Euclidian fitness score : ' + str(ret.fitness))
        print('Total time taken : ' + str(elapsed_time) + 'ms')
        return self.lidar_DB, est_x, est_y, est_theta, diff_x, diff_y, diff_theta