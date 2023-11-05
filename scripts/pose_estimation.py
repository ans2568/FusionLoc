import cv2
import time
import numpy as np
import open3d as o3d
from os.path import join
from pathlib import Path

root = Path(__file__).parent.parent

class RootSIFT:
	def __init__(self):
		# initialize the SIFT feature extractor
		self.sift = cv2.xfeatures2d.SIFT_create()

	def compute(self, image, eps=1e-7):
		# compute SIFT descriptors
		kps, descs = self.sift.detectAndCompute(image, None)

		# if there are no keypoints or descriptors, return an empty tuple
		if len(kps) == 0:
			return ([], None)

		# apply the Hellinger kernel by first L1-normalizing, taking the
		# square-root, and then L2-normalizing
		descs /= (np.linalg.norm(descs, axis=0, ord=2) + eps)
		descs /= (descs.sum(axis=0) + eps)
		descs = np.sqrt(descs)
		# return a tuple of the keypoints and descriptors
		return (kps, descs)

# default_path는 경로에 따라 변경 필요

def to2Dcloud(points):
        pcd_arr = np.asarray(points.points)
        pcd_arr[:,2] = 0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_arr)
        return pcd

class PoseEstimation:
    def __init__(self, inputStruct, outputStruct, dataset):
        self.image_path = join(root, 'data', dataset, 'camera')
        self.lidar_path = join(root, 'data', dataset, 'lidar')
        if dataset == 'KingsCollege':
              self.image_path = join(root, 'data', dataset)
        elif dataset == '7_scenes':
              self.image_path = join(root, 'data', dataset)
              inputStruct[0][0] += '.color'
              outputStruct[0] += '.color'
        # inputStruct는 2차원 배열
        # [time, gt_x, gt_y, gt_theta, image_path, lidar_path]
        self.image_Query = join(self.image_path, inputStruct[0][0] + '.png')
        self.lidar_Query = join(self.lidar_path, inputStruct[0][0] + '.pcd')
        self.gt_x = float(inputStruct[0][1]) # 입력 이미지의 groundtruth x
        self.gt_y = float(inputStruct[0][2]) # 입력 이미지의 groundtruth y
        self.gt_theta = float(inputStruct[0][3]) # 입력 이미지의 groundtruth theta
        self.camera_matrix = np.array([[1073.3427734375, 0.0, 980.5746459960938],
                                [0.0, 1073.3427734375, 554.7113647460938],
                                [0.0, 0.0, 1.0]])
        # outputStruct는 리스트
        # [time, gt_x, gt_y, gt_theta, image_path, lidar_path]
        self.image_DB = join(self.image_path, outputStruct[0] + '.png')
        self.lidar_DB = join(self.lidar_path, outputStruct[0] + '.pcd')
        self.gt_x_db = float(outputStruct[1]) # 후보 이미지의 groundtruth x
        self.gt_y_db = float(outputStruct[2]) # 후보 이미지의 groundtruth y
        self.gt_theta_db = float(outputStruct[3]) # 후보 이미지의 groundtruth theta
        self.cos_db = np.cos(self.gt_theta_db)
        self.sin_db = np.sin(self.gt_theta_db)
        self.gt_rotation_db = np.array([[self.cos_db, -self.sin_db, 0], # 후보 이미지의 groundtruth Rotation
                                [self.sin_db, self.cos_db, 0],
                                [0, 0, 1]])
        self.gt_translation_db = np.array([self.gt_x_db, self.gt_y_db, 0]) # 후보 이미지의 groundtruth translation
        self.gt_transformation_db = np.zeros((4,4))
        self.gt_transformation_db[:3, :3] = self.gt_rotation_db
        self.gt_transformation_db[:4, 3] = np.append(self.gt_translation_db, 1) # 후보 이미지의 groundtruth Transformation

    def fivePointRANSAC(self):
        start = time.time()
        img_Query = cv2.imread(self.image_Query)
        img_DB = cv2.imread(self.image_DB)
        if img_Query is None or img_DB is None:
            print('Could not find Image')
            return
        img_query = cv2.cvtColor(img_Query, cv2.COLOR_BGR2GRAY)
        img_db = cv2.cvtColor(img_DB, cv2.COLOR_BGR2GRAY)

        rs = RootSIFT()
        keypoints1, descriptors1 = rs.compute(image=img_query)
        keypoints2, descriptors2 = rs.compute(image=img_db)

        descriptors1= descriptors1.astype(np.float32)
        descriptors2= descriptors2.astype(np.float32)

        # BF Matcher 초기화
        bf = cv2.BFMatcher()

        # # 특징점 매칭
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.85 * n.distance:
                good_matches.append(m)

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        E, _ = cv2.findEssentialMat(src_pts, dst_pts, self.camera_matrix, cv2.RANSAC)

        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, cameraMatrix=self.camera_matrix)

        T = np.hstack((R, t))
        T = np.vstack((T, [0, 0, 0, 1]))
        est_T = np.matmul(self.gt_transformation_db, T)
        est_R = est_T[:3, :3]
        est_t = est_T[:3, 3]
        rot_vec = cv2.Rodrigues(est_R)[0]
        est_theta = rot_vec[2][0]
        # est_theta = np.arctan2(est_R[1, 0], est_R[0, 0])
        est_x = est_t[0]
        est_y = est_t[1]
        diff_x = abs(abs(est_x) - abs(self.gt_x))
        diff_y = abs(abs(est_y) - abs(self.gt_y))
        diff_theta = abs(abs(est_theta) - abs(self.gt_theta)) # 추정 이미지의 estimated theta - 입력 이미지의 groundtruth theta
        end = time.time()
        elapsed_time = (end-start)*1000 # milliseconds
        print('----------------------------------------------')
        print('Query Image : ' + self.image_Query)
        print('DB Image : ' + self.image_DB)
        print('Total time taken : ' + str(elapsed_time) + 'ms')
        return self.image_DB, est_x, est_y, est_theta, diff_x, diff_y, diff_theta

    def icp(self):
        start = time.time()
        query_cloud = o3d.io.read_point_cloud(self.lidar_Query)
        db_cloud = o3d.io.read_point_cloud(self.lidar_DB)
        if not query_cloud.has_points() or not db_cloud.has_points():
             print("Could not find pcd file")
             return
        cloud_in = to2Dcloud(query_cloud)
        cloud_out = to2Dcloud(db_cloud)
        ret = o3d.pipelines.registration.registration_icp(cloud_in, cloud_out, max_correspondence_distance=0.1, estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
        est_T = np.dot(self.gt_transformation_db, ret.transformation)
        est_R = est_T[:3, :3]
        est_t = est_T[:3, 3]
        est_theta = np.arctan2(est_R[1, 0], est_R[0, 0])
        est_x = est_t[0]
        est_y = est_t[1]
        diff_x = abs(abs(est_x) - abs(self.gt_x))
        diff_y = abs(abs(est_y) - abs(self.gt_y))
        diff_theta = abs(abs(est_theta) - abs(self.gt_theta)) # 추정 라이다의 estimated theta - 입력 라이다의 groundtruth theta
        end = time.time()
        elapsed_time = (end-start)*1000 # milliseconds
        print('----------------------------------------------')
        print('Query LiDAR : ' + self.lidar_Query)
        print('DB LiDAR : ' + self.lidar_DB)
        print('Euclidian fitness score : ' + str(ret.fitness))
        print('Total time taken : ' + str(elapsed_time) + 'ms')
        return self.lidar_DB, est_x, est_y, est_theta, diff_x, diff_y, diff_theta