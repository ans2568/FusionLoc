import cv2
import time
import numpy as np

# default_path는 경로에 따라 변경 필요

class CameraT:
    def __init__(self, inputStruct, dataset):
        if dataset == 'gazebo':
            self.default_path = 'data/gazebo_dataset/camera/'
        elif dataset == 'NIA':
            self.default_path = 'data/NIA/camera/'
        elif dataset == 'iiclab':
            self.default_path = 'data/iiclab_real/camera/'
        # inputStruct는 2차원 배열
        # [time, gt_x, gt_y, gt_theta, image_path, lidar_path]
        self.image_Query = self.default_path + inputStruct[0][0] + '.png'
        self.gt_x = float(inputStruct[0][1]) # 입력 이미지의 groundtruth x
        self.gt_y = float(inputStruct[0][2]) # 입력 이미지의 groundtruth y
        self.gt_theta = float(inputStruct[0][3]) # 입력 이미지의 groundtruth theta
        self.cos = np.cos(self.gt_theta)
        self.sin = np.sin(self.gt_theta)
        self.gt_rotation = np.array([[self.cos, -self.sin, 0],
                                  [self.sin, self.cos, 0],
                                  [0, 0, 1]])
        self.gt_translation = np.array([self.gt_x, self.gt_y, 0])
        self.gt_transformation = np.zeros((4,4))
        self.gt_transformation[:3, :3] = self.gt_rotation
        self.gt_transformation[3, :4] = np.append(self.gt_translation, 1)
    
    def featureDetection(self, img, threshold=20, nonmaxSuppression=True):
        keypoints = []
        fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmaxSuppression)
        keypoints = fast.detect(img, None)
        points = cv2.KeyPoint_convert(keypoints)
        return points

    def featureTracking(self, img1, img2, points1, points2, status):
        # tracking 실패 시 자동적으로 points를 제거해주면서 tracking하는 함수
        winsize = (21, 21)
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01)
        # Lukas Kanade method를 사용한 Optical Flow method
        points2, status, _ = cv2.calcOpticalFlowPyrLK(img1, img2, points1, None, winSize=winsize, maxLevel=3, criteria=criteria)
        indexCorrection = 0
        for i in range(len(status)):
            pt = points2[i - indexCorrection]
            if (status[i]==0) or (pt[0]<0) or (pt[1]<0):
                if (pt[0]<0) or (pt[1]<0):
                    status[i] = 0
                np.delete(points1, i - indexCorrection)
                np.delete(points2, i - indexCorrection)
                indexCorrection += 1
        return points1, points2

    def opticalFlow(self, outputStruct):
        # outputStruct는 리스트
        # [time, gt_x, gt_y, gt_theta, image_path, lidar_path]
        self.image_DB = self.default_path + outputStruct[0] + '.png'
        self.gt_x_db = float(outputStruct[1]) # 후보 이미지의 groundtruth x
        self.gt_y_db = float(outputStruct[2]) # 후보 이미지의 groundtruth y
        self.gt_theta_db = float(outputStruct[3]) # 후보 이미지의 groundtruth theta
        cos = np.cos(self.gt_theta_db)
        sin = np.sin(self.gt_theta_db)
        gt_rotation = np.array([[cos, -sin, 0], # 후보 이미지의 groundtruth Rotation
                                [sin, cos, 0],
                                [0, 0, 1]])
        gt_translation_db = np.array([self.gt_x_db, self.gt_y_db, 0]) # 후보 이미지의 groundtruth translation
        gt_transformation_db = np.zeros((4,4))
        gt_transformation_db[:3, :3] = gt_rotation
        gt_transformation_db[:4, 3] = np.append(gt_translation_db, 1) # 후보 이미지의 groundtruth Transformation

        img_Query = cv2.imread(self.image_Query)
        img_DB = cv2.imread(self.image_DB)
        if img_Query is None or img_DB is None:
            print('Could not find Image')
            return
        img_query = cv2.cvtColor(img_Query, cv2.COLOR_BGR2GRAY)
        img_db = cv2.cvtColor(img_DB, cv2.COLOR_BGR2GRAY)
        points1 = self.featureDetection(img_query)
        points2 = []
        status = []
        points1, points2 = self.featureTracking(img_query, img_db, points1, points2, status)
        focal = 1073.3427
        cameraMat = np.array([[focal, 0, 980.5746], [0, focal, 554.7113], [0,0,1]])
        start = time.time()
        E, mask = cv2.findEssentialMat(points2, points1, cameraMatrix=cameraMat, method=cv2.RANSAC, threshold=1, prob=0.999)
        _, R, t, mask = cv2.recoverPose(E, points2, points1, cameraMatrix=cameraMat, mask=mask)
        end = time.time()
        R_mat = np.array(R)
        t_vec = np.array(t).reshape((3,1))
        T = np.hstack((R_mat, t_vec))
        T = np.vstack((T, [0, 0, 0, 1]))
        est_T = np.dot(T, gt_transformation_db)
        est_R = est_T[:3, :3]
        est_t = T[:3, 3] + gt_transformation_db[:3, 3]
        est_R_vec, _ = cv2.Rodrigues(est_R)
        est_theta = np.linalg.norm(est_R_vec)
        est_x = est_t[0]
        est_y = est_t[1]
        diff_x = abs(abs(self.gt_x) - abs(est_x))
        diff_y = abs(abs(self.gt_y) - abs(est_y))
        diff_theta = abs(abs(self.gt_theta) - abs(est_theta)) # 입력 이미지의 groundtruth theta - 추정 이미지의 estimated theta
        elapsed_time = (end-start)*1000 # milliseconds
        print('----------------------------------------------')
        print('Query Image : ' + self.image_Query)
        print('DB Image : ' + self.image_DB)
        print('Total time taken : ' + str(elapsed_time) + 'ms')
        return self.image_DB, est_x, est_y, est_theta, diff_x, diff_y, diff_theta