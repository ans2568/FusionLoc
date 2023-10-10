from __future__ import print_function
import time
import random
import numpy as np
from PIL import Image
from os.path import join, exists, isfile

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import h5py
import faiss

import cv2
import open3d as o3d

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from submodules.Struct import Struct
import submodules.netvlad as netvlad
from submodules.load import Dataset, loadFeature, input_transform

root = '/root/wip/FusionLoc'

class ImageDataset(data.Dataset):
    def __init__(self, image, input_transform=None):
        super().__init__()
        self.image = image
        self.input_transform = input_transform

    def __getitem__(self, index):
        img = Image.open(self.image)

        if self.input_transform:
            img = self.input_transform(img)
        return img, index

    def __len__(self):
        return 1


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

def scan2open3d(lidar):
    
    # LaserScan 메시지로부터 거리 및 각도 데이터 추출
    ranges = lidar.ranges
    angles = np.linspace(lidar.angle_min, lidar.angle_max, len(ranges))

    # 거리와 각도를 사용하여 PointCloud 생성
    points = []
    for i in range(len(ranges)):
        if not np.isinf(ranges[i]):
            x = ranges[i] * np.cos(angles[i])
            y = ranges[i] * np.sin(angles[i])
            z = 0.0  # LaserScan은 2D 스캔이므로 z 좌표는 0
            points.append([x, y, z])

    if points:
        # Open3D의 PointCloud 데이터로 변환
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
    # ranges = np.array(lidar.ranges)

    # # 거리값이 유효한 범위 내에 있는 샘플만 선택 (유효한 범위는 0.0 ~ lidar.range_max 사이)
    # valid_indices = np.where((ranges >= lidar.range_min) & (ranges <= lidar.range_max))[0]

    # # 유효한 샘플의 거리와 각도 추출
    # valid_ranges = ranges[valid_indices]
    # print(len(valid_ranges))
    # valid_angles = np.arange(lidar.angle_min, lidar.angle_max, lidar.angle_increment)[valid_indices]

    # # 거리와 각도를 극 좌표에서 직교 좌표로 변환
    # x = valid_ranges * np.cos(valid_angles)
    # y = valid_ranges * np.sin(valid_angles)
    # z = np.zeros_like(x)  # 2D 레이저 스캐너의 경우 z 좌표는 0으로 설정

    # # Open3D 포인트 클라우드로 변환
    # point_cloud = o3d.geometry.PointCloud()
    # points = np.column_stack((x, y, z))
    # point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def to2Dcloud(points):
        pcd_arr = np.asarray(points.points)
        pcd_arr[:,2] = 0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_arr)
        return pcd

class PoseEstimation:
    def __init__(self, outputStruct, dataset):
        dir = join(root, 'data', dataset)
        self.image_path = join(dir, 'camera')
        self.lidar_path = join(dir, 'lidar')
        self.camera_matrix = np.array([[1073.3427734375, 0.0, 980.5746459960938],
                                [0.0, 1073.3427734375, 554.7113647460938],
                                [0.0, 0.0, 1.0]])
        # outputStruct는 리스트
        # [time, gt_x, gt_y, gt_theta, image_path, lidar_path]
        self.image_DB = join(self.image_path, outputStruct[0][0] + '.png')
        self.lidar_DB = join(self.lidar_path, outputStruct[0][0] + '.pcd')
        self.gt_x_db = float(outputStruct[0][1]) # 후보 이미지의 groundtruth x
        self.gt_y_db = float(outputStruct[0][2]) # 후보 이미지의 groundtruth y
        self.gt_theta_db = float(outputStruct[0][3]) # 후보 이미지의 groundtruth theta
        self.cos_db = np.cos(self.gt_theta_db)
        self.sin_db = np.sin(self.gt_theta_db)
        self.gt_rotation_db = np.array([[self.cos_db, -self.sin_db, 0], # 후보 이미지의 groundtruth Rotation
                                [self.sin_db, self.cos_db, 0],
                                [0, 0, 1]])
        self.gt_translation_db = np.array([self.gt_x_db, self.gt_y_db, 0]) # 후보 이미지의 groundtruth translation
        self.gt_transformation_db = np.zeros((4,4))
        self.gt_transformation_db[:3, :3] = self.gt_rotation_db
        self.gt_transformation_db[:4, 3] = np.append(self.gt_translation_db, 1) # 후보 이미지의 groundtruth Transformation

    def fivePointRANSAC(self, image):
        start = time.time()
        image = Image.open(image)
        img_Query = np.array(image)
        img_DB = cv2.imread(self.image_DB)
        if img_Query is None or img_DB is None:
            print('Could not find Image')
            return
        img_query = cv2.cvtColor(img_Query, cv2.COLOR_RGB2GRAY)
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

        # Fundamental Matrix 추정
        _, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 5)

        src_pts = src_pts[mask.ravel() == 1]
        dst_pts = dst_pts[mask.ravel() == 1]

        E, _ = cv2.findEssentialMat(src_pts, dst_pts, self.camera_matrix, cv2.RANSAC)

        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, cameraMatrix=self.camera_matrix)

        T = np.hstack((R, t))
        T = np.vstack((T, [0, 0, 0, 1]))
        est_T = np.matmul(self.gt_transformation_db, T)
        est_R = est_T[:3, :3]
        est_t = est_T[:3, 3]
        est_theta = np.arctan2(est_R[1, 0], est_R[0, 0])
        est_x = est_t[0]
        est_y = est_t[1]
        end = time.time()
        elapsed_time = (end-start)*1000 # milliseconds
        print('----------------------------------------------')
        print('Total time taken : ' + str(elapsed_time) + 'ms')
        return est_x, est_y, est_theta

    def icp(self, lidar):
        start = time.time()
        query_cloud = scan2open3d(lidar)
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
        end = time.time()
        elapsed_time = (end-start)*1000 # milliseconds
        print('----------------------------------------------')
        print('Euclidian fitness score : ' + str(ret.fitness))
        print('Total time taken : ' + str(elapsed_time) + 'ms')
        return est_x, est_y, est_theta

def initial_pose_estimation(image = None, lidar = None, cuda = True, seed = 123, dataset = 'iiclab', resume = join(root, 'runsPath', 'Sep14_11-06-09_vgg16_netvlad')):
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with cuda = False")

    device = torch.device("cuda" if cuda else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    dataset_type = Dataset(dataset)
    print('===> Loading dataset(s)')
    whole_test_set = dataset_type.get_DB_test_set()
    print('===> Building model')

    encoder_dim = 512
    encoder = models.vgg16(pretrained=True)
    # capture only feature part and remove last relu and maxpool
    layers = list(encoder.features.children())[:-2]

    encoder = nn.Sequential(*layers)
    model = nn.Module() 
    model.add_module('encoder', encoder)

    net_vlad = netvlad.NetVLAD(num_clusters=64, dim=encoder_dim, vladv2=False)

    initcache = join(root, 'dataPath', 'centroids', "vgg16" + '_' + dataset + '_' + str(64) +'_desc_cen.hdf5')

    if not exists(initcache):
        raise FileNotFoundError('Could not find clusters, please run with cluster.py before proceeding')

    with h5py.File(initcache, mode='r') as h5: 
        clsts = h5.get("centroids")[...]
        traindescs = h5.get("descriptors")[...]
        net_vlad.init_params(clsts, traindescs) 
        del clsts, traindescs

    model.add_module('pool', net_vlad)

    resume_ckpt = join(resume, 'checkpoints', 'checkpoint.pth.tar')

    if isfile(resume_ckpt):
        print("=> loading checkpoint '{}'".format(resume_ckpt))
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_ckpt, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_ckpt))

    model.eval()

    data_set = ImageDataset(image, input_transform=input_transform())
    data_loader = DataLoader(dataset=data_set)

    dbFeat = np.array(loadFeature(join(root, 'feature', 'dbFeature_' + dataset + '.npy')), dtype=np.float32)
    with torch.no_grad():
        print('====> Extracting Features')
        pool_size = encoder_dim
        pool_size *= 64
        qFeat = np.empty((1, pool_size))
        for input, index in data_loader:
            if index.detach().numpy()[0] == 1:
                break
            input = input.to(device)
            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding)
            qFeat[index.detach().numpy(),:] = vlad_encoding.detach().cpu().numpy()
            del input, image_encoding, vlad_encoding
    del data_loader

    # extracted for both db and query, now split in own sets
    qFeat = qFeat[:].astype('float32')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    n_values = [30] # 해당 값이 이미지 출력 개수와 동일
    qFeature = qFeat[0][np.newaxis, :]
    _, predictions = faiss_index.search(qFeature, max(n_values))
    # for each query get those within threshold distance
    gt = whole_test_set.getPositives()
    correct_at_n = np.zeros(len(n_values))
    output_Struct = Struct(dataset)
    #TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        for i,n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            m = np.in1d(pred[:], gt[qIx])
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                for i, index in enumerate(pred[m]):
                    img, timestamp = whole_test_set.get(index)                        
                    output = transforms.ToPILImage()(img)
                    output.save("prediction/" + dataset + "/output_" + str(timestamp) + "_" + str(index) + ".png")
                    output_Struct.append(timestamp=timestamp)
                    print(timestamp)
                break

    pe = PoseEstimation(output_Struct.get(), dataset)
    est_x_c, est_y_c, est_theta_c= pe.fivePointRANSAC(image) # return image_db, est_x, est_y, est_theta, diff_x, diff_y, diff_theta
    est_x_l, est_y_l, est_theta_l= pe.icp(lidar) # return lidar_db, est_x, est_y, est_theta, diff_x, diff_y, diff_theta

    # 람다 = 0.9
    lambda_value = 0.9
    est_final_x = lambda_value*est_x_l + (1-lambda_value)*est_x_c # 최종 포즈 x
    est_final_y = lambda_value*est_y_l + (1-lambda_value)*est_y_c # 최종 포즈 y
    est_final_theta = lambda_value*est_theta_l + (1-lambda_value)*est_theta_c # 최종 포즈 theta

    return est_final_x, est_final_y, est_final_theta