from __future__ import print_function
import argparse
import random
from os import makedirs
from os.path import join, exists, isfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from PIL import Image
import torchvision.models as models
import h5py
import faiss

from Struct import Struct
from util.load import Dataset, loadFeature
from pose_estimation import PoseEstimation
import netvlad as netvlad

import csv
import time
import numpy as np
from pathlib import Path

root = Path(__file__).parent.parent

parser = argparse.ArgumentParser(description='Initial Pose Estimation using NetVLAD')
parser.add_argument('--cacheBatchSize', type=int, default=1, help='Batch size for caching and testing')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=1, help='Number of threads for each data loader to use')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--dataPath', type=str, default=join(root, 'dataPath'), help='Path for centroid data.')
parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='latest', 
        help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--arch', type=str, default='vgg16', 
        help='basenetwork to use', choices=['vgg16', 'alexnet'])
parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--fromscratch', action='store_true', help='Train from scratch rather than using pretrained models')
parser.add_argument('--dataset', type=str, default='NIA', help='select Dataset type [gazebo, NIA, iiclab]')
parser.add_argument('--mode', type=str, default='both', help='select mode [camera, both]')

def test(eval_set):
    # TODO what if features dont fit in memory? 
    test_data_loader = DataLoader(dataset=eval_set, num_workers=opt.threads, batch_size=opt.cacheBatchSize, 
                                  shuffle=False, pin_memory=cuda)

    model.eval()
    loading_time = time.time()
    dbFeat = np.array(loadFeature(join(root, 'feature', 'dbFeature_' + opt.dataset + '.npy')), dtype=np.float32)
    with torch.no_grad():
        print('====> Extracting Features')
        pool_size = encoder_dim
        pool_size *= opt.num_clusters
        qFeat = np.empty((len(eval_set.dbStruct.qImage), pool_size))
        for input, indices in test_data_loader:
            if indices.detach().numpy()[0] == len(eval_set.dbStruct.qImage):
                break
            input = input.to(device)
            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding)
            qFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()

            del input, image_encoding, vlad_encoding
    del test_data_loader

    end_loading_time = time.time()
    loading_elapsed_time = (end_loading_time-loading_time)*1000 # milliseconds
    print('Query Image Feature Extraction time taken : ' + str(loading_elapsed_time) + 'ms')

    # extracted for both db and query, now split in own sets
    qFeat = qFeat[:].astype('float32')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    for idx, image in enumerate(eval_set.dbStruct.qImage):
        input_Struct = None
        output_Struct = None
        query_timestamp = None
        input_Struct = Struct(eval_set.dataset)
        output_Struct = Struct(eval_set.dataset)
        path = join(root, 'data', eval_set.dataset)
        if eval_set.dataset == 'gazebo':
            query_timestamp = image[-18:-4]
        elif eval_set.dataset == 'NIA':
            query_timestamp = image[-22:-4]
        elif eval_set.dataset == 'iiclab':
            query_timestamp = image[-18:-4]
        elif eval_set.dataset == 'KingsCollege':
            query_timestamp = image[:-4]
        else:
            query_timestamp = image[:-10]
        input_Struct.append(query_timestamp)
        # input_img = join(path, image)
        # img = Image.open(input_img)
        # tf = transforms.Compose([transforms.ToTensor()])
        # img = tf(img)
        # input_img = transforms.ToPILImage()(img)
        # if not exists(join(root, 'prediction', eval_set.dataset)):
        #     makedirs(join(root, 'prediction', eval_set.dataset))
        # input_img.save(join(root, 'prediction', eval_set.dataset, 'input_' + str(query_timestamp) + '.png'))

        n_values = [1] # 해당 값이 이미지 출력 개수와 동일
        qFeature = qFeat[idx][np.newaxis, :]
        _, predictions = faiss_index.search(qFeature, max(n_values))
        # for each query get those within threshold distance
        gt = eval_set.getPositives_test(idx)
        correct_at_n = np.zeros(len(n_values))
        #TODO can we do this on the matrix in one go?
        count = 0
        for qIx, pred in enumerate(predictions):
            for i,n in enumerate(n_values):
                # if in top N then also in top NN, where NN > N
                m = np.in1d(pred[:], gt[qIx])
                if np.any(np.in1d(pred[:n], gt[qIx])):
                    correct_at_n[i:] += 1
                    for i, index in enumerate(pred[m]):
                        if count < 1:
                            img, timestamp = eval_set.get(index)
                        # output = transforms.ToPILImage()(img)
                        # output.save("../prediction/" + eval_set.dataset + "/output_" + str(timestamp) + "_" + str(index) + ".png")
                            output_Struct.append(timestamp=timestamp)
                        count += 1
                    break
        final_data = []
        for output in output_Struct.get():
            pe = PoseEstimation(input_Struct.get(), output, eval_set.dataset)
            t_stamp = output[0] # 추정된 pose 데이터의 timestamp
            gt_timestamp = input_Struct.get()[0][0] # 입력 데이터의 timestamp
            gt_x = input_Struct.get()[0][1] # 입력 데이터의 groundtruth x
            gt_y = input_Struct.get()[0][2] # 입력 데이터의 groundtruth y
            gt_theta = input_Struct.get()[0][3] # 입력 데이터의 groundtruth theta
            if opt.mode == 'camera':
                image_db, est_x_c, est_y_c, est_theta_c, diff_x_c, diff_y_c, diff_theta_c = pe.fivePointRANSAC() # return image_db, est_x, est_y, est_theta, diff_x, diff_y, diff_theta
                data = np.array([t_stamp, image_db, est_x_c, est_y_c, est_theta_c, diff_x_c, diff_y_c, diff_theta_c], dtype=object)
                final_data.append(data)
                float_cols = [5,6,7]
                dist_data = np.array(final_data)[:, float_cols].astype(float)
                euclidean_dist_c = np.sqrt(dist_data[:,0]**2 + dist_data[:,1]**2 + dist_data[:,2]**2) # 카메라에서의 유클리디안 거리
                min_index_c = np.argmin(euclidean_dist_c) # 카메라의 유클리디안 거리 중 실제 값과 차이가 가장 작은 인덱스
                est_x_c = float(final_data[min_index_c][2]) # 카메라의 유클리디안 거리 중 실제 값과 차이가 가장 작은 인덱스의 x 값
                est_y_c = float(final_data[min_index_c][3]) # 카메라의 유클리디안 거리 중 실제 값과 차이가 가장 작은 인덱스의 y 값
                est_theta_c = float(final_data[min_index_c][4]) # 카메라의 유클리디안 거리 중 실제 값과 차이가 가장 작은 인덱스의 theta 값
                est_timestamp = str(final_data[min_index_c][0]) # 카메라의 유클리디안 거리 중 실제 값과 차이가 가장 작은 인덱스의 timestamp

                est_final_x = est_x_c # 최종 포즈 x
                est_final_y = est_y_c # 최종 포즈 y
                est_final_theta = est_theta_c # 최종 포즈 theta
                lambda_value = 0

            elif opt.mode == 'both':
                image_db, est_x_c, est_y_c, est_theta_c, diff_x_c, diff_y_c, diff_theta_c = pe.fivePointRANSAC() # return image_db, est_x, est_y, est_theta, diff_x, diff_y, diff_theta
                lidar_db, est_x_l, est_y_l, est_theta_l, diff_x_l, diff_y_l, diff_theta_l = pe.icp() # return lidar_db, est_x, est_y, est_theta, diff_x, diff_y, diff_theta
                data = np.array([t_stamp, image_db, est_x_c, est_y_c, est_theta_c, diff_x_c, diff_y_c, diff_theta_c, lidar_db, est_x_l, est_y_l, est_theta_l, diff_x_l, diff_y_l, diff_theta_l], dtype=object)
                final_data.append(data)
                float_cols = [5,6,7,12,13,14]
                dist_data = np.array(final_data)[:, float_cols].astype(float)
                euclidean_dist_c = np.sqrt(dist_data[:,0]**2 + dist_data[:,1]**2 + dist_data[:,2]**2) # 카메라에서의 유클리디안 거리
                euclidean_dist_l = np.sqrt(dist_data[:,3]**2 + dist_data[:,4]**2 + dist_data[:,5]**2) # 라이다에서의 유클리디안 거리
                min_index_c = np.argmin(euclidean_dist_c) # 카메라의 유클리디안 거리 중 실제 값과 차이가 가장 작은 인덱스
                min_index_l = np.argmin(euclidean_dist_l) # 라이다의 유클리디안 거리 중 실제 값과 차이가 가장 작은 인덱스
                est_x_c = float(final_data[min_index_c][2]) # 카메라의 유클리디안 거리 중 실제 값과 차이가 가장 작은 인덱스의 x 값
                est_y_c = float(final_data[min_index_c][3]) # 카메라의 유클리디안 거리 중 실제 값과 차이가 가장 작은 인덱스의 y 값
                est_theta_c = float(final_data[min_index_c][4]) # 카메라의 유클리디안 거리 중 실제 값과 차이가 가장 작은 인덱스의 theta 값
                est_x_l = float(final_data[min_index_l][9]) # 라이다의 유클리디안 거리 중 실제 값과 차이가 가장 작은 인덱스의 x 값
                est_y_l = float(final_data[min_index_l][10]) # 라이다의 유클리디안 거리 중 실제 값과 차이가 가장 작은 인덱스의 y 값
                est_theta_l = float(final_data[min_index_l][11]) # 라이다의 유클리디안 거리 중 실제 값과 차이가 가장 작은 인덱스의 theta 값
                est_timestamp = str(final_data[min_index_l][0]) # 라이다의 유클리디안 거리 중 실제 값과 차이가 가장 작은 인덱스의 timestamp

                # 람다 = 0.9
                lambda_value = 0.9
                est_final_x = lambda_value*est_x_l + (1-lambda_value)*est_x_c # 최종 포즈 x
                est_final_y = lambda_value*est_y_l + (1-lambda_value)*est_y_c # 최종 포즈 y
                est_final_theta = lambda_value*est_theta_l + (1-lambda_value)*est_theta_c # 최종 포즈 theta

            # input_groundtruth.csv : 입력 데이터에 대한 timestamp와 실제 lidar pose와 camera pose에 대한 정보를 행으로 갖고 있는 csv 파일
            result_filename = join(root, 'result', 'result_' + opt.dataset + "_" + str(lambda_value) + '.csv')
            header_input = ['input_timestamp', 'output_timestamp', 'input_X', 'output_X', 'input_Y',
                            'output_Y', 'input_theta', 'output_theta', 'difference_X(m)', 'difference_Y(m)',
                            'difference_theta(radian)', 'error_Position_L2 norm(m)', 'error_Orientation(degree)']
            error_position_L2_norm = np.sqrt((est_final_x-gt_x)**2 + (est_final_y-gt_y)**2)
            result = np.array([gt_timestamp, est_timestamp, gt_x, est_final_x, gt_y, est_final_y, gt_theta,
                            est_final_theta, est_final_x-gt_x, est_final_y-gt_y, est_final_theta-gt_theta,
                            error_position_L2_norm, np.degrees(est_final_theta-gt_theta)])

            # CSV 파일 열기 (존재하지 않을 경우 헤더와 함께 생성됨)
            if not exists(result_filename):
                with open(result_filename, mode='w', newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(header_input)
                    csv_writer.writerow(result)
            else:
                with open(result_filename, mode='a', newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(result)
            print("Check the experiment results in the " + result_filename)

if __name__ == "__main__":
    start_time = time.time()
    opt = parser.parse_args()
    print(opt)

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    dataset = Dataset(opt.dataset)
    print('===> Loading dataset(s)')
    whole_test_set = dataset.get_Query_test_set()
    print('===> Evaluating on test set')
    print('====> Query count:', whole_test_set.dbStruct.numQ)
    print('===> Building model')

    # default opt.fromscratch = False
    pretrained = not opt.fromscratch
    if opt.arch.lower() == 'alexnet':
        encoder_dim = 256
        encoder = models.alexnet(pretrained=pretrained)
        # capture only features and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False

    elif opt.arch.lower() == 'vgg16':
        encoder_dim = 512
        encoder = models.vgg16(pretrained=pretrained)
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]: 
                for p in l.parameters():
                    p.requires_grad = False

    encoder = nn.Sequential(*layers)
    model = nn.Module() 
    model.add_module('encoder', encoder)

    net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=opt.vladv2)
    if not opt.resume: 
        initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + whole_test_set.dataset + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')

        if not exists(initcache):
            raise FileNotFoundError('Could not find clusters, please run with cluster.py before proceeding')

        with h5py.File(initcache, mode='r') as h5: 
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            net_vlad.init_params(clsts, traindescs) 
            del clsts, traindescs

    model.add_module('pool', net_vlad)

    isParallel = False
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        model.pool = nn.DataParallel(model.pool)
        isParallel = True

    if not opt.resume:
        model = model.to(device)

    if opt.resume:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.resume, 'checkpoints', 'model_best.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_score']
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))

    test(whole_test_set)

    end_time = time.time()
    elapsed_time = (end_time-start_time)*1000 # milliseconds
    print('Total time taken : ' + str(elapsed_time) + 'ms')