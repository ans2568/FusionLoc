from __future__ import print_function
import argparse
from math import ceil
import random, shutil, json
import matplotlib.pyplot as plt
from os import makedirs, remove
from os.path import join, exists, isfile

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms

from PIL import Image
from datetime import datetime
import torchvision.models as models
import h5py
import faiss

import Struct as s
import custom as dataset
from pose_estimation import PoseEstimation

from tensorboardX import SummaryWriter
import numpy as np
import netvlad

parser = argparse.ArgumentParser(description='Initial Pose Estimation using NetVLAD')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test', 'cluster'])
parser.add_argument('--batchSize', type=int, default=4, 
        help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
parser.add_argument('--cacheBatchSize', type=int, default=24, help='Batch size for caching and testing')
parser.add_argument('--cacheRefreshRate', type=int, default=1000, 
        help='How often to refresh cache, in number of queries. 0 for off')
parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
        help='manual epoch number (useful on restarts)')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use', choices=['SGD', 'ADAM'])
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--dataPath', type=str, default='dataPath/', help='Path for centroid data.')
parser.add_argument('--runsPath', type=str, default='runsPath/', help='Path to save runs to.')
parser.add_argument('--savePath', type=str, default='checkpoints', 
        help='Path to save checkpoints to in logdir. Default=checkpoints/')
parser.add_argument('--cachePath', type=str, default='cache/', help='Path to save cache to.')
parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='latest', 
        help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--evalEvery', type=int, default=1, 
        help='Do a validation set run, and save, every N epochs.')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping. 0 is off.')
parser.add_argument('--arch', type=str, default='vgg16', 
        help='basenetwork to use', choices=['vgg16', 'alexnet'])
parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use',
        choices=['netvlad', 'max', 'avg'])
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
parser.add_argument('--split', type=str, default='val', help='Data split to use for testing. Default is val', 
        choices=['test', 'test250k', 'train', 'val'])
parser.add_argument('--fromscratch', action='store_true', help='Train from scratch rather than using pretrained models')
parser.add_argument('--dataset', type=str, default='NIA', help='select Dataset type [gazebo, NIA, iiclab]')

def train(epoch):
    epoch_loss = 0
    startIter = 1 # keep track of batch iter across subsets for logging

    if opt.cacheRefreshRate > 0:
        subsetN = ceil(len(train_set) / opt.cacheRefreshRate)
        #TODO randomise the arange before splitting?
        subsetIdx = np.array_split(np.arange(len(train_set)), subsetN)
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(train_set))]
    nBatches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    for subIter in range(subsetN):
        print('====> Building Cache')
        model.eval()
        train_set.cache = join(opt.cachePath, train_set.whichSet + '_feat_cache.hdf5')
        with h5py.File(train_set.cache, mode='w') as h5: 
            pool_size = encoder_dim
            if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
            h5feat = h5.create_dataset("features", 
                    [len(whole_train_set), pool_size], 
                    dtype=np.float32)
            with torch.no_grad():
                for iteration, (input, indices) in enumerate(whole_training_data_loader, 1):
                    input = input.to(device)
                    image_encoding = model.encoder(input)
                    vlad_encoding = model.pool(image_encoding) 
                    h5feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                    del input, image_encoding, vlad_encoding
        sub_train_set = Subset(dataset=train_set, indices=subsetIdx[subIter])

        training_data_loader = DataLoader(dataset=sub_train_set, num_workers=opt.threads, 
                    batch_size=opt.batchSize, shuffle=True, 
                    collate_fn=dataset.collate_fn, pin_memory=cuda)

        print('Allocated:', torch.cuda.memory_allocated())
        print('Cached:', torch.cuda.memory_cached())

        model.train()
        for iteration, (query, positives, negatives, 
                negCounts, indices) in enumerate(training_data_loader, startIter):
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
            if query is None: continue # in case we get an empty batch

            B, C, H, W = query.shape
            nNeg = torch.sum(negCounts)
            input = torch.cat([query, positives, negatives])

            input = input.to(device)
            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding) 

            vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, nNeg])

            optimizer.zero_grad()
            
            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to 
            # do it per query, per negative
            loss = 0
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss += criterion(vladQ[i:i+1], vladP[i:i+1], vladN[negIx:negIx+1])

            loss /= nNeg.float().to(device) # normalise by actual number of negatives
            loss.backward()
            optimizer.step()
            del input, image_encoding, vlad_encoding, vladQ, vladP, vladN
            del query, positives, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 50 == 0 or nBatches <= 10:
                print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, 
                    nBatches, batch_loss), flush=True)
                writer.add_scalar('Train/Loss', batch_loss, 
                        ((epoch-1) * nBatches) + iteration)
                writer.add_scalar('Train/nNeg', nNeg, 
                        ((epoch-1) * nBatches) + iteration)
                print('Allocated:', torch.cuda.memory_allocated())
                print('Cached:', torch.cuda.memory_cached())

        startIter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        remove(train_set.cache) # delete HDF5 cache

    avg_loss = epoch_loss / nBatches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), 
            flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)

def test(eval_set, epoch=0, write_tboard=False):
    input_Struct = s.Struct(eval_set.dataset)
    output_Struct = s.Struct(eval_set.dataset)
    # TODO what if features dont fit in memory? 
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)

    model.eval()
    with torch.no_grad():
        print('====> Extracting Features')
        pool_size = encoder_dim
        if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
        dbFeat = np.empty((len(eval_set), pool_size))
        for iteration, (input, indices) in enumerate(test_data_loader, 1): # enumerate(foo, 1)은 1부터 시작
            input = input.to(device)
            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding)
            dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, 
                    len(test_data_loader)), flush=True)

            del input, image_encoding, vlad_encoding
    del test_data_loader

    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[eval_set.dbStruct.numDb:].astype('float32')
    dbFeat = dbFeat[:eval_set.dbStruct.numDb].astype('float32')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)
    if eval_set.dataset == 'gazebo':
        # @ TODO Gazebo 사용으로 인한 경로 변경
        input_Struct.append(eval_set.dbStruct.qImage[0][-18:-4])
    elif eval_set.dataset == 'NIA':
        input_Struct.append(eval_set.dbStruct.qImage[0][-22:-4])
    elif eval_set.dataset == 'iiclab':
        input_Struct.append(eval_set.dbStruct.qImage[0][-18:-4])
    if opt.mode.lower() != 'train':
        if eval_set.dataset == 'gazebo':
            # @ TODO Gazebo 사용으로 인한 경로 변경
            path = 'data/gazebo_dataset'
        elif eval_set.dataset == 'NIA':
            path = 'data/NIA'
        elif eval_set.dataset == 'iiclab':
            path = 'data/iiclab_real/'
        input_img = path + eval_set.dbStruct.qImage[0]
        img = Image.open(input_img)
        tf = transforms.Compose([transforms.ToTensor()])
        img = tf(img)
        input_img = transforms.ToPILImage()(img)
        if not exists('prediction/' + eval_set.dataset):
            makedirs('prediction/' + eval_set.dataset)
        if eval_set.dataset == 'gazebo':
            # @ TODO Gazebo 사용으로 인한 경로 변경
            input_img.save("prediction/" + eval_set.dataset + "/input_" + str(eval_set.dbStruct.qImage[0][-18:-4]) + ".png")
        elif eval_set.dataset == 'NIA':
            input_img.save("prediction/" + eval_set.dataset + "/input_" + str(eval_set.dbStruct.qImage[0][-22:-4]) + ".png")
        elif eval_set.dataset == 'iiclab':
            input_img.save("prediction/" + eval_set.dataset + "/input_" + str(eval_set.dbStruct.qImage[0][-18:-4]) + ".png")

    pe = PoseEstimation(input_Struct.get(), eval_set.dataset)

    # print('====> Calculating recall @ N')
    n_values = [1] # 해당 값이 이미지 출력 개수와 동일

    _, predictions = faiss_index.search(qFeat, max(n_values)) 
    # for each query get those within threshold distance
    gt = eval_set.getPositives() 
    correct_at_n = np.zeros(len(n_values))
    #TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        for i,n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            m = np.in1d(pred[:], gt[qIx])
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                for i, index in enumerate(pred[m]):
                    img, timestamp = eval_set.get(index)
                    if opt.mode.lower() != 'train':
                        output = transforms.ToPILImage()(img)
                        output.save("prediction/" + eval_set.dataset + "/output_" + str(timestamp) + "_" + str(index) + ".png")
                    output_Struct.append(timestamp=timestamp)
                break
    # # output_Struct에 있는 내용마다 OpticalFlow 진행(현재 20개)
    # for idx, output in enumerate(output_Struct.get()):
    #     # opticalFlow에 대한 반환값 받기
    #     camera_T.opticalFlow(output) # return est_x, est_y, est_theta, diff_x, diff_y, diff_theta

    if opt.mode.lower() != 'train':
        input_Struct = s.Struct(eval_set.dataset)
        # @ TODO Gazebo 사용으로 인한 경로 변경
        if eval_set.dataset == 'gazebo':
            input_Struct.append(eval_set.dbStruct.qImage[0][-18:-4])
        elif eval_set.dataset == 'NIA':
            input_Struct.append(eval_set.dbStruct.qImage[0][-22:-4])
        elif eval_set.dataset == 'iiclab':
            input_Struct.append(eval_set.dbStruct.qImage[0][-18:-4])
        camera_T = OF.CameraT(input_Struct.get(), eval_set.dataset)
        lidar_T = icp.LiDART(input_Struct.get(), eval_set.dataset)
        # output_Struct에 있는 내용마다 OpticalFlow 진행(현재 20개)
        final_data = []
        gt_timestamp = []
        gt_x = []
        gt_y = []
        gt_theta = []
        for idx, output in enumerate(output_Struct.get()):
            t_stamp = output[0] # 추정된 pose 데이터의 timestamp
            gt_timestamp = input_Struct.get()[0][0] # 입력 데이터의 timestamp
            gt_x = input_Struct.get()[0][1] # 입력 데이터의 groundtruth x
            gt_y = input_Struct.get()[0][2] # 입력 데이터의 groundtruth y
            gt_theta = input_Struct.get()[0][3] # 입력 데이터의 groundtruth theta
            # opticalFlow, icp에 대한 반환값 받기
            image_db, est_x_c, est_y_c, est_theta_c, diff_x_c, diff_y_c, diff_theta_c = pe.fivePointRANSAC(output) # return image_db, est_x, est_y, est_theta, diff_x, diff_y, diff_theta
            # image_db, est_x_c, est_y_c, est_theta_c, diff_x_c, diff_y_c, diff_theta_c = camera_T.opticalFlow(output) # return image_db, est_x, est_y, est_theta, diff_x, diff_y, diff_theta
            lidar_db, est_x_l, est_y_l, est_theta_l, diff_x_l, diff_y_l, diff_theta_l = pe.icp(output) # return lidar_db, est_x, est_y, est_theta, diff_x, diff_y, diff_theta
            data = np.array([t_stamp, image_db, est_x_c, est_y_c, est_theta_c, diff_x_c, diff_y_c, diff_theta_c, lidar_db, est_x_l, est_y_l, est_theta_l, diff_x_l, diff_y_l, diff_theta_l])
            final_data.append(data)
        float_cols = [5,6,7,12,13,14]
        dist_data = np.array(final_data)[:, float_cols].astype(float)
        np_final_data = np.array(final_data)
        es_timestamp = np_final_data[:, 0]
        es_x = 0.9*np_final_data[:,9].astype(float) + 0.1*np_final_data[:,2].astype(float)
        es_y = 0.9*np_final_data[:,10].astype(float) + 0.1*np_final_data[:,3].astype(float)
        es_theta = 0.9*np_final_data[:,11].astype(float) + 0.1*np_final_data[:,4].astype(float)
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
        est_final_x = 0.9*est_x_l + 0.1*est_x_c # 최종 포즈 x
        est_final_y = 0.9*est_y_l + 0.1*est_y_c # 최종 포즈 y
        est_final_theta = 0.9*est_theta_l + 0.1*est_theta_c # 최종 포즈 theta
        
        es_timestamp = np.delete(es_timestamp, np.where(es_timestamp == est_timestamp))
        es_x = np.delete(es_x, np.where(es_x == est_final_x))
        es_y = np.delete(es_y, np.where(es_y == est_final_y))
        es_theta = np.delete(es_theta, np.where(es_theta == est_final_theta))
        
        # input_groundtruth.csv : 입력 데이터에 대한 timestamp와 실제 lidar pose와 camera pose에 대한 정보를 행으로 갖고 있는 csv 파일
        input_gt_filename = 'input_groundtruth_test.csv'
        header_input = ['gt_timestamp', 'gt_x', 'gt_y', 'gt_theta']
        input_gt = []
        input_gt.append(np.array([gt_timestamp, gt_x, gt_y, gt_theta]))
        with open(input_gt_filename, mode='w', newline='') as input_groundtruth:
            csv_writer = csv.writer(input_groundtruth)
            csv_writer.writerow(header_input)
            for row in input_gt:
                csv_writer.writerow(row)
        print('input_groundtruth.csv complete! check input_groundtruth_test.csv')

        output_estimate_filename = 'output_estimate_test.csv'
        final_header = ['est_final_timestamp', 'est_final_x', 'est_final_y', 'est_final_theta']
        header = ['est_timestamp', 'est_x', 'est_y', 'est_theta']
        output_final_est = []
        output_final_est.append(np.array([est_timestamp, est_final_x, est_final_y, est_final_theta]))
        output_est = []
        for idx, data in enumerate(es_timestamp):
            output_est.append(np.array([data, es_x[idx], es_y[idx], es_theta[idx]]))
        with open(output_estimate_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(final_header)
            for row in output_final_est:
                csv_writer.writerow(row)
            csv_writer.writerow(header)
            for row in output_est:
                csv_writer.writerow(row)
        print('output_estimate_test.csv complete! check output_estimate_test.csv')

        # 3D 산점도
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(est_final_x, est_final_y, est_final_theta, marker='o', c='#ff7f0e', label='estimated final pose')
        ax.scatter(gt_x, gt_y, gt_theta, marker='^', c='#2ca02c', label = 'groundtruth pose')
        ax.scatter(es_x, es_y, es_theta, marker='o', c='#000000', label='estimated pose')
        ax.set_xlim3d(est_final_x - 5, est_final_x + 5)
        ax.set_ylim3d(est_final_y - 5, est_final_y + 5)
        ax.set_zlim3d(est_final_theta - 5, est_final_theta + 5)

        ax.set_xlabel('x position(cm)')
        ax.set_ylabel('y position(cm)')
        ax.set_zlabel('theta angle(radian)')
        ax.legend()
        plt.title('Pose Distribution')

        plt.show()

    # 필요없는 recall 계산
    recall_at_n = correct_at_n / eval_set.dbStruct.numQ
    recalls = {} #make dict for output
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard: writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch)
    return recalls

def get_clusters(cluster_set):
    nDescriptors = 2500
    nPerImage = 100
    nIm = ceil(nDescriptors/nPerImage)



    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
    data_loader = DataLoader(dataset=cluster_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda,
                sampler=sampler)

    if not exists(join(opt.dataPath, 'centroids')):
        makedirs(join(opt.dataPath, 'centroids'))

    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + cluster_set.dataset + '_' + str(opt.num_clusters) + '_desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5: 
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            dbFeat = h5.create_dataset("descriptors", 
                        [nDescriptors, encoder_dim], 
                        dtype=np.float32)

            for iteration, (input, indices) in enumerate(data_loader, 1):
                input = input.to(device)
                image_descriptors = model.encoder(input).view(input.size(0), encoder_dim, -1).permute(0, 2, 1)

                batchix = (iteration-1)*opt.cacheBatchSize*nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix*nPerImage
                    dbFeat[startix:startix+nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print("==> Batch ({}/{})".format(iteration, 
                        ceil(nIm/opt.cacheBatchSize)), flush=True)
                del input, image_descriptors
        
        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(encoder_dim, opt.num_clusters, niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    model_out_path = join(opt.savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(opt.savePath, 'model_best.pth.tar'))

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

if __name__ == "__main__":
    opt = parser.parse_args()

    restore_var = ['lr', 'lrStep', 'lrGamma', 'weightDecay', 'momentum', 
            'runsPath', 'savePath', 'arch', 'num_clusters', 'pooling', 'optim',
            'margin', 'seed', 'patience']
    if opt.resume:
        flag_file = join(opt.resume, 'checkpoints', 'flags.json')
        if exists(flag_file):
            with open(flag_file, 'r') as f:
                stored_flags = {'--'+k : str(v) for k,v in json.load(f).items() if k in restore_var}
                to_del = []
                for flag, val in stored_flags.items():
                    for act in parser._actions:
                        if act.dest == flag[2:]:
                            # store_true / store_false args don't accept arguments, filter these 
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ''
                for flag in to_del: del stored_flags[flag]

                train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]
                print('Restored flags:', train_flags)
                opt = parser.parse_args(train_flags, namespace=opt)

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

    print('===> Loading dataset(s)')
    if opt.mode.lower() == 'train':
        if opt.dataset == 'gazebo':
            whole_train_set = dataset.get_gazebo_whole_training_set()
            train_set = dataset.get_gazebo_training_query_set(opt.margin)
            whole_test_set = dataset.get_gazebo_whole_test_set()
        elif opt.dataset == 'NIA':
            whole_train_set = dataset.get_whole_training_set()
            train_set = dataset.get_training_query_set(opt.margin)
            whole_test_set = dataset.get_whole_val_set()
        elif opt.dataset == 'iiclab':
            whole_train_set = dataset.get_iiclab_whole_training_set()
            train_set = dataset.get_iiclab_training_query_set(opt.margin)
            whole_test_set = dataset.get_iiclab_whole_test_set()

        whole_training_data_loader = DataLoader(dataset=whole_train_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)
        print('====> Training query set:', len(train_set))
        print('===> Evaluating on val set, query count:', whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == 'test':
        if opt.split.lower() == 'test':
            if opt.dataset == 'gazebo':
                whole_test_set = dataset.get_gazebo_whole_test_set()
            elif opt.dataset == 'NIA':
                whole_test_set = dataset.get_whole_test_set()
            elif opt.dataset == 'iiclab':
                whole_test_set = dataset.get_iiclab_whole_test_set()
            print('===> Evaluating on test set')
        # elif opt.split.lower() == 'train':
        #     whole_test_set = dataset.get_whole_training_set()
        #     print('===> Evaluating on train set')
        # elif opt.split.lower() == 'val':
        #     whole_test_set = dataset.get_whole_val_set()
        #     print('===> Evaluating on val set')
        else:
            raise ValueError('Unknown dataset split: ' + opt.split)
        print('====> Query count:', whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == 'cluster':
        if opt.dataset == 'gazebo':
            whole_train_set = dataset.get_gazebo_whole_training_set(onlyDB=True)
        elif opt.dataset == 'NIA':
            whole_train_set = dataset.get_whole_training_set(onlyDB=True)
        elif opt.dataset == 'iiclab':
            whole_train_set = dataset.get_iiclab_whole_training_set(onlyDB=True)

    print('===> Building model')

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

    if opt.mode.lower() == 'cluster' and not opt.vladv2:
        layers.append(L2Norm())

    encoder = nn.Sequential(*layers)
    model = nn.Module() 
    model.add_module('encoder', encoder)

    if opt.mode.lower() != 'cluster':
        if opt.pooling.lower() == 'netvlad':
            net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=opt.vladv2)
            if not opt.resume: 
                if opt.mode.lower() == 'train':
                    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + train_set.dataset + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')
                else:
                    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + whole_test_set.dataset + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')

                if not exists(initcache):
                    raise FileNotFoundError('Could not find clusters, please run with --mode=cluster before proceeding')

                with h5py.File(initcache, mode='r') as h5: 
                    clsts = h5.get("centroids")[...]
                    traindescs = h5.get("descriptors")[...]
                    net_vlad.init_params(clsts, traindescs) 
                    del clsts, traindescs

            model.add_module('pool', net_vlad)
        elif opt.pooling.lower() == 'max':
            global_pool = nn.AdaptiveMaxPool2d((1,1))
            model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
        elif opt.pooling.lower() == 'avg':
            global_pool = nn.AdaptiveAvgPool2d((1,1))
            model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
        else:
            raise ValueError('Unknown pooling type: ' + opt.pooling)

    isParallel = False
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        if opt.mode.lower() != 'cluster':
            model.pool = nn.DataParallel(model.pool)
        isParallel = True

    if not opt.resume:
        model = model.to(device)
    
    if opt.mode.lower() == 'train':
        if opt.optim.upper() == 'ADAM':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr)#, betas=(0,0.9))
        elif opt.optim.upper() == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr,
                momentum=opt.momentum,
                weight_decay=opt.weightDecay)

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
        else:
            raise ValueError('Unknown optimizer: ' + opt.optim)

        # original paper/code doesn't sqrt() the distances, we do, so sqrt() the margin, I think :D
        criterion = nn.TripletMarginLoss(margin=opt.margin**0.5, 
                p=2, reduction='sum').to(device)

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
            if opt.mode == 'train':
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))

    if opt.mode.lower() == 'test':
        print('===> Running evaluation step')
        epoch = 1
        test(whole_test_set, epoch, write_tboard=False)
        # @TODO 여기 부분에 추가해야함
    elif opt.mode.lower() == 'cluster':
        print('===> Calculating descriptors and clusters')
        get_clusters(whole_train_set)
    elif opt.mode.lower() == 'train':
        print('===> Training model')
        writer = SummaryWriter(log_dir=join(opt.runsPath, datetime.now().strftime('%b%d_%H-%M-%S')+'_'+opt.arch+'_'+opt.pooling))

        # write checkpoints in logdir
        logdir = writer.file_writer.get_logdir()
        opt.savePath = join(logdir, opt.savePath)
        if not opt.resume:
            makedirs(opt.savePath)

        with open(join(opt.savePath, 'flags.json'), 'w') as f:
            f.write(json.dumps(
                {k:v for k,v in vars(opt).items()}
                ))
        print('===> Saving state to:', logdir)

        not_improved = 0
        best_score = 0
        for epoch in range(opt.start_epoch+1, opt.nEpochs + 1):
            if opt.optim.upper() == 'SGD':
                scheduler.step(epoch)
            train(epoch)
            if (epoch % opt.evalEvery) == 0:
                recalls = test(whole_test_set, epoch, write_tboard=True)
                is_best = recalls[3] > best_score 
                if is_best:
                    not_improved = 0
                    best_score = recalls[3]
                else: 
                    not_improved += 1

                save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'recalls': recalls,
                        'best_score': best_score,
                        'optimizer' : optimizer.state_dict(),
                        'parallel' : isParallel,
                }, is_best)

                if opt.patience > 0 and not_improved > (opt.patience / opt.evalEvery):
                    print('Performance did not improve for', opt.patience, 'epochs. Stopping.')
                    break

        print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
        writer.close()
