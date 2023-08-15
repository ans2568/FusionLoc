import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
import numpy as np
from collections import namedtuple
from PIL import Image
import pandas as pd

from sklearn.neighbors import NearestNeighbors
import h5py

root_dir = 'data/NIA/'
if not exists(root_dir):
    raise FileNotFoundError(root_dir + ' is hardcoded, please adjust to point to custom dataset')

struct_dir = join(root_dir, 'csv')
queries_dir = root_dir


gz_root_dir = 'data/gazebo_dataset/'
if not exists(gz_root_dir):
    raise FileNotFoundError(gz_root_dir + ' is hardcoded, please adjust to point to custom dataset')

gz_struct_dir = join(gz_root_dir, 'csv')
gz_queries_dir = gz_root_dir

iic_dir = 'data/iiclab_real/'
if not exists(iic_dir):
    raise FileNotFoundError(iic_dir + ' is hardcoded, please adjust to point to custom dataset')

iic_struct_dir = join(iic_dir, 'csv')
iic_queries_dir = iic_dir

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])


# .mat 파일이 아닌 .csv 파일로 db_CSV_file, query_CSV_file로 두 개 보내기 
# gazebo 데이터 셋
def get_gazebo_whole_training_set(onlyDB=False): # 학습할 때 사용하는 전체 데이터 셋(DB만 사용)
    dbFile = join(gz_struct_dir, 'train_db_data.csv')
    queryFile = join(gz_struct_dir, 'train_query_data.csv')
    return WholeDatasetFromStruct(dbFile, queryFile,
                             input_transform=input_transform(),
                             onlyDB=onlyDB, dataset='gazebo')

def get_gazebo_whole_test_set():
    dbFile = join(gz_struct_dir, 'test_db_data_reduce.csv')
    queryFile = join(gz_struct_dir, 'test_query_one_data.csv')
    return WholeDatasetFromStruct(dbFile, queryCSVFile=queryFile,
                             input_transform=input_transform(), dataset='gazebo')

def get_gazebo_whole_val_set():
    dbFile = join(gz_struct_dir, 'val_db_data.csv')
    queryFile = join(gz_struct_dir, 'val_query_data.csv')
    return WholeDatasetFromStruct(dbFile, queryFile,
                             input_transform=input_transform(), dataset='gazebo')

def get_gazebo_training_query_set(margin=0.1): # 학습할 때 사용하는 전체 데이터 셋(Query와 DB)
    dbFile = join(gz_struct_dir, 'train_db_data.csv')
    queryFile = join(gz_struct_dir, 'train_query_data.csv')
    return QueryDatasetFromStruct(dbFile, queryFile,
                             input_transform=input_transform(), margin=margin, dataset='gazebo')

# 연구실 데이터 셋
def get_iiclab_whole_training_set(onlyDB=False): # 학습할 때 사용하는 전체 데이터 셋(DB만 사용)
    dbFile = join(iic_struct_dir, 'train_db_data.csv')
    queryFile = join(iic_struct_dir, 'train_query_data.csv')
    return WholeDatasetFromStruct(dbFile, queryFile,
                             input_transform=input_transform(),
                             onlyDB=onlyDB, dataset='iiclab')

def get_iiclab_whole_test_set():
    dbFile = join(iic_struct_dir, 'test_db_data.csv')
    queryFile = join(iic_struct_dir, 'test_query_one_data.csv')
    # queryFile = join(iic_struct_dir, 'test_query_data.csv')
    return WholeDatasetFromStruct(dbFile, queryCSVFile=queryFile,
                             input_transform=input_transform(), dataset='iiclab')

def get_iiclab_whole_val_set():
    dbFile = join(iic_struct_dir, 'val_db_data.csv')
    queryFile = join(iic_struct_dir, 'val_query_data.csv')
    return WholeDatasetFromStruct(dbFile, queryFile,
                             input_transform=input_transform(), dataset='iiclab')

def get_iiclab_training_query_set(margin=0.1): # 학습할 때 사용하는 전체 데이터 셋(Query와 DB)
    dbFile = join(iic_struct_dir, 'train_db_data.csv')
    queryFile = join(iic_struct_dir, 'train_query_data.csv')
    return QueryDatasetFromStruct(dbFile, queryFile,
                             input_transform=input_transform(), margin=margin, dataset='iiclab')

# 청주시외버스터미널 데이터 셋
def get_whole_training_set(onlyDB=False): # 학습할 때 사용하는 전체 데이터 셋(DB만 사용)
    dbFile = join(struct_dir, 'train_db_data.csv')
    queryFile = join(struct_dir, 'train_query_data.csv')
    return WholeDatasetFromStruct(dbFile, queryFile,
                             input_transform=input_transform(),
                             onlyDB=onlyDB)

def get_whole_test_set():
    dbFile = join(struct_dir, 'test_db_data.csv')
    queryFile = join(struct_dir, 'test_query_data.csv')
    return WholeDatasetFromStruct(dbFile, queryCSVFile=queryFile,
                             input_transform=input_transform())

def get_whole_val_set():
    dbFile = join(struct_dir, 'val_db_data.csv')
    queryFile = join(struct_dir, 'val_query_data.csv')
    return WholeDatasetFromStruct(dbFile, queryFile,
                             input_transform=input_transform())

def get_training_query_set(margin=0.1): # 학습할 때 사용하는 전체 데이터 셋(Query와 DB)
    dbFile = join(struct_dir, 'train_db_data.csv')
    queryFile = join(struct_dir, 'train_query_data.csv')
    return QueryDatasetFromStruct(dbFile, queryFile,
                             input_transform=input_transform(), margin=margin)

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
    'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

def parse_dbStruct(db_path, query_path, dataset):
    # 해당 부분에서 csv파일로 읽을 수 있도록 변경하면 .mat파일 사용할 필요가 없음
    db_data = pd.read_csv(db_path, index_col=False)
    query_data = pd.read_csv(query_path, index_col=False)

    dataset = dataset

    # 해당 내용을 구분할 방법이 필요, train, test, val
    whichSet = 'test'

    dbImage = db_data['image_path'].tolist()
    utmDb = []
    for i in range(len(db_data)):
        utmDb.append([db_data.loc[i, 'GT_pose_x'], db_data.loc[i, 'GT_pose_y'], db_data.loc[i, 'GT_pose_theta']])

    qImage = query_data['image_path'].tolist()
    utmQ = []
    for i in range(len(query_data)):
        utmQ.append([query_data.loc[i, 'GT_pose_x'], query_data.loc[i, 'GT_pose_y'], query_data.loc[i, 'GT_pose_theta']])

    numDb = len(db_data['image_path'])
    numQ = len(query_data['image_path'])

    posDistThr = 3
    posDistSqThr = 9
    nonTrivPosDistSqThr = 5

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, 
            utmQ, numDb, numQ, posDistThr, 
            posDistSqThr, nonTrivPosDistSqThr)

class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, dbCSVFile, queryCSVFile, input_transform=None, onlyDB=False, dataset='NIA'):
        super().__init__()

        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(dbCSVFile, queryCSVFile, dataset)
        if dataset == 'gazebo':
            self.images = [join(gz_queries_dir, dbIm[1:]) for dbIm in self.dbStruct.dbImage]
            if not onlyDB:
                self.images += [join(gz_queries_dir, qIm[1:]) for qIm in self.dbStruct.qImage]
        elif dataset == 'NIA':
            self.images = [join(queries_dir, dbIm[1:]) for dbIm in self.dbStruct.dbImage]
            if not onlyDB:
                self.images += [join(queries_dir, qIm[1:]) for qIm in self.dbStruct.qImage]
        elif dataset == 'iiclab':
            self.images = [join(iic_queries_dir, dbIm[:]) for dbIm in self.dbStruct.dbImage]
            if not onlyDB:
                self.images += [join(iic_queries_dir, qIm[:]) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def get(self, index):
        path = self.images[index]
        img = Image.open(path)
        tf = transforms.Compose([transforms.ToTensor()])
        img = tf(img)
        if path.endswith(".png"):
            if self.dataset == 'gazebo':
                path = path[-18:-4]
            elif self.dataset == 'NIA':
                path = path[-22:-4]
            elif self.dataset == 'iiclab':
                # @TODO dataset에따라 이름 설정
                path = path[-18:-4]
        return img, path

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        #fit NN to find them, search by radius
        if  self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.dbStruct.utmDb)

            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,
                    radius=self.dbStruct.posDistThr)
        return self.positives

def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).
    
    Args:
        data: list of tuple (query, positive, negatives). 
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))
    return query, positive, negatives, negCounts, indices

class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, dbCSVFile, queryCSVFile, nNegSample=50, nNeg=2, margin=0.1, input_transform=None, dataset='NIA'):
        super().__init__()

        self.input_transform = input_transform
        self.margin = margin

        self.dbStruct = parse_dbStruct(dbCSVFile, queryCSVFile, dataset)
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = nNegSample # number of negatives to randomly sample
        self.nNeg = nNeg # number of negatives used for training

        # potential positives are those within nontrivial threshold range
        #fit NN to find them, search by radius
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)

        # TODO use sqeuclidean as metric?
        self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.nonTrivPosDistSqThr**0.5, 
                return_distance=False))
        # radius returns unsorted, sort once now so we dont have to later
        for i,posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)
        # its possible some queries don't have any non trivial potential positives
        # lets filter those out
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives])>0)[0]
        # potential negatives are those outside of posDistThr range
        potential_positives = knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.posDistThr, 
                return_distance=False)
        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb),
                pos, assume_unique=True))

        self.cache = None # filepath of HDF5 containing feature vectors for images

        self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]

    def __getitem__(self, index):
        index = self.queries[index] # re-map index to match dataset
        with h5py.File(self.cache, mode='r') as h5: 
            h5feat = h5.get("features")

            # qOffset = self.dbStruct.numDb 
            # qFeat = h5feat[index+qOffset]
            qFeat = h5feat[index]
            posFeat = h5feat[self.nontrivial_positives[index].tolist()]
            knn = NearestNeighbors(n_jobs=-1) # TODO replace with faiss?
            knn.fit(posFeat)
            dPos, posNN = knn.kneighbors(qFeat.reshape(1,-1), 1)
            dPos = dPos.item()
            posIndex = self.nontrivial_positives[index][posNN[0]].item()

            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]))
            negSample = negSample.astype(np.int32)
            negFeat = h5feat[negSample.tolist()]
            knn.fit(negFeat)

            dNeg, negNN = knn.kneighbors(qFeat.reshape(1,-1),
                    self.nNeg) # to quote netvlad paper code: 10x is hacky but fine
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)

            # try to find negatives that are within margin, if there aren't any return none
            violatingNeg = dNeg < dPos + self.margin**0.5
     
            if np.sum(violatingNeg) < 1:
                #if none are violating then skip this query
                return None

            negNN = negNN[violatingNeg][:self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)
            self.negCache[index] = negIndices
        if self.dataset == 'gazebo':
            query = Image.open(join(gz_queries_dir, self.dbStruct.qImage[index][1:]))
            positive = Image.open(join(gz_queries_dir, self.dbStruct.dbImage[posIndex][1:]))
        elif self.dataset == 'NIA':
            query = Image.open(join(queries_dir, self.dbStruct.qImage[index][1:]))
            positive = Image.open(join(queries_dir, self.dbStruct.dbImage[posIndex][1:]))
        elif self.dataset == 'iiclab':
            query = Image.open(join(iic_queries_dir, self.dbStruct.qImage[index][:]))
            positive = Image.open(join(iic_queries_dir, self.dbStruct.dbImage[posIndex][:]))

        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)

        negatives = []
        for negIndex in negIndices:
            if self.dataset == 'gazebo':
                negative = Image.open(join(gz_queries_dir, self.dbStruct.dbImage[negIndex][1:]))
            elif self.dataset == 'NIA':
                negative = Image.open(join(queries_dir, self.dbStruct.dbImage[negIndex][1:]))
            elif self.dataset == 'iiclab':
                negative = Image.open(join(iic_queries_dir, self.dbStruct.dbImage[negIndex][:]))
            if self.input_transform:
                negative = self.input_transform(negative)
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [index, posIndex]+negIndices.tolist()

    def __len__(self):
        return len(self.queries)
