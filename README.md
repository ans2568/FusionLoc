# 지도가 주어진 상태에서 로봇의 포즈 추정

## Dataset
`NIA_dataset`

`iicnas.kangwon.ac.kr`
- 청주시외버스터미널 -> 시나리오 C -> sunny -> 221110_09-11
- 고려대 세종 캠퍼스 -> 시나리오 A -> sunny -> 20220928_09-11

# Setup

## Dependencies

1. [PyTorch](https://pytorch.org/get-started/locally/) (at least v0.4.0)
2. [Faiss](https://github.com/facebookresearch/faiss)
3. [scipy](https://www.scipy.org/)
    - [numpy](http://www.numpy.org/)
    - [sklearn](https://scikit-learn.org/stable/)
    - [h5py](https://www.h5py.org/)
4. [tensorboardX](https://github.com/lanpa/tensorboardX)

## Data

Camera
- 파일 이름이 timestamp인 이미지 파일

LiDAR
- 파일 이름이 timestamp인 pcd 파일

csv
- timestamp, GT_pose_x, GT_pose_y, GT_pose_theta, image_path, lidar_path

# Usage

`main_custom.py` 파일은 train, test, cluster 모드를 전부 포함한 파일로 --mode=[`train`, `test`, `cluster`]에 따라 실행이 달라짐

새로운 데이터 셋 추가 시 코드 수정이 필요함
- `opticalFlow.py`, `icp_pcl.py`, `Struct.py`, `custom.py` 파일들에 경로 추가 및 `--dataset`에 추가

## ros2bag to csv

bag 파일을 읽어와 csv 파일로 만들고, 각 timestamp에 맞는 이미지와 LiDAR 데이터를 저장

bag 파일의 경로는 직접 파일 내부에서 수정(`bag2csv.py`)

```bash
python bag2csv.py
```

## CLI args

- `--mode` : train, test, cluster 중 선택(default : train)
- `--batchSize` : Number of triplets (query, positive, negatives) 각 triplet은 12개의 이미지를 포함
- `--cacheBatchSize` : 배치 사이즈
- `--cacheRefreshRate` : How often to refresh cache
- `--nEpochs` : 학습에 사용할 에포크 수
- `--start-epoch` : 시작할 에포크 숫자
- `--nGPU` : GPU 사용 개수
- `--optim` : 어떤 optimizer를 사용할 것인지 [`SGD`, `ADAM`]
- `--lr` : learning rate
- `--lrStep` : 몇 에포크 마다 learning rate를 decay 할 건지
- `--lrGamma` : Multiply LR by Gamma for decaying
- `--weightDecay` : Weight decay for SGD
- `--momentum` : Momentum for SGD
- `--threads` : dataloader에 사용할 thread 숫자
- `--seed` : 랜덤 시드
- `--dataPath` : 학습 시 사용하는 cluster를 저장한 데이터(centroid data path)
- `--runsPath` : `--resume`로 저장한 checkpoint의 폴더 경로
- `--ckpt` : `--resume`로 checkpoint를 저장할 때 어떤 checkpoint를 사용할 것인지 [`latest`, `best`]
- `--evalEvery` : 검증을 몇 에포크마다 할 것인지(default : 1)
- `--arch` : 어떤 아키텍쳐 사용할 것인가 [`vgg16`, `alexnet`](default : `vgg16`)
- `--pooling` : 어떤 풀링 방식 사용할 것인가(default : `netvlad`) [`netvlad`, `max`, `avg`]
- `--num_clusters` : NetVLAD 클러스터의 개수 (default : 64)
- `--split` : test 시 사용할 데이터 셋 (default : val) [`test`, `train`, `val`]
- `--dataset` : NIA 데이터 셋을 사용할 것인지 gazebo 데이터 셋을 사용할 것인지 (default : `NIA`) [`NIA`, `gazebo`]


## Train

학습을 진행하려면 먼저 --mode=cluster 모드로 hdf5 파일을 생성시킨 후 진행해야함

```bash
python main_custom.py --mode=train --savePath=custom_checkpoint/checkpoint --cacheBatchSize=10 --cacheRefreshRate=0 --threads=4 --dataset=gazebo
```

## Test

`--resume`는 옵션

```bash
python main_custom.py --mode=test --split=test --resume=/root/wip/pytorch-NetVlad/runsPath/Apr22_17-03-05_vgg16_netvlad/custom_checkpoint/ --dataset=gazebo
```

## Cluster

```bash
python main_custom.py --mode=cluster --arch=vgg16 --pooling=netvlad --num_clusters=64 --dataset=gazebo
```

## ROS2 bag file
gazebo 시뮬레이션을 활용하여 데이터 셋을 구축 시 ros2 bag을 활용한 토픽 메세지 녹화

```bash
ros2 bag record -a -o data.bag
```

녹화된 bag 파일을 읽어와 csv 파일로 변환

```bash
python3 bag2csv.py
```

csv 파일에서 같은 timestamp 끼리 그룹화 및 필요한 데이터가 없는 행의 경우 삭제시켜서 완전한 데이터 셋을 만드는 파이썬 파일

```bash
python3 synchronize.py
```

순서
1. ros2 bag cli로 bag 파일 생성
2. bag2csv.py 실행
3. synchronize.py 실행
4. 아키텍쳐에 활용할 csv 파일 및 이미지, PCD 파일 생성
