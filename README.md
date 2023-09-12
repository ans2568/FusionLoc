# 영상 및 라이다 기반 로봇의 초기 포즈 추정

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
  - `추후 이름 변경 예정`

# Usage

`scripts/train.py` 파일은 NetVLAD 학습 시 사용하는 파일로 --dataset 설정 필요

**새로운 데이터 셋 추가 시 코드 수정이 필요함**
- `scripts/pose_estimation.py`, `scripts/Struct.py`, `scripts/util/load.py` 파일들에 경로 추가 및 `--dataset`에 추가
    - 추후 ArgumentParser로 실행 시 argument로 변경 예정

## CLI args

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

---
## 학습 및 테스트

### Cluster

**학습을 진행하기에 앞서 hdf5 파일 생성이 필요**

```bash
python scripts/cluster.py --dataset=NIA
```

### Train

cluster.py 진행 후 NetVLAD 학습 진행

```bash
python scripts/train.py --dataset=NIA
```

### Test

`--resume : train.py로 학습된 가중치 파일의 checkpoints 경로`

```bash
python scripts/test.py --resume=runsPath/Apr22_17-03-05_vgg16_netvlad/ --dataset=NIA
```

**Note : 만약, DB 특징을 미리 추출하고 Test를 진행하고 싶은 경우 다음과 같이 진행**

```bash
# extract pre-build map feature
python util/saveDBFeature.py --resume=runsPath/Aug21_09-46-30_vgg16_netvlad/ --dataset=NIA

# test by using pre-built map feature
python scripts/test_preExtract.py --resume=runsPath/Aug21_09-46-30_vgg16_netvlad/ --dataset=NIA
```