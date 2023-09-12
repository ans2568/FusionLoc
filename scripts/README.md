# 학습 및 테스트

### Cluster

**학습을 진행하기에 앞서 hdf5 파일 생성이 필요**

```bash
cd ~/FusionLoc
python scripts/cluster.py --dataset=NIA
```

### Train

cluster.py 진행 후 NetVLAD 학습 진행

```bash
cd ~/FusionLoc
python scripts/train.py --dataset=NIA
```

### Test

`--resume : train.py로 학습된 가중치 파일의 checkpoints 경로`

```bash
cd ~/FusionLoc
python scripts/test.py --resume=runsPath/Apr22_17-03-05_vgg16_netvlad/ --dataset=NIA
```

**Note : 만약, DB 특징을 미리 추출하고 Test를 진행하고 싶은 경우 다음과 같이 진행**

```bash
cd ~/FusionLoc
# extract pre-build map feature
python scripts/util/saveDBFeature.py --resume=runsPath/Aug21_09-46-30_vgg16_netvlad/ --dataset=NIA

# test by using pre-built map feature
python scripts/test_preExtract.py --resume=runsPath/Aug21_09-46-30_vgg16_netvlad/ --dataset=NIA
```