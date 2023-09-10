# 영상 및 라이다 데이터로부터 사전 데이터베이스 만들기

```scripts/test_preExtract.py``` 파일 이용 시 미리 DB 특징 추출이 필요하므로 ```saveDBFeature.py``` 파일로 해당 과정 수행

```bash
# extract pre-build map feature
python saveDBFeature.py --resume=../runsPath/Aug21_09-46-30_vgg16_netvlad/ --dataset=NIA
```