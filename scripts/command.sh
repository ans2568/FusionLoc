#cluster
python main.py --mode=cluster --arch=vgg16 --pooling=netvlad --num_clusters=64 --dataset=NIA

# train
python main.py --mode=train --dataset=NIA

# test
python test.py --resume=../runsPath/Aug15_07-37-21_vgg16_netvlad/ --dataset=NIA

# extract pre-build map feature
python util/saveDBFeature.py --dataset=7scenes --resume=../runsPath/Aug21_09-46-30_vgg16_netvlad/

# test by using pre-built map feature
python test_preExtract.py --resume=../runsPath/Aug21_09-46-30_vgg16_netvlad/ --dataset=7scenes --mode=camera