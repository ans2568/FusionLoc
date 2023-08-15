#cluster
python main.py --mode=cluster --arch=vgg16 --pooling=netvlad --num_clusters=64 --dataset=NIA

# train
python main.py --mode=train --dataset=NIA

# test
python main.py --mode=test --split=test --resume=/root/wip/FusionLoc/runsPath/Aug15_07-37-21_vgg16_netvlad/ --dataset=NIA