# SFT: Few-Shot Learning via Self-supervised Feature Fusion with Transformer

This repository contains the **pytorch** code for the paper: "[SFT: Few-Shot Learning via Self-supervised Feature Fusion with Transformer](https://doi.org/10.1109/ACCESS.2024.3416327)" Jit Yan Lim, Kian Ming Lim, Chin Poo Lee, Yong Xuan Tan

## Environment
The code is tested on Windows 10 with Anaconda3 and following packages:
- python 3.7.4
- pytorch 1.3.1

## Preparation
1. Change the ROOT_PATH value in the following files to yours:
    - `datasets/mini_imagenet.py`
    - `datasets/tiered_imagenet.py`
    - `datasets/cifarfs.py`

2. Download the datasets and put them into corresponding folders that mentioned in the ROOT_PATH:<br/>
    - ***mini*ImageNet**: download from [CSS](https://github.com/anyuexuan/CSS) and put in `data/mini-imagenet` folder.

    - ***tiered*ImageNet**: download from [RFS](https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0) and put in `data/tiered-imagenet-kwon` folder.

    - **CIFARFS**: download from [MetaOptNet](https://github.com/kjunelee/MetaOptNet) and put in `data/cifar-fs` folder.

## Pre-trained Models
[Optional] The pre-trained models can be downloaded from [here](https://drive.google.com/file/d/1sHG7GJIL4z-l6TA3KwTeNtTgw4oORE8A/view?usp=drive_link). Extract and put the content in the `save` folder. To evaluate the model, run the `test_TL.py` file with the proper save path as in the next section.

## Experiments
To pre-train on MiniImageNet:<br/>
```
python train_rot.py --gpu 0 --gamma 2.0 --dataset mini --save-path ./save/mini-stage1-rot
python train_dist.py --gpu 0 --gamma 0.05 --dataset mini --save-path ./save/mini-stage1-dist
```
To train on 5-way 1-shot and 5-way 5-shot MiniImageNet:<br/>
```
python train_TL.py --gpu 0 --shot 1 --h-dim 640 --dropout 0.4 --beta 0.5 --gamma 0.5 --lr 0.0001 --balance 1.0 --temperature 32 --temperature2 128 --epochs 60 --dataset mini --pretrain-path-1 ./save/mini-stage1-rot --pretrain-path-2 ./save/mini-stage1-dist --save-path ./save/mini-stage2-1shot
python train_TL.py --gpu 0 --shot 5 --h-dim 640 --dropout 0.4 --beta 0.5 --gamma 0.5 --lr 0.0001 --balance 1.0 --temperature 32 --temperature2 64 --epochs 60 --dataset mini --pretrain-path-1 ./save/mini-stage1-rot --pretrain-path-2 ./save/mini-stage1-dist --save-path ./save/mini-stage2-5shot
```
To evaluate on 5-way 1-shot and 5-way 5-shot MiniImageNet:<br/>
```
python test_TL.py --gpu 0 --shot 1 --h-dim 640 --beta 0.5 --gamma 0.5 --dataset mini --pretrain-path-1 ./save/mini-stage1-rot --pretrain-path-2 ./save/mini-stage1-dist --save-path ./save/mini-stage2-1shot
python test_TL.py --gpu 0 --shot 5 --h-dim 640 --beta 0.5 --gamma 0.5 --dataset mini --pretrain-path-1 ./save/mini-stage1-rot --pretrain-path-2 ./save/mini-stage1-dist --save-path ./save/mini-stage2-5shot
```

To pre-train on TieredImageNet:<br/>
```
python train_rot.py --gpu 0 --gamma 2.0 --dataset tiered --save-path ./save/tiered-stage1-rot
python train_dist.py --gpu 0 --gamma 0.02 --dataset tiered --save-path ./save/tiered-stage1-dist
```
To train on 5-way 1-shot and 5-way 5-shot TieredImageNet:<br/>
```
python train_TL.py --gpu 0 --shot 1 --h-dim 640 --dropout 0.2 --beta 0.5 --gamma 0.5 --lr 0.00005 --balance 0.01 --temperature 32 --temperature2 128 --epochs 60 --lr-decay-epochs 50 --dataset tiered --pretrain-path-1 ./save/tiered-stage1-rot --pretrain-path-2 ./save/tiered-stage1-dist --save-path ./save/tiered-stage2-1shot
python train_TL.py --gpu 0 --shot 5 --h-dim 640 --dropout 0.2 --beta 0.5 --gamma 0.5 --lr 0.00005 --balance 1.0 --temperature 64 --temperature2 128 --epochs 60 --lr-decay-epochs 50 --dataset tiered --pretrain-path-1 ./save/tiered-stage1-rot --pretrain-path-2 ./save/tiered-stage1-dist --save-path ./save/tiered-stage2-5shot
```
To evaluate on 5-way 1-shot and 5-way 5-shot TieredImageNet:<br/>
```
python test_TL.py --gpu 0 --shot 1 --h-dim 640 --beta 0.5 --gamma 0.5 --dataset tiered --pretrain-path-1 ./save/tiered-stage1-rot --pretrain-path-2 ./save/tiered-stage1-dist --save-path ./save/tiered-stage2-1shot
python test_TL.py --gpu 0 --shot 5 --h-dim 640 --beta 0.5 --gamma 0.5 --dataset tiered --pretrain-path-1 ./save/tiered-stage1-rot --pretrain-path-2 ./save/tiered-stage1-dist --save-path ./save/tiered-stage2-5shot
```


## Citation
If you find this repo useful for your research, please consider citing the paper:
```
@ARTICLE{10559997,
  author={Lim, Jit Yan and Lim, Kian Ming and Lee, Chin Poo and Tan, Yong Xuan},
  journal={IEEE Access}, 
  title={SFT: Few-Shot Learning via Self-Supervised Feature Fusion With Transformer}, 
  year={2024},
  volume={12},
  number={},
  pages={86690-86703},
  doi={10.1109/ACCESS.2024.3416327}
}
```


## Contacts
For any questions, please contact: <br/>

Jit Yan Lim (jityan95@gmail.com) <br/>
Kian Ming Lim (Kian-Ming.Lim@nottingham.edu.cn)

## Acknowlegements
This repo is based on **[Prototypical Networks](https://github.com/yinboc/prototypical-network-pytorch)**, **[RFS](https://github.com/WangYueFt/rfs)**, **[SKD](https://github.com/brjathu/SKD)**, and **[FEAT](https://github.com/Sha-Lab/FEAT)**.
