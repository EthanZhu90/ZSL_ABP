# ZSL_ABP
code for the conference paper:

Yizhe Zhu, Jianwen Xie , Bingchen Liu, Ahmed Elgammal
"Learning Feature-to-Feature Translator by Alternating Back-Propagation for Zero-Shot Learning", ICCV, 201９


## Results evaluated on [GBU setting](https://arxiv.org/abs/1707.00600)[1] 

Download the [data](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/) and compress it to the folder 'data/'.


```shell
python train_GBU.py --dataset CUB --preprocessing --z_dim 100
python train_GBU.py --dataset AWA1 --preprocessing --z_dim 10
python train_GBU.py --dataset AWA2 --preprocessing --z_dim 10
python train_GBU.py --dataset SUN --preprocessing --z_dim 10
```
