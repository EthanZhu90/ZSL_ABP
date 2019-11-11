# ZSL_ABP
code for the conference paper:

Yizhe Zhu, Jianwen Xie , Bingchen Liu, Ahmed Elgammal
"Learning Feature-to-Feature Translator by Alternating Back-Propagation for Zero-Shot Learning", ICCV, 2019


## Results evaluated on [GBU setting](https://arxiv.org/abs/1707.00600)[1] 

Download the [data](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/) and compress it to the folder 'data/'.

To train the model, run the following command. 
CUB datset:
```shell
python train_ABP.py --dataset CUB --z_dim 10 --sigma 0.3 --langevin_s 0.1 --langevin_step 5   --batchsize 64 --nSample 300
```
AWA1 datset:
```shell
python train_ABP.py --dataset AWA1 --z_dim 10 --sigma 0.3 --langevin_s 0.1 --langevin_step 5   --batchsize 64 --nSample 1500
```
AWA2 datset:
```shell
python train_ABP.py --dataset AWA2 --z_dim 10 --sigma 0.3 --langevin_s 0.1 --langevin_step 5   --batchsize 64 --nSample 1500
```
SUN datset:
```shell
python train_ABP.py --dataset SUN  --z_dim 10 --sigma 0.3 --langevin_s 0.1 --langevin_step 5   --batchsize 64 --nSample 300 
```


