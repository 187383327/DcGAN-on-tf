# DcGAN-on-tf
using DcGAN for generating japanese cartoon head image
using Tenorflow1.10, but i think version >=1.4 is ok

firstly you need to be prepare for dataset
dataset is from here: https://pan.baidu.com/s/1FkhBnDS50EddnFDGYSWyEw

secondly, change the variable named dataset in  'train()' in file './DcGan_train.py'
and runing your code

open tensorboard to look at effect and graph. 
using  tensorboard --logdir=logs   to open tensorboard in terminal

![image](https://github.com/shoutOutYangJie/DcGAN-on-tf/blob/master/training_on_5_epochs/QQ%E6%88%AA%E5%9B%BE20190119180148.jpg)

