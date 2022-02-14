#!/bin/bash

##############################
#     joint (main + ssl)
##############################

EPOCH=500
BS=256
BALANCE=0.9

CKPT=${SAVEIDR}/cifar10_models/SupCE_cifar10_resnet50_lr_0.2_decay_0.0001_bsz_256_trial_0_cosine/last.pth

python main_joint.py --batch_size ${BS} --dataset 'cifar10' --save_freq 50 --num_workers 20 --cosine --data_folder ${DATADIR} --epochs ${EPOCH} --save_dir ${SAVEIDR} --ckpt ${CKPT} --balance ${BALANCE}
