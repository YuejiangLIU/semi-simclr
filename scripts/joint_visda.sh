#!/bin/bash

##############################
#     joint (main + ssl)
##############################

EPOCH=10
LR=0.0001
BS=128

BALANCE=0.9

CKPT=${SAVEIDR}/visda_models/SupCE_visda_resnet50_lr_0.0001_decay_0.0001_bsz_128_trial_0_cosine/last.pth

python main_joint.py --batch_size ${BS} --dataset 'visda' --save_freq 10 --num_workers 20 --cosine --data_folder ${DATADIR} --epochs ${EPOCH} --save_dir ${SAVEIDR} --learning_rate ${LR} --ckpt ${CKPT} --balance ${BALANCE} --trainable 'head'
