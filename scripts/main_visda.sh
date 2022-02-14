#!/bin/bash

###########################
#		main task
###########################

EPOCH=10
LR=0.0001
BS=128

python main_ce.py --batch_size ${BS} --dataset 'visda' --save_freq 1 --num_workers 20 --cosine --data_folder ${DATADIR} --epochs ${EPOCH} --save_dir ${SAVEIDR} --learning_rate ${LR}
