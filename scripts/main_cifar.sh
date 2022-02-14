#!/bin/bash

###########################
#		main task
###########################

EPOCH=500
BS=256

python main_ce.py --batch_size ${BS} --dataset 'cifar10' --save_freq 50 --num_workers 20 --cosine --data_folder ${DATADIR} --epochs ${EPOCH} --save_dir ${SAVEIDR}
