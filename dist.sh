#!/bin/bash
# This scropt is for distributed training
# ---------------------------------------
# `run_dist.sh' will access each remote node and excute this script.
# You do not have to run it manually.
HOST_FILE=./Hosts
WORLD_SIZE=$(cat ${HOST_FILE} | wc -l)

python train_kinetics.py \
--world-size ${WORLD_SIZE} \
2>&1 | tee -a full-record.log

# add other options if you want:
# --resume-epoch 5
# etc...