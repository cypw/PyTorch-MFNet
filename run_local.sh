#!/bin/bash
# This scropt is for fine-tuning on single node
# ---------------------------------------
# Note that:
# - You may need to tune the params

export PATH=$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH

python train_ucf101.py

# python train_hmdb51.py

# python train_kinetics.py 
