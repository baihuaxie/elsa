#!/bin/bash

#export LD_LIBRARY_PATH=/home/baihuaxie/anaconda3/envs/elsa/lib/:$LD_LIBRARY_PATH

# specify the experiments
EXPERIMENT="owt/gpt2m"

# specify project name
PROJECT_NAME="elsa-new"

# specify run id
RUN_ID="orion-alpha"

# specify the path to the log directory
BASE_DIR="./checkpoints"
LOG_DIR="${BASE_DIR}/${EXPERIMENT}/${RUN_ID}"
mkdir -p $LOG_DIR

# run the python script with nohup and save the output to the log file
nohup python run.py experiment=$EXPERIMENT loggers.wandb.project=$PROJECT_NAME loggers.wandb.id=$RUN_ID "$@"  >> "${LOG_DIR}/run.log" &