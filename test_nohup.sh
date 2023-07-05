#!/bin/bash

# specify the name of the python script to run
SCRIPT_NAME="run.py"

# specify the experiments
EXPERIMENT="owt/gpt2s"

# specify project name
PROJECT_NAME="elsa-develop"

# specify run id
RUN_ID="test-nohup"

# specify the path to the log directory
BASE_DIR="./checkpoints"
LOG_DIR="${BASE_DIR}/${EXPERIMENT}/${RUN_ID}"
mkdir -p $LOG_DIR

# specify the maximum steps
MAX_STEPS="10"

# run the python script with nohup and save the output to the log file
nohup python $SCRIPT_NAME experiment=$EXPERIMENT trainer.max_steps=$MAX_STEPS loggers.wandb.project=$PROJECT_NAME loggers.wandb.id=$RUN_ID > "${LOG_DIR}/run.log" &
