#!/bin/bash

export LD_LIBRARY_PATH=/home/baihuaxie/anaconda3/envs/elsa/lib/:$LD_LIBRARY_PATH
python run.py "$@"