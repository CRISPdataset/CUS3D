#!/usr/bin/env bash
CONFIG=$1
CHECK_POINT=$2
GPUS=$3
PORT=${PORT:-29501}

OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --master_port=$PORT  $(dirname "$0")/heatmap.py $CONFIG $CHECK_POINT --dist ${@:4}
#--out outs_text_debug --text │····························"chair"
