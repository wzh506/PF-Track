#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28500}

TORCH_CUDNN_V8_API_ENABLED=1 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
# bash tools/dist_train.sh projects/configs/tracking/petr/f3_q5_800x320.py 2 --work-dir work_dirs/f3_pf_track/ 