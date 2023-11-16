#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
BACKBONE='mit_b1'
PORT=${PORT:-1777}
save_path='dual_teacher'
mkdir -p $save_path
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1  --node_rank=0 --master_addr=localhost --master_port=$PORT tools/train.py --port $PORT --backbone $BACKBONE --save_path '1022/1263_print_test' --ddp

