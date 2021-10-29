#!/bin/bash
MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)
MASTER_PORT=${PORT:-20000}
NNODES=${SLURM_NNODES}
RANK=${SLURM_NODEID}

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --nnodes=$NNODES \
  --node_rank=$RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  run_beit_pretraining.py $*
