#!/bin/bash
#PBS -l select=1:ncpus=4:mem=480gb:ngpus=4:gpu_type=L40S
#PBS -l walltime=72:00:00

eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda activate flow_grpo

conda install nvidia::cuda-nvcc

cd ~/flow_grpo

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600  # 2 hours instead of 30 minutes

export WANDB_API_KEY="ceaaf9b7ea779499fff2a3a9b02dccab03b0c043"

accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero3.yaml \
  --num_processes=4 --main_process_port 29501 \
  -m scripts.train_sd3_pnt2 --config config/grpo.py:pickscore_sd3_pnt_4gpu