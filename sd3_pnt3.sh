#!/bin/bash
#PBS -l select=1:ncpus=4:mem=480gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=72:00:00

eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda activate flow_grpo

conda install nvidia::cuda-nvcc

cd ~/flow_grpo

export WANDB_API_KEY="ceaaf9b7ea779499fff2a3a9b02dccab03b0c043"

accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=1 --main_process_port 29501 \
  -m scripts.train_sd3_pnt2 --config config/grpo.py:general_ocr_sd3_pnt_1gpu