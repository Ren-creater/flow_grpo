#!/bin/bash
#PBS -l select=1:ncpus=4:mem=480gb:ngpus=4:gpu_type=L40S
#PBS -l walltime=72:00:00

eval "$(~/anaconda3/bin/conda shell.bash hook)"

source activate tpdm

cd ~/flow_grpo/TPDM

export NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
export OMP_NUM_THREADS=4
export WANDB_PROJECT="test"
export WANDB_MODE="offline"
export RUN_NAME="test"

OUTPUT_DIR="outputs/$(date +'%Y-%m-%d')/$RUN_NAME"

python -m torch.distributed.run --nproc_per_node $NUM_GPUS --nnodes 1 --standalone \
    main_diff_rloo_trainer.py \
    --model_config configs/models/sd3_pnt.yaml \
    --reward_model_config configs/models/image_reward.yaml \
    --train_dataset configs/datasets/hf_json_list.yaml \
    --data_collator configs/datasets/json_prompt_collator.yaml \
    --gamma 0.97 \
    --world_size $NUM_GPUS \
    --init_alpha 2.5 \
    --init_beta 1.0 \
    --kl_coef 0.00 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_steps 0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --adam_epsilon 1e-5 \
    --weight_decay 0.0 \
    --max_grad_norm 1.0 \
    --num_train_epochs 1 \
    --eval_steps 50 \
    --save_steps 100 \
    --torch_empty_cache_steps 10 \
    --logging_steps 1 \
    --report_to wandb \
    --resume_from_checkpoint true \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    --deepspeed configs/deepspeed/deepspeed_stage_0.json
# --fsdp "shard_grad_op auto_wrap" \
# --fsdp_config configs/fsdp/fsdp_sd3.json

echo '--------------------------'
echo main training task done
echo '--------------------------'