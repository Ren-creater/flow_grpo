#!/bin/bash
#PBS -l select=1:ncpus=4:mem=480gb:ngpus=4:gpu_type=L40S
#PBS -l walltime=72:00:00

eval "$(~/anaconda3/bin/conda shell.bash hook)"

source activate tpdm

cd ~/flow_grpo/TPDM

export NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
export OMP_NUM_THREADS=4
export WANDB_PROJECT="sd35_vit_timepredictor"
export WANDB_MODE="offline"
export RUN_NAME="sd35_vit_pnt_$(date +'%Y%m%d_%H%M%S')"

OUTPUT_DIR="outputs/$(date +'%Y-%m-%d')/$RUN_NAME"

echo "Starting SD3.5 ViT TimePredictor training..."
echo "Number of GPUs: $NUM_GPUS"
echo "Run name: $RUN_NAME"
echo "Output directory: $OUTPUT_DIR"

python -m torch.distributed.run --nproc_per_node $NUM_GPUS --nnodes 1 --standalone \
    main_diff_rloo_trainer.py \
    --model_config configs/models/sd35_pnt_vit.yaml \
    --reward_model_config configs/models/image_reward.yaml \
    --train_dataset configs/datasets/hf_json_list.yaml \
    --data_collator configs/datasets/json_prompt_collator.yaml \
    --gamma 0.97 \
    --world_size $NUM_GPUS \
    --init_alpha 2.0 \
    --init_beta 1.0 \
    --kl_coef 0.00 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --warmup_steps 100 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --num_train_epochs 3 \
    --eval_steps 100 \
    --save_steps 200 \
    --torch_empty_cache_steps 10 \
    --logging_steps 5 \
    --report_to wandb \
    --resume_from_checkpoint true \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    --deepspeed configs/deepspeed/deepspeed_stage_1.json \
    --fp16 true \
    --dataloader_num_workers 4 \
    --remove_unused_columns false

echo '--------------------------'
echo SD3.5 ViT TimePredictor training task done
echo '--------------------------'

# Optional: Run evaluation after training
echo "Running final evaluation..."
python -m torch.distributed.run --nproc_per_node $NUM_GPUS --nnodes 1 --standalone \
    eval_model.py \
    --model_config configs/models/sd35_pnt_vit.yaml \
    --checkpoint_path $OUTPUT_DIR/final_checkpoint \
    --eval_dataset configs/datasets/eval_prompts.yaml \
    --output_dir $OUTPUT_DIR/evaluation \
    --batch_size 4 \
    --num_samples 1000

echo "Evaluation completed. Results saved to $OUTPUT_DIR/evaluation"
