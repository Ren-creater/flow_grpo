#!/bin/bash
#PBS -l select=1:ncpus=4:mem=480gb:ngpus=4:gpu_type=L40S
#PBS -l walltime=72:00:00

# Usage: qsub -v MODEL_TYPE=sd35_vit,PREDICTOR_TYPE=vit launch_flexible_train.sh
# MODEL_TYPE: sd3, sd35
# PREDICTOR_TYPE: cnn, vit

eval "$(~/anaconda3/bin/conda shell.bash hook)"

source activate tpdm

cd ~/flow_grpo/TPDM

export NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
export OMP_NUM_THREADS=4
export WANDB_PROJECT="timepredictor_experiments"
export WANDB_MODE="offline"

# Set defaults if not provided
MODEL_TYPE=${MODEL_TYPE:-sd35}
PREDICTOR_TYPE=${PREDICTOR_TYPE:-vit}

export RUN_NAME="${MODEL_TYPE}_${PREDICTOR_TYPE}_pnt_$(date +'%Y%m%d_%H%M%S')"
OUTPUT_DIR="outputs/$(date +'%Y-%m-%d')/$RUN_NAME"

echo "Starting TimePredictor training..."
echo "Model type: $MODEL_TYPE"
echo "Predictor type: $PREDICTOR_TYPE"
echo "Number of GPUs: $NUM_GPUS"
echo "Run name: $RUN_NAME"
echo "Output directory: $OUTPUT_DIR"

# Choose model config based on parameters
if [ "$MODEL_TYPE" = "sd35" ] && [ "$PREDICTOR_TYPE" = "vit" ]; then
    MODEL_CONFIG="configs/models/sd35_pnt_vit.yaml"
    BATCH_SIZE=6
    GRAD_ACCUM=2
    LEARNING_RATE=5e-6
    SCHEDULER="cosine"
    WARMUP_STEPS=100
    WEIGHT_DECAY=0.01
elif [ "$MODEL_TYPE" = "sd35" ] && [ "$PREDICTOR_TYPE" = "cnn" ]; then
    MODEL_CONFIG="configs/models/sd3_pnt.yaml"  # Will need to create sd35_pnt.yaml
    BATCH_SIZE=8
    GRAD_ACCUM=1
    LEARNING_RATE=1e-6
    SCHEDULER="constant_with_warmup"
    WARMUP_STEPS=0
    WEIGHT_DECAY=0.0
elif [ "$MODEL_TYPE" = "sd3" ] && [ "$PREDICTOR_TYPE" = "vit" ]; then
    MODEL_CONFIG="configs/models/sd3_pnt_vit.yaml"
    BATCH_SIZE=8
    GRAD_ACCUM=1
    LEARNING_RATE=1e-5
    SCHEDULER="cosine"
    WARMUP_STEPS=50
    WEIGHT_DECAY=0.01
else
    MODEL_CONFIG="configs/models/sd3_pnt.yaml"
    BATCH_SIZE=8
    GRAD_ACCUM=1
    LEARNING_RATE=1e-6
    SCHEDULER="constant_with_warmup"
    WARMUP_STEPS=0
    WEIGHT_DECAY=0.0
fi

echo "Using model config: $MODEL_CONFIG"
echo "Batch size: $BATCH_SIZE, Grad accumulation: $GRAD_ACCUM"
echo "Learning rate: $LEARNING_RATE, Scheduler: $SCHEDULER"

python -m torch.distributed.run --nproc_per_node $NUM_GPUS --nnodes 1 --standalone \
    main_diff_rloo_trainer.py \
    --model_config $MODEL_CONFIG \
    --reward_model_config configs/models/image_reward.yaml \
    --train_dataset configs/datasets/hf_json_list.yaml \
    --data_collator configs/datasets/json_prompt_collator.yaml \
    --gamma 0.97 \
    --world_size $NUM_GPUS \
    --init_alpha 2.0 \
    --init_beta 1.0 \
    --kl_coef 0.00 \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type $SCHEDULER \
    --warmup_steps $WARMUP_STEPS \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --weight_decay $WEIGHT_DECAY \
    --max_grad_norm 1.0 \
    --num_train_epochs 2 \
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
echo "TimePredictor training completed: $MODEL_TYPE + $PREDICTOR_TYPE"
echo "Results saved to: $OUTPUT_DIR"
echo '--------------------------'

# Save experiment info
echo "Model: $MODEL_TYPE" > $OUTPUT_DIR/experiment_info.txt
echo "Predictor: $PREDICTOR_TYPE" >> $OUTPUT_DIR/experiment_info.txt
echo "Config: $MODEL_CONFIG" >> $OUTPUT_DIR/experiment_info.txt
echo "Start time: $(date)" >> $OUTPUT_DIR/experiment_info.txt
