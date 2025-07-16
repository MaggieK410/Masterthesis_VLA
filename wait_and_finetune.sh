#!/bin/bash

# === Config ===
REQUIRED_FREE_MEM_MB=24441   # Adjust if needed
GPU_IDS=(0 1)
CHECK_INTERVAL=30            # seconds
CMD="CUDA_VISIBLE_DEVICES=0,1 accelerate launch scripts/finetune_accelerate.py \
  --output_dir ./finetuned-model_trained_20000_pick_up_ball_larger_dataset \
  --batch_size 1 \
  --dataset_path ./finetune_data_pick_up_ball_small_larger_dataset/ \
  --embodiment_tag gr1 \
  --max_steps 20000 \
  --tune_visual"

# === Function to check if all specified GPUs are free enough ===
check_gpus_free() {
    local all_free=true
    for id in "${GPU_IDS[@]}"; do
        mem_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$((id + 1))p")
        if (( mem_free < REQUIRED_FREE_MEM_MB )); then
            echo "⛔ GPU $id only has $mem_free MB free (need $REQUIRED_FREE_MEM_MB)"
            all_free=false
        else
            echo "✅ GPU $id has $mem_free MB free"
        fi
    done
    $all_free && return 0 || return 1
}

# === Main wait loop ===
echo "🔁 Waiting for GPUs ${GPU_IDS[*]} to be free..."

while true; do
    if check_gpus_free; then
        echo "🚀 Launching job!"
        eval $CMD
        break
    else
        echo "⏳ Retrying in $CHECK_INTERVAL seconds..."
        sleep $CHECK_INTERVAL
    fi
done
