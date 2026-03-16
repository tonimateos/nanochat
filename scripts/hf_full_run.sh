#!/bin/bash

# Configuration
export OMP_NUM_THREADS=1

# Determine number of GPUs for multi-GPU setups (like 4xL40S)
N_GPUS=$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 1)")

if [ "$N_GPUS" -gt 1 ]; then
    CMD_PREFIX="torchrun --standalone --nproc_per_node=$N_GPUS"
    echo "Detected $N_GPUS GPUs. Using torchrun."
else
    CMD_PREFIX="python"
    echo "Detected 1 GPU. Using standard python execution."
fi

# Optional: Add --fp8 flag if you select Hopper/Ada GPUs (e.g., L4, L40S, H100)
# EXTRA_ARGS="--fp8" 
EXTRA_ARGS=""

echo "========================================="
echo "1. Tokenizer & Dataset Preparation"
echo "========================================="
# Download the first ~2B characters of pretraining dataset
python -m nanochat.dataset -n 8

# Immediately also kick off downloading more shards in the background while tokenizer trains
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!

# Train the tokenizer
python -m scripts.tok_train

# Evaluate the tokenizer
python -m scripts.tok_eval

echo "Waiting for background dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

echo "========================================="
echo "2. Base Model Pretraining (from scratch)"
echo "========================================="
# Train the base model (d24 is equivalent to GPT-2 size)
$CMD_PREFIX -m scripts.base_train --depth=24 --target-param-data-ratio=9.5 --device-batch-size=16 $EXTRA_ARGS

echo "========================================="
echo "3. Supervised Fine-Tuning (SFT)"
echo "========================================="
# Download synthetic identity conversations for persona
NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-"$HOME/.cache/nanochat"}
mkdir -p $NANOCHAT_BASE_DIR
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# Run SFT (This script will automatically upload checkpoints to HF_REPO and pause HF_SPACE when finished)
$CMD_PREFIX -m scripts.chat_sft --device-batch-size=16

echo "========================================="
echo "Full Pipeline Complete!"
echo "========================================="
