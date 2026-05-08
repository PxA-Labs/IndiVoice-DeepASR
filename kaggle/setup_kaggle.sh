#!/bin/bash
# IndiVoice-DeepASR: Kaggle Setup Script
# Use this to prepare the Kaggle environment for training.

echo "[LOG] Starting Kaggle Environment Setup..."

# 1. Path Configuration
KAGGLE_INPUT="/kaggle/input"
REPO_DIR="/kaggle/working/IndiVoice-DeepASR"
REPO_URL="https://github.com/purvanshjoshi/IndiVoice-DeepASR.git"

# 2. Setup Working Directory
echo "[LOG] Setting up repository..."
mkdir -p /kaggle/working
cd /kaggle/working

# Robust Nesting Protection: While we are inside a directory named IndiVoice-DeepASR, move up
while [[ $(basename $(pwd)) == "IndiVoice-DeepASR" ]]; do
    cd ..
done

if [ ! -d "IndiVoice-DeepASR" ]; then
    git clone $REPO_URL
else
    cd IndiVoice-DeepASR
    git fetch --all
    git reset --hard origin/master
fi
cd "$REPO_DIR"

# 3. Create Infrastructure
echo "[LOG] Creating directory structure..."
mkdir -p data/processed
mkdir -p models/whisper-indian-lora

# 4. Link Kaggle Input Datasets/Checkpoints
# Smart detection of 'indivoice-resumption' or other provided datasets
echo "[LOG] Searching for input data..."
find $KAGGLE_INPUT -name "svarah_manifest.json" -exec ln -sf {} data/processed/svarah_manifest.json \;

# Resumption Check: If the user provided a 'checkpoint-*' folder in any dataset, copy it
CHECKPOINTS=$(find $KAGGLE_INPUT -name "checkpoint-*" -type d)
if [ ! -z "$CHECKPOINTS" ]; then
    echo "[RESUMING] Found saved checkpoints. Preparing for resumption..."
    for cp_path in $CHECKPOINTS; do
        cp -rn "$cp_path" models/whisper-indian-lora/
    done
    echo "[SUCCESS] Successfully restored $(ls -1 models/whisper-indian-lora/ | grep checkpoint | wc -l) checkpoints."
fi

# 5. Automated Accelerate Config (Dual-T4 Optimized)
echo "[LOG] Configuring Multi-GPU Accelerator..."
mkdir -p ~/.cache/huggingface/accelerate
cat <<EOF > ~/.cache/huggingface/accelerate/default_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

# 6. Install Dependencies
echo "[LOG] Installing optimized dependencies..."
pip install -r requirements.txt --quiet
pip install bitsandbytes --quiet 

# 7. Auto-Recovery: Download Data/Manifest if missing
SHOULD_RECOVER=false
if [[ ! -f "data/processed/svarah_manifest.json" ]]; then
    echo "[LOG] Manifest not found. Triggering download..."
    SHOULD_RECOVER=true
elif [[ ! -d "data/processed/svarah" ]] || [[ -z "$(ls -A data/processed/svarah 2>/dev/null)" ]]; then
    echo "[LOG] Audio data not found or empty. Triggering download..."
    SHOULD_RECOVER=true
fi

if [ "$SHOULD_RECOVER" = true ]; then
    echo "[WARNING] Data incomplete! Launching Auto-Recovery..."
    
    # Check for Hugging Face Token (Gated Dataset Requirement)
    if [[ -z "$HF_TOKEN" ]]; then
        echo "[ERROR] WARNING: HF_TOKEN not found in environment."
        echo "   Svarah is a GATED dataset. To fix this:"
        echo "   1. Add 'HF_TOKEN' to your Kaggle Secrets (Add-ons -> Secrets)."
        echo "   2. Accept the terms at: https://huggingface.co/datasets/ai4bharat/Svarah"
        echo "   Continuing attempt anyway..."
    fi

    mkdir -p data/processed/svarah
    
    # Manifest Liberation: If the manifest is a symlink (to Read-Only input), delete it
    # so we can write a fresh local version if needed.
    if [ -L "data/processed/svarah_manifest.json" ]; then
        echo "[LOG] Liberating manifest from read-only symlink..."
        rm "data/processed/svarah_manifest.json"
    fi

    python src/preprocess.py \
        --hf_dataset ai4bharat/Svarah \
        --output_dir data/processed/svarah \
        --manifest_path data/processed/svarah_manifest.json \
        --target_sr 16000
    
    if [[ ! -f "data/processed/svarah_manifest.json" ]]; then
        echo "[ERROR] Auto-Recovery failed! Manifest was not created."
    else
        echo "[SUCCESS] Auto-Recovery Complete! Data and manifest ready."
    fi
fi

echo "[LOG] Kaggle Setup Complete! Repository is ready for Dual-T4 training."
echo "Launch command: accelerate launch src/train.py"
