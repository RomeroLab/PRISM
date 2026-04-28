#!/bin/bash
# ==============================================================================
# Linear Probing Experiments for Residue-Level GL/NGL Classification
# ==============================================================================
#
# This script runs linear probing experiments for all 5 PLM models IN PARALLEL:
#   1. ESM-2 35M (480 dim)
#   2. ESM-2 650M (1280 dim)
#   3. AbLang2 (480 dim)
#   4. AntiBERTy (512 dim)
#   5. Sapiens (128 dim) - Merged from separate H/L files
#
# All models run simultaneously on the same GPU (linear probes are small).
#
# Prerequisites:
#   - Run extract_residue_embeddings.py first to generate embedding files
#   - Each model produces separate files: train_linear_{model}.pkl, etc.
#   - For Sapiens: run merge_sapiens_embeddings.py first (or use --merge-sapiens flag)
#
# Usage:
#   bash run_linear_probes.sh           # Run all models in parallel
#   bash run_linear_probes.sh esm2_35m  # Run specific model only
#   bash run_linear_probes.sh --merge-sapiens  # Merge Sapiens files first, then run all
#
# ==============================================================================

set -e  # Exit on error

# Configuration
DATA_DIR="data/unpaired_OAS/linear_probe_data"
LOG_BASE_DIR="runs/linear_probe"
BATCH_SIZE=64
LR=1e-3
EPOCHS=50
SEED=42
NUM_WORKERS=4

# Conda environment
CONDA_ENV="devant"

# ==============================================================================
# Model Configurations
# ==============================================================================
# Format: MODEL_NAME:EMBEDDING_DIM:EMBEDDING_COL_PREFIX

declare -A MODELS=(
    ["esm2_35m"]="480:embed_esm2_35m"
    ["esm2_650m"]="1280:embed_esm2_650m"
    ["ablang2"]="480:embed_ablang2"
    ["antiberty"]="512:embed_antiberty"
    ["sapiens"]="128:embed_sapiens"
)

# ==============================================================================
# Helper Functions
# ==============================================================================

run_probe() {
    local model_name=$1
    local input_dim=$2
    local embed_prefix=$3

    local train_path="${DATA_DIR}/train_linear_${model_name}.pkl"
    local val_path="${DATA_DIR}/val_linear_${model_name}.pkl"
    local test_path="${DATA_DIR}/test_linear_${model_name}.pkl"
    local log_dir="${LOG_BASE_DIR}/${model_name}"

    # Check if embedding files exist
    if [[ ! -f "$train_path" ]]; then
        echo "[${model_name}] WARNING: Train file not found: $train_path"
        return 1
    fi

    if [[ ! -f "$val_path" ]]; then
        echo "[${model_name}] WARNING: Val file not found: $val_path"
        return 1
    fi

    # Build command
    local cmd="python train_probe.py"
    cmd+=" --train_path ${train_path}"
    cmd+=" --val_path ${val_path}"
    if [[ -f "$test_path" ]]; then
        cmd+=" --test_path ${test_path}"
    fi
    cmd+=" --embedding_col_prefix ${embed_prefix}"
    cmd+=" --input_dim ${input_dim}"
    cmd+=" --batch_size ${BATCH_SIZE}"
    cmd+=" --lr ${LR}"
    cmd+=" --epochs ${EPOCHS}"
    cmd+=" --seed ${SEED}"
    cmd+=" --num_workers ${NUM_WORKERS}"
    cmd+=" --log_dir ${log_dir}"

    echo "[${model_name}] Starting... (log: ${log_dir}/${model_name}.log)"

    # Run with conda, redirect output to log file
    mkdir -p "${log_dir}"
    conda run -n ${CONDA_ENV} ${cmd} > "${log_dir}/${model_name}.log" 2>&1

    echo "[${model_name}] Completed! Best model: ${log_dir}/best_model.pt"
}

# ==============================================================================
# Sapiens Merge Function
# ==============================================================================

merge_sapiens() {
    echo ""
    echo "=============================================================="
    echo "Merging Sapiens H/L embedding files..."
    echo "=============================================================="

    conda run -n ${CONDA_ENV} python merge_sapiens_embeddings.py --data_dir ${DATA_DIR}

    echo "Sapiens merge complete!"
    echo ""
}

# ==============================================================================
# Main
# ==============================================================================

# Parse command line arguments
MERGE_SAPIENS=false
TARGET_MODEL="all"

for arg in "$@"; do
    case $arg in
        --merge-sapiens)
            MERGE_SAPIENS=true
            shift
            ;;
        *)
            TARGET_MODEL="$arg"
            ;;
    esac
done

echo "=============================================================="
echo "Linear Probing Experiments (PARALLEL EXECUTION)"
echo "=============================================================="
echo "Data directory: ${DATA_DIR}"
echo "Log directory: ${LOG_BASE_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LR}"
echo "Epochs: ${EPOCHS}"
echo "Seed: ${SEED}"
echo "Target model: ${TARGET_MODEL}"
echo "Merge Sapiens: ${MERGE_SAPIENS}"
echo ""

# Merge Sapiens files if requested
if [[ "$MERGE_SAPIENS" == "true" ]]; then
    merge_sapiens
fi

if [[ "$TARGET_MODEL" == "all" ]]; then
    echo "Launching all models in parallel..."
    echo ""

    # Array to store PIDs
    declare -a PIDS=()
    declare -a MODEL_NAMES=()

    # Launch all models in parallel
    for model_name in "${!MODELS[@]}"; do
        config=${MODELS[$model_name]}
        input_dim=$(echo $config | cut -d: -f1)
        embed_prefix=$(echo $config | cut -d: -f2)

        # Run in background
        run_probe "$model_name" "$input_dim" "$embed_prefix" &
        PIDS+=($!)
        MODEL_NAMES+=("$model_name")
    done

    echo ""
    echo "All models launched! PIDs: ${PIDS[@]}"
    echo "Waiting for all jobs to complete..."
    echo ""

    # Wait for all background jobs and track results
    declare -a FAILED=()
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        model=${MODEL_NAMES[$i]}
        if wait $pid; then
            echo "[${model}] SUCCESS"
        else
            echo "[${model}] FAILED (check log for details)"
            FAILED+=("$model")
        fi
    done

    echo ""
    echo "=============================================================="
    echo "All experiments completed!"
    echo "=============================================================="

    if [[ ${#FAILED[@]} -gt 0 ]]; then
        echo ""
        echo "FAILED MODELS: ${FAILED[@]}"
        echo "Check logs in ${LOG_BASE_DIR}/<model>/<model>.log"
    fi

else
    # Run specific model (not in background)
    if [[ -v MODELS[$TARGET_MODEL] ]]; then
        config=${MODELS[$TARGET_MODEL]}
        input_dim=$(echo $config | cut -d: -f1)
        embed_prefix=$(echo $config | cut -d: -f2)

        run_probe "$TARGET_MODEL" "$input_dim" "$embed_prefix"
    else
        echo "ERROR: Unknown model: ${TARGET_MODEL}"
        echo "Available models: ${!MODELS[@]}"
        exit 1
    fi
fi

echo ""
echo "View TensorBoard logs:"
echo "  tensorboard --logdir ${LOG_BASE_DIR}"
echo ""
echo "View individual training logs:"
for model_name in "${!MODELS[@]}"; do
    echo "  tail -f ${LOG_BASE_DIR}/${model_name}/${model_name}.log"
done
echo ""


# ==============================================================================
# Individual Commands (for reference / manual execution)
# ==============================================================================
#
# ESM-2 35M:
# conda run -n devant python train_probe.py \
#     --train_path data/unpaired_OAS/linear_probe_data/train_linear_esm2_35m.pkl \
#     --val_path data/unpaired_OAS/linear_probe_data/val_linear_esm2_35m.pkl \
#     --test_path data/unpaired_OAS/linear_probe_data/test_linear_esm2_35m.pkl \
#     --embedding_col_prefix embed_esm2_35m \
#     --input_dim 480 \
#     --batch_size 64 \
#     --lr 1e-3 \
#     --epochs 50 \
#     --log_dir runs/linear_probe/esm2_35m
#
# ESM-2 650M:
# conda run -n devant python train_probe.py \
#     --train_path data/unpaired_OAS/linear_probe_data/train_linear_esm2_650m.pkl \
#     --val_path data/unpaired_OAS/linear_probe_data/val_linear_esm2_650m.pkl \
#     --test_path data/unpaired_OAS/linear_probe_data/test_linear_esm2_650m.pkl \
#     --embedding_col_prefix embed_esm2_650m \
#     --input_dim 1280 \
#     --batch_size 64 \
#     --lr 1e-3 \
#     --epochs 50 \
#     --log_dir runs/linear_probe/esm2_650m
#
# AbLang2:
# conda run -n devant python train_probe.py \
#     --train_path data/unpaired_OAS/linear_probe_data/train_linear_ablang2.pkl \
#     --val_path data/unpaired_OAS/linear_probe_data/val_linear_ablang2.pkl \
#     --test_path data/unpaired_OAS/linear_probe_data/test_linear_ablang2.pkl \
#     --embedding_col_prefix embed_ablang2 \
#     --input_dim 480 \
#     --batch_size 64 \
#     --lr 1e-3 \
#     --epochs 50 \
#     --log_dir runs/linear_probe/ablang2
#
# AntiBERTy:
# conda run -n devant python train_probe.py \
#     --train_path data/unpaired_OAS/linear_probe_data/train_linear_antiberty.pkl \
#     --val_path data/unpaired_OAS/linear_probe_data/val_linear_antiberty.pkl \
#     --test_path data/unpaired_OAS/linear_probe_data/test_linear_antiberty.pkl \
#     --embedding_col_prefix embed_antiberty \
#     --input_dim 512 \
#     --batch_size 64 \
#     --lr 1e-3 \
#     --epochs 50 \
#     --log_dir runs/linear_probe/antiberty
#
# Sapiens (requires merging H/L files first):
# conda run -n devant python merge_sapiens_embeddings.py --data_dir data/unpaired_OAS/linear_probe_data
# conda run -n devant python train_probe.py \
#     --train_path data/unpaired_OAS/linear_probe_data/train_linear_sapiens.pkl \
#     --val_path data/unpaired_OAS/linear_probe_data/val_linear_sapiens.pkl \
#     --test_path data/unpaired_OAS/linear_probe_data/test_linear_sapiens.pkl \
#     --embedding_col_prefix embed_sapiens \
#     --input_dim 128 \
#     --batch_size 64 \
#     --lr 1e-3 \
#     --epochs 50 \
#     --log_dir runs/linear_probe/sapiens
# ==============================================================================
