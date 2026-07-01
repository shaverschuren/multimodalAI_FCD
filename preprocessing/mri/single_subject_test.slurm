#!/bin/bash
#SBATCH --job-name=mri_preproc_test
#SBATCH --partition=gpu_h100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/projects/prjs1713/MultimodalAI_Sjors/preprocessing/logs/%x_%j.out
#SBATCH --error=/projects/prjs1713/MultimodalAI_Sjors/preprocessing/logs/%x_%j.err

set -euo pipefail

echo "=== Job starting on $(hostname) at $(date) ==="

# Directory setup
PROJECT_ROOT=/projects/prjs1713/MultimodalAI_Sjors
LOG_DIR=$PROJECT_ROOT/preprocessing/logs
INPUT_DIR=$PROJECT_ROOT/preprocessing/data/INPUT
OUTPUT_DIR=$PROJECT_ROOT/preprocessing/data/OUTPUT

# Load environment
source /home/fgomezacebo/venvs/mri_venv/activate_env.sh

# Define test subject
SUBJ_ID="RESP0648"

# Scratch workspace
SCRATCH_DIR=${TMPDIR}
WORK_INPUT_DIR="$SCRATCH_DIR/input"
WORK_OUTPUT_DIR="$SCRATCH_DIR/output"
mkdir -p "$WORK_INPUT_DIR/$SUBJ_ID" "$WORK_OUTPUT_DIR/$SUBJ_ID"

echo "Using scratch directory: $SCRATCH_DIR"

# ---------------------------
# Input / Output paths (for testing now)
# ---------------------------
T1_PATH="$WORK_INPUT_DIR/$SUBJ_ID/${SUBJ_ID}-preop-T1w.nii.gz"
FLAIR_PATH="$WORK_INPUT_DIR/$SUBJ_ID/${SUBJ_ID}-preop-FLAIR.nii.gz"
POSTOP_PATH="$WORK_INPUT_DIR/$SUBJ_ID/${SUBJ_ID}-postop-T1w.nii.gz"
GT_PATH="$WORK_INPUT_DIR/$SUBJ_ID/${SUBJ_ID}_gt_mask.nii.gz"
WORK_OUTPUT_SUBJECT_DIR="$WORK_OUTPUT_DIR/$SUBJ_ID"

echo "Copying data..."
rsync -a "$INPUT_DIR/$SUBJ_ID/" "$WORK_INPUT_DIR/$SUBJ_ID/"

# ---------------------------

# Define cleanup function
cleanup () {
    echo "[$(date)] Cleaning up job $SLURM_JOB_ID ..."
    if [ -d "$SCRATCH_DIR" ]; then
        echo "Copying data from $WORK_OUTPUT_DIR to $OUTPUT_DIR"
        rsync -a "$WORK_OUTPUT_DIR/$SUBJ_ID/" "$OUTPUT_DIR/$SUBJ_ID/"
        rm -rf "$SCRATCH_DIR"
    fi
    echo "Cleanup complete."
}

# Register cleanup function to run on exit or cancellation
trap cleanup EXIT SIGINT SIGTERM

# Build command with optional arguments
CMD="python -u $PROJECT_ROOT/preprocessing/scripts/preprocess_mri.py \
    $SUBJ_ID \
    $T1_PATH \
    $FLAIR_PATH \
    $WORK_OUTPUT_DIR/$SUBJ_ID"

if [ -f "$POSTOP_PATH" ]; then
    CMD="$CMD --postop_path $POSTOP_PATH"
fi
if [ -f "$GT_PATH" ]; then
    CMD="$CMD --gt_path $GT_PATH"
fi

CMD="$CMD --keep_intermediates"

printf "\n[$SUBJ_ID] ---------- Invoking preprocess_mri.py ----------\n"
printf "$CMD\n"
eval $CMD

echo "Preprocessing finished."
