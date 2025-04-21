#!/bin/bash
#SBATCH --job-name=marsbench_detection_test
#SBATCH --output=../outputs/sbatch/detection_test_%j.log
#SBATCH --error=../outputs/sbatch/detection_test_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH -p general
#SBATCH -q public
#SBATCH --time=1-0

# Load environment
module load mamba
source activate env_mb

# Resolving Out of Memory Issue
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to project root directory
cd ..

# Function to run test and handle errors
run_test() {
    echo "Running: $1"
    eval "$1"
    if [ $? -eq 0 ]; then
        echo " Test passed: $1"
    else
        echo " Test failed: $1"
    fi
    echo "----------------------------------------"
}

echo "Starting MarsBench Detection Testing Suite..."

# === Detection model tests ===
datasets=("cone_quest" "mars_dust_devil")
models=("fasterrcnn" "retinanet" "ssd" "efficientdet" "detr")

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        run_test "python -m marsbench.main task=detection data_name=$dataset model_name=$model training.trainer.max_epochs=2 logger.wandb.enabled=False logger.tensorboard.enabled=false logger.mlflow.enabled=False logger.csv.enabled=False"
    done
done

# Print summary
echo "MarsBench Detection Testing Suite Completed!"
echo "Check the logs above for individual test results."
