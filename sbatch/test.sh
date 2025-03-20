#!/bin/bash
#SBATCH --job-name=marsbench_test
#SBATCH --output=../outputs/sbatch/test_%j.log
#SBATCH --error=../outputs/sbatch/test_%j.err
#SBATCH --gres=gpu:a30:2
#SBATCH --cpus-per-task=4
#SBATCH -p general
#SBATCH -q public
#SBATCH --time=1-0

# Load environment
module load mamba/latest
source activate vl

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

echo "Starting MarsBench Testing Suite..."

# 1. Basic Model Training Test
run_test "python -m marsbench.main mode=train"

# 2. Model Testing
run_test "python -m marsbench.main mode=test"

# 3. Model Prediction
run_test "python -m marsbench.main mode=predict"

# 4. Multi-GPU Training Test
run_test "python -m marsbench.main mode=train training.trainer.devices=2"

# 5. Different Model Architecture Test
run_test "python -m marsbench.main mode=train model_name=resnet50"

# Print summary
echo "MarsBench Testing Suite Completed!"
echo "Check the logs above for individual test results."
