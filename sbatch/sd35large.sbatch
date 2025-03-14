#!/bin/bash
#SBATCH --job-name=sd35                         # Job name
#SBATCH --output=generate_images_%j.log
#SBATCH --error=generate_images_%j.err          #
#SBATCH -G a100:1                               # GPU
#SBATCH --mem=64G
#SBATCH -p general                              # Partition Name
#SBATCH -q public                               # QOS level
#SBATCH --time=0-4                              # Maximum runtime

# Load environment
module load mamba/latest
source activate vl

# Resolving Out of Memory Issue
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the Python script for processing CLIP scores
cd ..
python scripts/generate_images_sd35large.py
