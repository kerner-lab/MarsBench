#!/bin/bash
#SBATCH --output=../outputs/sbatch/test.log
#SBATCH --error=../outputs/sbatch/test.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH -p general
#SBATCH -q public
#SBATCH --time=1-0

# Load environment
module load mamba/latest
source activate env_mb

# Change to project root directory
cd ..

# -------------- DETECTION --------------
echo "Running: conequest_detection"
python -m marsbench.main task=detection data_name=conequest_detection model_name=fasterrcnn training.trainer.max_epochs=2

echo "Running: dust_devil_detection"
python -m marsbench.main task=detection data_name=dust_devil_detection model_name=fasterrcnn training.trainer.max_epochs=2


# -------------- SEGMENTATION --------------
echo "Running: conequest_segmentation"
python -m marsbench.main task=segmentation data_name=conequest_segmentation model_name=unet training.trainer.max_epochs=2

echo "Running: boulder_segmentation"
python -m marsbench.main task=segmentation data_name=boulder_segmentation model_name=deeplab training.trainer.max_epochs=2

echo "Running: mars_seg_mer"
python -m marsbench.main task=segmentation data_name=mars_seg_mer model_name=unet training.trainer.max_epochs=2

echo "Running: mars_seg_msl"
# python -m marsbench.main task=segmentation data_name=mars_seg_msl model_name=unet training.trainer.max_epochs=2

echo "Running: mmls_segmentation"
python -m marsbench.main task=segmentation data_name=mmls model_name=deeplab training.trainer.max_epochs=2

echo "Running: s5mars_segmentation"
python -m marsbench.main task=segmentation data_name=s5mars model_name=unet training.trainer.max_epochs=2
