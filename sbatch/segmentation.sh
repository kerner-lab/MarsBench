#!/bin/bash

parallel ::: \
"python -m marsbench.main task=segmentation data_name=cone_quest model_name=unet training.trainer.max_epochs=100 training.criterion.name=cross_entropy logger.wandb.name=unet_cross_entropy" \
"python -m marsbench.main task=segmentation data_name=cone_quest model_name=unet training.trainer.max_epochs=100 training.criterion.name=dice logger.wandb.name=unet_dice" \
"python -m marsbench.main task=segmentation data_name=cone_quest model_name=unet training.trainer.max_epochs=100 training.criterion.name=generalized_dice logger.wandb.name=unet_generalized_dice" \
"python -m marsbench.main task=segmentation data_name=cone_quest model_name=unet training.trainer.max_epochs=100 training.criterion.name=combined logger.wandb.name=unet_combined"
