#!/bin/bash

module load mamba/latest
source deactivate
source activate vl


python -m marsbench.main data_name=frost_classification training.batch_size=256 training.trainer.max_epochs=3 || true
python -m marsbench.main data_name=multi_label_mer training.batch_size=64 training.trainer.max_epochs=3 || true
python -m marsbench.main training.batch_size=64 training.trainer.max_epochs=3 || true
