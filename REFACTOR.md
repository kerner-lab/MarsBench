# Refactor

How I see this project coming together.

```bash
├── README.md
├── setup.py / setup.cfg
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/                     # (Optional) Sample data or data download scripts
│
├── configs/                  # Configuration files managed by Hydra
│   ├── config.yaml           # Default configuration
│   ├── data/
|   |_
│   │   ├── dataset1.yaml
│   │   └── dataset2.yaml
│   └── model/
│       ├── model1.yaml
│       └── model2.yaml
│
├── src/
│   ├── __init__.py
│   │
├── data/
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── domars16k_dataset.py
│   │   ├── deepmars_classification_dataset.py
│   │   ├── martian_frost_dataset.py
│   │   ├── dusty_vs_nondusty_dataset.py
│   │   └── ... (other classification datasets)
│   ├── segmentation/
│   │   ├── __init__.py
│   │   ├── ai4mars_dataset.py
│   │   ├── s5mars_dataset.py
│   │   ├── conequest_dataset.py
│   │   └── ... (other segmentation datasets)
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── cone_detection_dataset.py
│   │   ├── novelty_detection_dataset.py
│   │   └── ... (other detection datasets)
│   └── base_dataset.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_base.py     # Base model class/interface
│   │   ├── model1.py         # Model implementations
│   │   └── model2.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py        # Training loop and logic
│   │   ├── evaluator.py      # Evaluation and validation logic
│   │   └── metrics.py        # Metrics computation
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py        # Logging utilities
│   │   ├── config.py         # Configuration utilities
│   │   └── misc.py           # Other utility functions
│   │
│   └── main.py               # Entry point for running experiments
│
├── scripts/
│   ├── download_data.sh      # Scripts for data downloading and preprocessing
│   ├── run_experiments.sh    # Bash scripts to run multiple experiments
│   └── setup_env.sh          # Environment setup scripts
│
└── tests/
    ├── __init__.py
    ├── test_data.py          # Unit tests for data modules
    ├── test_models.py        # Unit tests for model modules
    ├── test_training.py      # Unit tests for training modules
    └── test_utils.py         # Unit tests for utility modules
```

