# MarsBench

MarsBench is an initiative to develop a comprehensive benchmarking platform for Mars science datasets, similar to GeoBench. It aims to organize, evaluate, and benchmark datasets related to Mars research for various machine learning (ML) tasks.

## Overview

**Objective:** To create a centralized resource for Mars science datasets, providing ML-readiness, evaluations, and a leaderboard for state-of-the-art (SOTA) results.

**Tasks:**
- **Dataset Curation:** Collect and organize Mars datasets. Ensure they are ML-ready with proper splits and categorization (e.g., classification, segmentation, object detection).
- **Evaluation:** Test datasets using popular models (e.g., U-Net, SegFormer) and document performance.
- **Website and Leaderboard:** Develop a website for the MarsBench platform, featuring a leaderboard similar to GLUE or Hugging Face.

## Installation

To install MarsBench and its dependencies:

```bash
# Install the package with core dependencies
pip install -e .

# Install with development dependencies (for testing, linting, etc.)
pip install -e ".[dev]"
```

## Tasks and Updates

- **Task 0:** Complete Mars Datasets Documentation. Ensure all datasets have accurate information about train, validation, and test splits, and check for existing SOTA results.
- **Task 1:** Search for Additional Datasets. Explore platforms like Zenodo, Kaggle, Radiant ML, and DataVerse for new datasets.
- **Task 2:** Group Datasets and Prepare Pipelines:
  - **Classification:** Use models like ResNet, ViT, YOLO, MobileNet, Swin Transformer.
  - **Segmentation:** Evaluate using models like U-Net, DeepLab.

## Data Storage

Datasets are stored in the following directory:

```  /data/hkerner/MarsBench ```


## MarsBench: Classification Task Workflow

**Objective:** Evaluate various models on Mars Science datasets for classification tasks.

1. **Project Setup**
   - **Directory Structure:** Store datasets and results in `/data/hkerner/MarsBench`.
   - **Libraries:** Use essential libraries like `torch`, `wandb`, and `cv2` for data processing and model training.

2. **Data Preparation**
   - **Custom Dataset Classes:** Implement dataset classes inheriting from `CustomDataset` to handle various Mars datasets:
     - `MarsDataset`: For Mars Image Content Classification.
     - `DoMars16k`: For the DoMars16k dataset.
     - `DeepMars_Landmark`: For DeepMars Landmark dataset.
     - `DeepMars_Surface`: For DeepMars Surface dataset.
     - `MartianFrostDataset`: For Martian Frost dataset, including handling JSON labels.

3. **Dataset Handling**
   - **Loading Data:** Read image paths and labels from text files or directory structures.
   - **Data Splitting:** Organize data into training, validation, and test sets based on provided splits or create new splits if necessary.

4. **Model Training**
   - **Models:** Use models like ResNet50, VIT-16, and Swin-Transformer.
   - **Training Process:**
     - **Initialization:** Set up model, loss function, and optimizer.
     - **Training Loop:** For each epoch, iterate through batches of training data, perform forward and backward passes, and update model weights.
     - **Validation:** Evaluate model performance on the validation set to track accuracy and loss.

5. **Metrics and Logging**
   - **Metrics Tracking:** Record metrics such as training loss, validation loss, accuracy, precision, recall, and F1-score.
   - **Wandb Integration:** Log training and validation metrics using Weights & Biases (Wandb) for visualization and analysis.

6. **Results Evaluation**
   - **Initial Results:** Review performance metrics from initial runs. Fine-tune models if necessary.
   - **Final Results:** Aggregate results and compare model performance on different datasets.

7. **Documentation and Updates**
   - **Results Table:** Create a results table documenting metrics for different models and datasets.
   - **Future Work:** Incorporate additional models or datasets and refine existing pipelines based on performance.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.
