# Universal Data Analyzer

A Python-based command-line tool for comprehensive data analysis, machine learning (ML) pipelines, and result visualization. It supports various data formats (CSV, Excel, JSON), advanced imputation, optional hyperparameter tuning, clustering, SHAP feature importance, and more.

## Key Features

- **Data Loading & EDA**: Reads CSV/Excel/JSON files and provides descriptive statistics, missing-value summaries, correlation heatmaps, and histogram plots.
- **Flexible ML Pipelines**:
  - **Supervised** (Classification or Regression) with optional hyperparameter tuning.
  - **Unsupervised** (Clustering) with PCA-based visualization.
- **Advanced Imputation**: IterativeImputer (or SimpleImputer) to handle missing data effectively.
- **Column Dropping**: Optionally remove non-informative columns.
- **SHAP Interpretability**: Generates SHAP plots to explain model predictions.
- **Unique Output Folder**: Each run creates a timestamped folder for logs, plots, and outputs.

## Requirements

- **Python** 3.7+ recommended
- Core packages:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `shap`
  - `pickle` (part of Python’s standard library, used for model serialization)
  - `argparse` (standard library)
  - `logging` (standard library)
 
## Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/<YourUsername>/universal-data-analyzer.git

## Usage

You can run the script in **training mode** or **prediction mode**.

### 1) Training Mode

For **supervised** learning:
```bash
python universal_dataset_ml_pipeline.py <file_path> --ml --target <TARGET_COLUMN> [--tune] [--drop COL1,COL2,...] [--model MODEL_NAME] [--save-model MODEL.pkl]
--ml: Executes ML pipeline.
--target: Target column for regression or classification.
--tune: (Optional) Perform hyperparameter tuning.
--drop: (Optional) Comma-separated list of columns to drop.
--model: (Optional) Model name (logistic, svm, randomforest, etc.).
--save-model: (Optional) Path to save the trained pipeline.

For unsupervised clustering:

```bash
python universal_dataset_ml_pipeline.py <file_path> --ml [--drop COL1,COL2,...] --cluster-model <kmeans|dbscan>
--cluster-model: Chooses clustering algorithm (default kmeans).
