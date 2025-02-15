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
```
  - `ml`: Executes ML pipeline.
  - `target`: Target column for regression or classification.
  - `tune`: (Optional) Perform hyperparameter tuning.
  - `drop`: (Optional) Comma-separated list of columns to drop.
  - `model`: (Optional) Model name (logistic, svm, randomforest, etc.).
  - `save-model`: (Optional) Path to save the trained pipeline.

For **unsupervised** clustering:

```bash
python universal_dataset_ml_pipeline.py <file_path> --ml [--drop COL1,COL2,...] --cluster-model <kmeans|dbscan>
```
  - `cluster-model`: Chooses clustering algorithm (default kmeans).

### 2) Prediction Mode (Load a Pre-Trained Model)
```bash
python lab_dataset_ml_pipeline.py <file_path> --load-model MODEL.pkl
```
  - `load-model`: Path to a pre-trained pipeline (pickled).
    Note: The dataset columns must match what the model expects (except the target column).

## Example Commands
### Regression with Tuning & Dropping Columns:
```bash
python universal_dataset_ml_pipeline.py data/california_housing.csv --ml --target median_house_value --tune --drop ocean_proximity --model ridge --save-model cali_model.pkl
```

### Clustering (Unsupervised):
```bash
python universal_dataset_ml_pipeline.py data/sample_data.csv --ml --drop id --cluster-model kmeans
```

### Load Model & Predict:
```bash
python universal_dataset_ml_pipeline.py data/new_data.csv --load-model cali_model.pkl
```

## Outputs
Each run creates a timestamped folder (e.g., analysis_20230915_143210) containing:

  - Logs (e.g., analysis.log).
  - Dataset Summaries (dataset_summary.txt).
  - Plots (histograms, correlation heatmaps, confusion matrices, etc.).
  - SHAP Summary (shap_summary.png).
  - Predictions (when loading a model for inference)

## Contributing
Feel free to submit pull requests or open issues if you want to improve the pipeline, add new features, or enhance its flexibility.
