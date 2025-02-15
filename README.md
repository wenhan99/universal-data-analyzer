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
