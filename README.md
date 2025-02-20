# UniML Explorer

A Python-based solution for end-to-end data ingestion, exploratory data analysis (EDA), and machine learning (ML) pipelines. Includes a Streamlit GUI (`app.py`) for easy file uploads, column-dropping, supervised/unsupervised ML, and visualization.

## Features

- **Ingestion**: Accepts CSV, Excel, or JSON.
- **Exploratory Analysis**: Logs dataset summary, missing values, and statistical info.
- **Optional Plots**: Histograms, correlation heatmaps.
- **ML Pipelines**:
  - **Supervised** (Classification or Regression) with optional hyperparameter tuning.
  - **Unsupervised** (Clustering) with PCA-based visualization.
- **Streamlit GUI**: Non-technical users can upload data, select columns to drop, specify target column, and run analysis with a single click.

## Requirements
See [requirements.txt](requirements.txt) for the full list. Install via:
```bash
pip install -r requirements.txt
```
 
## Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/wenhan99/universal-data-analyzer.git
   cd universal-data-analyzer
   ```

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
python universal_dataset_ml_pipeline.py <file_path> --load-model MODEL.pkl
```
  - `load-model`: Path to a pre-trained pipeline (pickled).
    Note: The dataset columns must match what the model expects (except the target column).

## Example Commands
### Regression with Tuning & Dropping Columns:
```bash
python universal_dataset_ml_pipeline.py data/sample_data.csv --ml --target outcome --tune --drop id --model randomforest --save-model model_1.pkl
```

### Clustering (Unsupervised):
```bash
python universal_dataset_ml_pipeline.py data/sample_data.csv --ml --drop id --cluster-model kmeans
```

### Load Model & Predict:
```bash
python universal_dataset_ml_pipeline.py data/new_data.csv --load-model model_1.pkl
```

### Run the Streamlit App
```bash
streamlit run app.py
```
Open your browser to the URL shown (e.g., HTTP://localhost:8501), then interact with the app.
  - Upload your data (CSV, Excel, or JSON)
  - (Optional) Choose columns to drop.
  - (Optional) Check "Run ML Pipeline" and select a target column for supervised or leave blank for unsupervised.
  - Click *Run Analysis* to generate logs, EDA, plots, and ML results.

## Outputs
Each run creates a timestamped folder (e.g., analysis_20230915_143210) containing:

  - Logs (e.g., analysis.log).
  - Dataset Summaries (dataset_summary.txt).
  - Plots (histograms, correlation heatmaps, confusion matrices, etc.).
  - SHAP Summary (shap_summary.png).
  - Predictions (when loading a model for inference)

## File Structure
  - `app.py`: Streamlit GUI for user interaction.
  - `universal_dataset_ml_pipeline.py`: Core pipeline logic for data cleaning, imputation, EDA logs, supervised/unsupervised ML, and SHAP interpretability.
  - `requirements.txt`: Python dependencies.

## Contributing
Feel free to submit pull requests or open issues if you want to improve the pipeline, add new features, or enhance its flexibility.
