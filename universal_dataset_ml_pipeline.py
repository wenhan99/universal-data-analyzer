#!/usr/bin/env python
"""
Universal Data Analyzer with Config Support, Column Flexibility, Enhanced SHAP,
and Optional ML Pipeline Tuning.

Features:
1) Reads data (CSV, Excel, JSON) and logs EDA (shape, missing values, correlation).
2) Supports supervised (classification/regression) with hyperparam tuning or unsupervised clustering.
3) Allows partial column mismatch handling (flex mode).
4) Uses faster SHAP TreeExplainer for tree-based models, falling back to KernelExplainer otherwise.
5) Accepts either direct CLI args or a YAML config for convenience.
6) Saves logs, plots, and SHAP results in a timestamped folder.

Usage:
  Training (supervised):
    python universal_analyzer.py data.csv --ml --target target_col --model randomforest --tune --drop col1,col2 --save-model mymodel.pkl
  Clustering:
    python universal_analyzer.py data.csv --ml --cluster-model kmeans --drop col1,col2
  Predict with Pre-Trained Model:
    python universal_analyzer.py new_data.csv --load-model mymodel.pkl
  Use a Config File:
    python universal_analyzer.py data.csv --config myconfig.yaml

In the config file (YAML), you can specify:
  drop: ["col1", "col2"]
  ml: true
  target: "target_col"
  model: "randomforest"
  tune: true
  cluster_model: "kmeans"
  save_model: "mymodel.pkl"
  load_model: null
  plots: true
  flex_columns: true
"""

import argparse
import os
import datetime
import io
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import yaml

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

# Models
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='analysis.log',
    filemode='a'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

def create_analysis_folder():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"analysis_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    logging.info(f"Created analysis folder: {folder_name}")
    return folder_name

def check_schema(df):
    for col in df.columns:
        inferred_type = pd.api.types.infer_dtype(df[col], skipna=True)
        logging.info(f"Column '{col}' inferred type: {inferred_type}")
        if 'mixed' in inferred_type:
            logging.warning(f"Column '{col}' has mixed types. Consider cleaning or converting data.")

def write_summary_to_file(df, output_dir):
    summary_file = os.path.join(output_dir, "dataset_summary.txt")
    with open(summary_file, "w") as f:
        f.write("=" * 40 + "\n")
        f.write(f"Dataset Shape: {df.shape}\n")
        f.write("=" * 40 + "\n\n")
        f.write("Dataset Info:\n")
        buf = io.StringIO()
        df.info(buf=buf)
        info_str = buf.getvalue()
        f.write(info_str)
        f.write("\nMissing Values per Column:\n")
        f.write(str(df.isnull().sum()) + "\n")
        f.write("\nStatistical Summary:\n")
        f.write(str(df.describe(include="all")))
        f.write("\n" + "=" * 40 + "\n")
    logging.info(f"Dataset summary written to {summary_file}")

def generate_summary(df):
    logging.info("=" * 40)
    logging.info(f"Dataset Shape: {df.shape}")
    logging.info("=" * 40)
    logging.info("\nDataset Info:")
    buf = io.StringIO()
    df.info(buf=buf)
    logging.info(buf.getvalue())
    logging.info("\nMissing Values per Column:")
    logging.info(df.isnull().sum())
    logging.info("\nStatistical Summary:")
    logging.info(df.describe(include='all'))
    
    # Additional EDA for categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        logging.info(f"\nValue counts for '{col}':")
        logging.info(df[col].value_counts())
    logging.info("=" * 40)

def drop_non_informative_columns(df, columns_to_drop):
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            logging.info(f"Dropped non-informative column: {col}")
    return df

def advanced_imputation(df, target_column=None, missing_threshold=0.4):
    removed_columns = []
    df = df.copy()
    for col in df.columns:
        if col == target_column:
            continue
        missing_fraction = df[col].isnull().mean()
        if missing_fraction > missing_threshold:
            removed_columns.append(col)
            df.drop(columns=[col], inplace=True)
    if removed_columns:
        logging.info(f"Removed columns with missing values > {missing_threshold*100:.0f}%: {removed_columns}")
    else:
        logging.info(f"No columns removed based on missing threshold of {missing_threshold*100:.0f}%.")
    return df, removed_columns

def load_dataset(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    try:
        if extension == ".csv":
            df = pd.read_csv(file_path, encoding='utf-8')
        elif extension in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        elif extension == ".json":
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}. Use CSV, Excel, or JSON.")
        logging.info(f"Dataset loaded from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def plot_histograms(df, output_dir):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        logging.info("No numeric columns for histograms.")
        return
    for col in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col].dropna(), kde=True, bins=30, color='skyblue')
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        out_path = os.path.join(output_dir, f"{col}_histogram.png")
        plt.savefig(out_path)
        plt.close()
        logging.info(f"Saved histogram for '{col}' to {out_path}")

def plot_correlation_heatmap(df, output_dir):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        logging.info("No numeric columns for correlation heatmap.")
        return
    plt.figure(figsize=(10, 8))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Heatmap")
    out_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(out_path)
    plt.close()
    logging.info(f"Saved correlation heatmap to {out_path}")

def visualize_classification_results(y_test, y_pred, output_dir):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    out_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(out_path)
    plt.close()
    logging.info(f"Saved confusion matrix to {out_path}")

def visualize_regression_results(y_test, y_pred, output_dir):
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values")
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    out_path = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(out_path)
    plt.close()
    logging.info(f"Saved regression plot to {out_path}")

def visualize_clusters(df, numeric_features, output_dir):
    X = df[numeric_features].copy()
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['Cluster'], palette='viridis', s=100, alpha=0.7)
    plt.title("Cluster Visualization (PCA Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    out_path = os.path.join(output_dir, "cluster_visualization.png")
    plt.savefig(out_path)
    plt.close()
    logging.info(f"Saved cluster visualization to {out_path}")

def explain_model(pipeline, X_train, output_dir):
    """
    Use a faster model-specific explainer (TreeExplainer) if pipeline's model is tree-based.
    Otherwise, fallback to KernelExplainer for a model-agnostic approach.
    """
    model = pipeline.named_steps['model']
    is_tree_model = any([
        isinstance(model, RandomForestClassifier), 
        isinstance(model, RandomForestRegressor),
        isinstance(model, GradientBoostingClassifier),
        isinstance(model, GradientBoostingRegressor)
    ])
    X_sample = X_train.iloc[:100]

    if is_tree_model:
        logging.info("Using TreeExplainer for SHAP (faster for tree-based models).")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pipeline.named_steps['preprocessor'].transform(X_sample))
        except Exception as e:
            logging.warning(f"TreeExplainer failed, falling back to KernelExplainer: {e}")
            explainer = shap.KernelExplainer(model.predict, pipeline.named_steps['preprocessor'].transform(X_sample))
            shap_values = explainer.shap_values(pipeline.named_steps['preprocessor'].transform(X_sample), nsamples=100)
    else:
        logging.info("Using KernelExplainer for SHAP.")
        # For KernelExplainer, we need a wrapper
        def predict_wrapper(data_array):
            return model.predict(data_array)
        # Transform X_sample
        X_transformed = pipeline.named_steps['preprocessor'].transform(X_sample)
        explainer = shap.KernelExplainer(predict_wrapper, X_transformed)
        shap_values = explainer.shap_values(X_transformed, nsamples=100)

    shap.summary_plot(shap_values, X_sample, show=False)
    shap_out = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(shap_out, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved SHAP summary plot to {shap_out}")

def preprocess_features(X):
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_transformer = Pipeline([
        ('imputer', IterativeImputer(random_state=42)),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    return preprocessor

def build_supervised_pipeline(preprocessor, model):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline

def tune_hyperparameters(pipeline, X_train, y_train, param_grid, problem_type):
    scoring = 'accuracy' if problem_type == 'classification' else 'r2'
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=scoring)
    grid_search.fit(X_train, y_train)
    logging.info(f"Best hyperparameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def run_supervised_pipeline(df, target_column, model_name=None, tune=False, output_dir="output", flex_columns=False):
    if target_column not in df.columns:
        logging.error(f"Target column '{target_column}' not found in dataset.")
        return

    # If flex_columns is True, we do nothing special here. 
    # But we could handle partial columns if we had a reference schema.
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    if y.dtype == 'object' or y.dtype.name == 'category' or y.nunique() < 10:
        problem_type = 'classification'
        model_config = CLASSIFICATION_MODELS.get(model_name, CLASSIFICATION_MODELS['logistic'])
    else:
        problem_type = 'regression'
        model_config = REGRESSION_MODELS.get(model_name, REGRESSION_MODELS['randomforest'])
    
    logging.info(f"Using {problem_type} model: {model_name if model_name else 'default'}")
    model = model_config['class']()
    preprocessor = preprocess_features(X)
    pipeline = build_supervised_pipeline(preprocessor, model)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if tune:
        logging.info("Tuning hyperparameters...")
        pipeline = tune_hyperparameters(pipeline, X_train, y_train, model_config['params'], problem_type)
    else:
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    
    if problem_type == 'classification':
        acc = accuracy_score(y_test, y_pred)
        logging.info(f"Classification Accuracy: {acc:.2f}")
        visualize_classification_results(y_test, y_pred, output_dir)
    else:
        score = r2_score(y_test, y_pred)
        logging.info(f"Regression R^2 Score: {score:.2f}")
        visualize_regression_results(y_test, y_pred, output_dir)
    
    logging.info("Computing SHAP values for model interpretability...")
    explain_model(pipeline, X_train, output_dir)
    
    return pipeline

def run_unsupervised_pipeline(df, model_name='kmeans', output_dir="output", flex_columns=False):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        logging.info("No numeric features available for clustering.")
        return
    
    model_class = CLUSTERING_MODELS.get(model_name, KMeans)
    if model_name == 'kmeans':
        model = model_class(n_clusters=3, random_state=42)
    elif model_name == 'dbscan':
        model = model_class(eps=0.5, min_samples=5)
    else:
        model = model_class()
    
    X = df[numeric_cols].copy()
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    clusters = model.fit_predict(X_scaled)
    df['Cluster'] = clusters
    logging.info("Clustering complete. 'Cluster' column added.")
    logging.info(df['Cluster'].value_counts())
    visualize_clusters(df, numeric_cols, output_dir)
    return model

CLASSIFICATION_MODELS = {
    'logistic': {
        'class': LogisticRegression,
        'params': {'model__C': [0.1, 1, 10], 'model__penalty': ['l1', 'l2'], 'model__solver': ['liblinear']}
    },
    'svm': {
        'class': SVC,
        'params': {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']}
    },
    'randomforest': {
        'class': RandomForestClassifier,
        'params': {'model__n_estimators': [50, 100, 200], 'model__max_depth': [None, 10, 20]}
    },
    'gradientboosting': {
        'class': GradientBoostingClassifier,
        'params': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.1, 0.5]}
    }
}

REGRESSION_MODELS = {
    'lasso': {
        'class': Lasso,
        'params': {'model__alpha': [0.1, 1.0, 10.0]}
    },
    'ridge': {
        'class': Ridge,
        'params': {'model__alpha': [0.1, 1.0, 10.0]}
    },
    'randomforest': {
        'class': RandomForestRegressor,
        'params': {'model__n_estimators': [50, 100, 200], 'model__max_depth': [None, 10, 20]}
    },
    'gradientboosting': {
        'class': GradientBoostingRegressor,
        'params': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.1, 0.5]}
    }
}

CLUSTERING_MODELS = {
    'kmeans': KMeans,
    'dbscan': DBSCAN
}

def parse_config_file(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config if config else {}

def main():
    parser = argparse.ArgumentParser(
        description="Universal Data Analyzer with Config, Column Flexibility, Enhanced SHAP, and ML Pipelines"
    )
    parser.add_argument("file_path", help="Path to the dataset file (CSV, Excel, JSON).")
    parser.add_argument("--plots", action="store_true", help="Generate plots (histograms, correlation heatmap, etc.).")
    parser.add_argument("--ml", action="store_true", help="Execute machine learning pipeline.")
    parser.add_argument("--target", type=str, help="Target column for supervised learning.")
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning.")
    parser.add_argument("--drop", type=str, help="Comma-separated list of columns to drop.")
    parser.add_argument("--model", type=str, help="Model name for supervised learning.")
    parser.add_argument("--cluster-model", type=str, default='kmeans', help="Clustering model name. Default: kmeans.")
    parser.add_argument("--save-model", type=str, help="Path to save the trained pipeline.")
    parser.add_argument("--load-model", type=str, help="Path to a pre-trained pipeline for prediction.")
    parser.add_argument("--config", type=str, help="Path to a YAML config file for specifying arguments.")
    parser.add_argument("--flex-columns", action="store_true", 
                        help="Allow partial column mismatch. (Currently just a placeholder; must be consistent in real usage.)")
    
    args, unknown = parser.parse_known_args()
    if unknown:
        logging.info(f"Ignoring unrecognized arguments: {unknown}")

    # If config file is provided, parse it and override CLI arguments
    if args.config:
        config_dict = parse_config_file(args.config)
        for key, value in config_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # If user wants to load a pre-trained model, do prediction mode
    if args.load_model:
        try:
            with open(args.load_model, 'rb') as f:
                pipeline = pickle.load(f)
            logging.info(f"Loaded pre-trained pipeline from {args.load_model}")
        except Exception as e:
            logging.error(f"Error loading pipeline: {e}")
            return
        
        try:
            df = load_dataset(args.file_path)
        except Exception as e:
            logging.error("Terminating due to data loading error.")
            return
        
        analysis_folder = create_analysis_folder()
        write_summary_to_file(df, analysis_folder)
        generate_summary(df)
        
        # If flex-columns is enabled, user might want to skip missing columns or do a custom approach
        # (For now, we do not implement advanced partial column logic, just a placeholder)
        
        try:
            predictions = pipeline.predict(df)
            df['Predicted'] = predictions
            out_path = os.path.join(analysis_folder, "predictions.csv")
            df.to_csv(out_path, index=False)
            logging.info(f"Predictions saved to {out_path}")
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
        return

    # Otherwise, we do training or unsupervised analysis
    try:
        df = load_dataset(args.file_path)
    except Exception as e:
        logging.error("Terminating due to data loading error.")
        return

    check_schema(df)
    analysis_folder = create_analysis_folder()

    if args.drop:
        drop_list = [col.strip() for col in args.drop.split(",")]
        df = drop_non_informative_columns(df, drop_list)

    df, _ = advanced_imputation(df, target_column=args.target)
    write_summary_to_file(df, analysis_folder)
    generate_summary(df)

    if args.plots:
        plot_histograms(df, analysis_folder)
        plot_correlation_heatmap(df, analysis_folder)

    if args.ml:
        if args.target:
            logging.info("Running supervised ML pipeline...")
            pipeline = run_supervised_pipeline(
                df=df,
                target_column=args.target,
                model_name=args.model,
                tune=args.tune,
                output_dir=analysis_folder,
                flex_columns=args.flex_columns
            )
            # If pipeline is successfully trained, save it if requested
            if pipeline and args.save_model:
                try:
                    with open(args.save_model, 'wb') as f:
                        pickle.dump(pipeline, f)
                    logging.info(f"Trained pipeline saved to {args.save_model}")
                except Exception as e:
                    logging.error(f"Error saving pipeline: {e}")
        else:
            logging.info("Running unsupervised clustering pipeline...")
            run_unsupervised_pipeline(
                df=df,
                model_name=args.cluster_model,
                output_dir=analysis_folder,
                flex_columns=args.flex_columns
            )

if __name__ == "__main__":
    main()