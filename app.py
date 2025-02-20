import streamlit as st
import pandas as pd
import io
import os
import logging

# Bring in the pipeline functions from your universal_dataset_ml_pipeline
from universal_dataset_ml_pipeline import (
    drop_non_informative_columns,
    advanced_imputation,
    run_unsupervised_pipeline,
    create_analysis_folder,
    write_summary_to_file,
    generate_summary,
    # We'll re-import relevant pieces in our local function run_supervised_pipeline_with_metrics
)

def run_supervised_pipeline_with_metrics(
    df, 
    target_column, 
    model_name=None, 
    tune=False, 
    output_dir="output"
):

    import numpy as np
    from universal_dataset_ml_pipeline import (
        preprocess_features,
        build_supervised_pipeline,
        tune_hyperparameters,
        explain_model,
        CLASSIFICATION_MODELS,
        REGRESSION_MODELS,
        visualize_classification_results,
        visualize_regression_results
    )
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, r2_score

    # Basic checks
    if target_column not in df.columns:
        logging.error(f"Target column '{target_column}' not found in dataset.")
        return None

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Classification if y is object/categorical or few unique values
    if y.dtype == 'object' or y.dtype.name == 'category' or y.nunique() < 10:
        problem_type = 'classification'
        model_config = CLASSIFICATION_MODELS.get(model_name, CLASSIFICATION_MODELS['logistic'])
    else:
        problem_type = 'regression'
        model_config = REGRESSION_MODELS.get(model_name, REGRESSION_MODELS['randomforest'])

    logging.info(f"Using {problem_type} model: {model_name if model_name else 'default'}")
    model = model_config['class']()
    preprocessor = preprocess_features(X)

    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if tune:
        logging.info("Tuning hyperparameters...")
        pipeline = tune_hyperparameters(pipeline, X_train, y_train, model_config['params'], problem_type)
    else:
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metric_value = None
    if problem_type == 'classification':
        acc = accuracy_score(y_test, y_pred)
        logging.info(f"Classification Accuracy: {acc:.2f}")
        visualize_classification_results(y_test, y_pred, output_dir)
        metric_value = acc
    else:
        score = r2_score(y_test, y_pred)
        logging.info(f"Regression R^2 Score: {score:.2f}")
        visualize_regression_results(y_test, y_pred, output_dir)
        metric_value = score

    logging.info("Computing SHAP values for model interpretability...")
    explain_model(pipeline, X_train, output_dir)

    return pipeline, metric_value, problem_type

def main():
    st.title("UniML Explorer")

    st.write("""
    1. Upload a dataset (CSV, Excel, or JSON).
    2. (Optional) Choose columns to drop.
    3. Pick whether to run ML. If you select a target column, a supervised approach is used; 
       otherwise, it runs unsupervised clustering.
    4. Click "Run Analysis" to execute data cleaning, EDA, and optional ML tasks.
    """)

    # Step 1: File Upload
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv","xlsx","xls","json"])
    if not uploaded_file:
        st.info("Please upload a file to proceed.")
        return

    # Attempt to read a sample to list columns
    file_name = uploaded_file.name
    extension = os.path.splitext(file_name)[1].lower()

    df_sample_for_columns = None
    try:
        uploaded_file.seek(0)  # ensure pointer is at start
        if extension == ".csv":
            df_sample_for_columns = pd.read_csv(uploaded_file, nrows=100)
        elif extension in [".xlsx", ".xls"]:
            df_sample_for_columns = pd.read_excel(uploaded_file, nrows=100)
        elif extension == ".json":
            df_tmp = pd.read_json(uploaded_file)
            df_sample_for_columns = df_tmp.head(100)
        else:
            st.error("Unsupported file type.")
            return
    except Exception as e:
        st.error(f"Error reading file structure: {e}")
        return

    num_cols = 0
    if df_sample_for_columns is not None:
        num_cols = len(df_sample_for_columns.columns)

    # Step 2: Column-Drop Multi-Select
    if num_cols > 50:
        st.info(f"Your dataset has {num_cols} columns. Skipping multi-select to avoid performance issues.")
        drop_cols = []
    else:
        drop_cols = st.multiselect("Columns to drop:", df_sample_for_columns.columns, default=[])

    # Step 3: ML Pipeline Options
    run_plots = st.checkbox("Generate plots (histograms, correlation heatmap, etc.)")
    run_ml = st.checkbox("Run ML Pipeline?")

    # Let user pick target from a selectbox (or "None => unsupervised")
    target_choices = [None] + list(df_sample_for_columns.columns)
    chosen_target = st.selectbox("Select Target Column (None => unsupervised)", target_choices, index=0)

    tune_hparams = False
    model_name = "randomforest"
    cluster_model = "kmeans"

    if run_ml:
        tune_hparams = st.checkbox("Hyperparameter Tuning?")
        model_name = st.selectbox("Supervised Model (classification/regression)", 
                                  ["randomforest","ridge","lasso","svm","logistic","gradientboosting"],
                                  index=0)
        cluster_model = st.selectbox("Clustering Model", ["kmeans","dbscan"], index=0)

    # Step 4: The "Run Analysis" button
    if st.button("Run Analysis"):
        analysis_folder = create_analysis_folder()

        # Attempt to read the entire dataset
        uploaded_file.seek(0)
        try:
            if extension == ".csv":
                df = pd.read_csv(uploaded_file)
            elif extension in [".xlsx", ".xls"]:
                df = pd.read_excel(uploaded_file)
            elif extension == ".json":
                df = pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file type.")
                return
        except Exception as e:
            st.error(f"Error reading dataset fully: {e}")
            return

        st.success(f"File '{file_name}' loaded successfully!")

        # Drop selected columns
        if drop_cols:
            df = drop_non_informative_columns(df, drop_cols)

        # Advanced imputation
        target_str = chosen_target if chosen_target not in (None, "") else None
        df, _ = advanced_imputation(df, target_column=target_str)

        # Summaries
        write_summary_to_file(df, analysis_folder)
        generate_summary(df)
        st.info("Dataset summary and logs saved.")

        # Plots if requested
        if run_plots:
            from universal_dataset_ml_pipeline import plot_histograms, plot_correlation_heatmap
            plot_histograms(df, analysis_folder)
            plot_correlation_heatmap(df, analysis_folder)
            st.success("Plots generated and saved.")

        # ML pipeline if requested
        if run_ml:
            if target_str:
                st.info("Running supervised ML pipeline...")
                pipeline_and_metrics = run_supervised_pipeline_with_metrics(
                    df=df,
                    target_column=target_str,
                    model_name=model_name,
                    tune=tune_hparams,
                    output_dir=analysis_folder
                )
                if pipeline_and_metrics is not None:
                    pipeline_obj, metric_value, problem_type = pipeline_and_metrics
                    if pipeline_obj:
                        if problem_type == 'classification':
                            st.success(f"Classification Accuracy: {metric_value:.2f}")
                        else:
                            st.success(f"Regression R^2: {metric_value:.2f}")
                    else:
                        st.warning("No pipeline returned (check logs for errors).")
            else:
                st.info("Running unsupervised clustering pipeline...")
                run_unsupervised_pipeline(
                    df=df,
                    model_name=cluster_model,
                    output_dir=analysis_folder
                )
                st.success("Unsupervised clustering completed.")

        st.balloons()
        st.success("Analysis complete! Check the created folder for outputs/logs.")

if __name__ == "__main__":
    st.set_page_config(page_title="UniML Explorer", layout="wide")
    main()



