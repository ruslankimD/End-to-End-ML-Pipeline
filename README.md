# End-to-End Machine Learning Pipeline

This repository contains an end-to-end Machine Learning pipeline designed to predict **Telco Customer Churn**. It covers data validation, preprocessing, feature engineering, model training, and evaluation, integrated with MLflow for experiment tracking and Great Expectations for data quality validation.

## Business Goal

The goal of this project is to help telecom companies identify customers likely to churn, enabling proactive retention strategies.

## Features

- **Data Validation**: Uses `great_expectations` to validate the quality of incoming data before training.
- **Data Preprocessing & Feature Engineering**: Automated data cleaning, one-hot encoding, and binary variable transformation.
- **Model Training**: Implements an optimized `XGBoost` classifier, automatically handling class imbalance.
- **Experiment Tracking**: Tracks hyperparameters, model performance metrics, and artifacts via **MLflow**.
- **Model Serialization**: Saves preprocessing artifacts and trained models for serving.

## Tech Stack
- **Language**: Python 3
- **Machine Learning**: `xgboost`, `scikit-learn`, `pandas`, `numpy`
- **Experiment Tracking**: `mlflow`
- **Data Validation**: `great_expectations`

## Model Performance

- F1-score: 0.590
- ROC-AUC: 0.846
- Precision: 0.433
- Recall: 0.922

##  Project Structure

```text
├── app/                  # Web app/API serving (Flask/FastAPI/Gradio)
├── artifacts/            # Saved artifacts like preprocessing pipelines
├── configs/              # Configuration files
├── data/                 # Raw and processed datasets
├── docker/               # Docker configurations for deployment
├── great_expectations/   # Data validation configurations
├── mlruns/               # Local MLflow tracking files
├── notebooks/            # Jupyter Notebooks for EDA and experiments
├── scripts/
│   └── run_pipeline.py   # Main pipeline execution script
├── src/                  # Source code (data, features, models, utils)
├── tests/                # Unit tests
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ruslankimD/End-to-End-ML-Pipeline.git
   cd End-to-End-ML-Pipeline
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   # source .venv/bin/activate
   
   pip install -r requirements.txt
   ```

## Usage

To execute the entire training pipeline, run the `run_pipeline.py` script and specify the path to your raw dataset:

```bash
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv
```

### Key Arguments:
- `--input`: Path to the raw CSV data **[Required]**
- `--target`: Name of the target column (default: `Churn`)
- `--threshold`: Classification decision threshold (default: `0.35`)
- `--test_size`: Train/test split ratio (default: `0.2`)
- `--experiment`: MLflow experiment name (default: `Telco Churn`)


