# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

This repo contains code for the customer churn ML project based on a data set that Kaggle provided.

This project seeks to model a customer churn classifier  based on the available data with the following steps: 

- Exploratory data analysis and visualizations
- Feature engineering to transform data
- Model training and evaluation 

## Files and data description

```
├── README.md   <- readme
├── churn_library.py    <- main code to run
├── churn_script_logging_and_tests.py   <- contains unit tests
├── constants.py
├── data        <- req dataset
│   └── bank_data.csv
├── images  <- images and resutls 
│   ├── eda
│   │   ├── churn_distribution.png
│   │   ├── customer_age_distribution.png
│   │   ├── heatmap.png
│   │   ├── marital_status_distribution.png
│   │   └── total_transaction_distribution.png
│   └── results
│       ├── feature_importance_plot.jpg
│       ├── logistic_results.png
│       ├── rf_results.png
│       └── roc_curve_result.png
├── logs
│   └── results.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── requirements_py3.6.txt
├── requirements_py3.8.txt
```

## Running Files

- The pipeline gets executed with `python churn_library.py`. After the code ran you should see
    1. run update in `./logs/results.log`
    2. EDA images added to `./images/eda/`
    3. model training results in `./images/results/`
    4. trained models dumped in 1`./results`
- For testing with `pytest churn_script_logging_and_tests.py` and you should see 5 passed test