"""
Customer churn test churn_library.py 

Author: Sohrab Sani
Date: 2022-11-22
"""

import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/test.log',
    level=logging.INFO,
    filemode='a+',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    df = cls.import_data("./data/bank_data.csv")
    try:
        cls.perform_eda(df_input=df)
        logging.info("SUCCESS: perform_eda")
    except KeyError as err:
        logging.error('Column "%s" not found', err.args[0])
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''


def test_train_models(train_models):
    '''
    test train_models
    '''


if __name__ == "__main__":
    df = test_import("./data/bank_data.csv")
