"""
Customer churn test churn_library.py

Author: Sohrab Sani
Date: 2022-11-22
"""

import os
import logging
import pytest
import pandas as pd
from churn_library import (import_data,
                           perform_eda,
                           encoder_helper,
                           perform_feature_engineering,
                           train_models)

import constants as setting

logging.basicConfig(
    filename='./logs/test.log',
    level=logging.INFO,
    filemode='a+',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")


@pytest.fixture(scope="module")
def path():
    """
    Fixture - The test function test_import() will
    use the return of path() as an argument
    """
    return setting.DATA_PATH


def test_import(path):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df_input = pd.read_csv(path)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    try:
        assert df_input.shape[0] > 0
        assert df_input.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(path):
    '''
    test perform eda function
    '''
    df_input = import_data(path)
    try:
        perform_eda(df_input=df_input)
        logging.info("SUCCESS: perform_eda")
    except KeyError as err:
        logging.error('Column "%s" not found', err.args[0])
        raise err
    try:
        assert os.path.isfile("./images/eda/churn_distribution.png") is True
        logging.info('File %s was found', 'churn_distribution.png')
        assert os.path.isfile(
            "./images/eda/customer_age_distribution.png") is True
        logging.info('File %s was found', 'customer_age_distribution.png')
        assert os.path.isfile(
            "./images/eda/marital_status_distribution.png") is True

        assert os.path.isfile(
            "./images/eda/total_transaction_distribution.png") is True
        logging.info('File %s was found', 'total_transaction_distribution.png')
        assert os.path.isfile("./images/eda/heatmap.png") is True
        logging.info('File %s was found', 'heatmap.png')
    except AssertionError as err:
        logging.error(err)
        raise err


def test_encoder_helper(path):
    '''
    test encoder helper
    '''
    df_input = import_data(path)
    try:
        df_out = encoder_helper(df_input,
                                category_lst=setting.CAT_COLUMNS,
                                response='Churn')
        logging.info('df_shape: %s', df_out.shape)
    except AssertionError as err:
        logging.error(err)
        raise err


def test_perform_feature_engineering(path):
    '''
    test perform_feature_engineering
    '''
    df_input = import_data(path)
    df_input = encoder_helper(df_input,
                              category_lst=setting.CAT_COLUMNS,
                              response='Churn')
    try:
        X_train, X_test, y_train_df, y_test = perform_feature_engineering(
            df_input, "Churn")
        logging.info('df_shape: %s', X_train.shape)
        logging.info('df_shape: %s', y_train_df.shape)
        logging.info('df_shape: %s', X_test.shape)
        logging.info('df_shape: %s', y_test.shape)

    except Exception as err:
        logging.error(err)
        raise err


def test_train_models(path):
    '''
    test train_models
    '''
    df_input = import_data(path)
    df_input = encoder_helper(df_input,
                              category_lst=setting.CAT_COLUMNS,
                              response='Churn')
    X_train, X_test, y_train_df, y_test = perform_feature_engineering(
        df_input, "Churn")
    train_models(X_train, X_test, y_train_df, y_test)
    pass


if __name__ == "__main__":
    pass
