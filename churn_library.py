"""
Customer churn prediction and analysis

Author: Sohrab Sani
Date: 2022-11-22
"""

# import libraries
import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import constants as setting

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/results.log',
    level=logging.INFO,
    filemode='a+',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")


def import_data(pth):
    '''
    returns DataFrame for the csv found at pth

    input:
            pth: a str path to the csv
    output:
            df: pandas DataFrame
    '''
    try:
        df_input = pd.read_csv(pth)
        df_input['Churn'] = df_input['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        logging.info('SUCCESS: file %s loaded successfully', pth)
        logging.info('SUCCESS: df size %s', df_input.shape)
        df_out = df_input.drop(
            ['Unnamed: 0', 'CLIENTNUM', 'Attrition_Flag'], axis=1)
        return df_out
    except (FileNotFoundError, KeyError) as err:
        logging.error("ERROR:  %s", err)
        raise err


def perform_eda(df_input):
    '''
    perform eda on df_input and save figures to images folder
    input:
            df_input: pandas dataframe

    output:
            True
    '''
    hist_columns = {'Churn': 'churn_distribution.png',
                    'Customer_Age': 'customer_age_distribution.png',
                    }
    for key, value in hist_columns.items():
        plt.figure(figsize=(20, 10))
        df_input[key].hist()
        plt.savefig(fname=f'./images/eda/{value}')
        plt.close()
    # Marital_Status hist
    plt.figure(figsize=(20, 10))
    df_input.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(fname='./images/eda/marital_status_distribution.png')
    plt.close()
    # Total_Trans_Ct hist
    plt.figure(figsize=(20, 10))
    sns.histplot(df_input['Total_Trans_Ct'], kde=True)
    plt.savefig(fname='./images/eda/total_transaction_distribution.png')
    plt.close()
    # Heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(df_input.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(fname='./images/eda/heatmap.png')
    plt.close()
    logging.info('SUCCESS: EDA')
    return True


def encoder_helper(df_input, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df_input: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that
            could be used for naming variables or index y column]
    output:
            df_input: pandas dataframe with new columns for
    '''
    col_check = all(item in df_input.columns for item in category_lst)
    assert response in df_input.columns
    assert col_check
    for category in category_lst:
        column_groups = df_input.groupby(category).mean()[response]
        df_input[f'{category}_{response}'] = df_input[category].apply(
            lambda x: column_groups.loc[x])
        logging.info("SUCCESS: encoding %s", category)
    return df_input


def perform_feature_engineering(df_input, response):
    '''
    input:
              df_input: pandas dataframe
              response: string of response name [optional argument
              that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    col_check = all(item in df_input.columns for item in setting.KEEP_COLUMNS)
    assert response in df_input.columns
    assert col_check
    y_df = df_input[response]
    x_df = df_input[setting.KEEP_COLUMNS]
    x_train, x_test, y_train, y_test = train_test_split(x_df,
                                                        y_df,
                                                        test_size=0.3,
                                                        random_state=42)

    logging.info('SUCCESS: x_train shape %s', x_train.shape)
    logging.info('SUCCESS: y_train shape %s', y_train.shape)
    logging.info('SUCCESS: x_test shape %s', x_test.shape)
    logging.info('SUCCESS: y_test shape %s', y_test.shape)
    logging.info('SUCCESS: feature engineering')
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    # RandomForestClassifier
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
             str('Random Forest Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Random Forest Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/rf_results.png')
    plt.close()
    # LogisticRegression
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
             str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/logistic_results.png')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    try:
        # get feature importance
        importance_list = model.feature_importances_

        # sort feature importance_list in descending order
        indices = np.argsort(importance_list)[::-1]

        # match columns name with importance_list
        names = [X_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')
        plt.bar(range(X_data.shape[1]), importance_list[indices])
        plt.xticks(range(X_data.shape[1]), names, rotation=90)
        file_path = f'{output_pth}/feature_importance_plot.jpg'
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
        logging.info("SUCCESS: Feature importance added to %s",
                     file_path)
    except (AttributeError) as err:
        logging.error("ERROR: feature_importance_plot;  %s", err)
        raise err


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    logging.info('CV RF score %s', cv_rfc.best_score_)
    logging.info('CV RF best_params_ %s', cv_rfc.best_params_)
    logging.info('SUCCESS: RandomForestClassifier trained')
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(X_train, y_train)
    logging.info('SUCCESS: LogisticRegression trained')

    # save models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    logging.info('SUCCESS: Models saved')

    # predict train and test predictions for RF and LR
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    #  ROC curve
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    plot_roc_curve(lrc, X_test, y_test, ax=axis, alpha=0.8)
    plot_roc_curve(cv_rfc.best_estimator_,
                   X_test, y_test, ax=axis, alpha=0.8)
    plt.savefig(fname='./images/results/roc_curve_result.png')
    plt.close()
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(model=cv_rfc.best_estimator_,
                            X_data=X_test,
                            output_pth='./images/results/')
    return True


if __name__ == '__main__':
    # import data
    df = import_data(pth=setting.DATA_PATH)

    # perform eda
    perform_eda(df)

    # data transformation
    df = encoder_helper(df,
                        category_lst=setting.CAT_COLUMNS,
                        response='Churn')

    # spit data from model training
    X_train, X_test, y_train_df, y_test = perform_feature_engineering(
        df, "Churn")

    # model trining
    train_models(X_train, X_test, y_train_df, y_test)
