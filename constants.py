"""
Varibales shared between churn_library.py 
and churn_script_logging_and_tests.py

Author: Sohrab Sani
Date: 2022-11-22
"""

DATA_PATH = './data/bank_data.csv'

CAT_COLUMNS = ['Gender', 'Education_Level',
               'Marital_Status', 'Income_Category', 'Card_Category']

KEEP_COLUMNS = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                'Total_Relationship_Count', 'Months_Inactive_12_mon',
                'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                'Income_Category_Churn', 'Card_Category_Churn']
