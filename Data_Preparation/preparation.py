#Imports
import pandas as pd
from io import BytesIO
import pickle
from os import EX_SOFTWARE
from os.path import isfile, join
from pathlib import Path
import pandas as pd
import math
import numpy as np
from pathlib import Path
import os
from os import listdir
import joblib
import matplotlib
import matplotlib.pyplot as plt
from datalab.context import Context
import datalab.storage as storage
from io import BytesIO
import pickle
import seaborn as sns
from LABANKA_PUBLIC.utils import get_file_names, none_norm, df_optimized, get_dfs, missing_data, plot_stats, replace_none_values, one_hot_encoder, label_encoder, get_pickle_gcp

def main_dataset():
    df_app = get_application()
    df_prev_app = get_prev_application()
    df_ip = get_inst_payment()
    df_bureau = get_bureau()

    #merge all datasets
    df_global = pd.merge(df_app, df_prev_app, on='SK_ID_CURR', how='left')
    df_global = pd.merge(df_global, df_ip, on='SK_ID_CURR', how='left')
    df_global = pd.merge(df_global, df_bureau, on='SK_ID_CURR', how='left' )

    #Clean the Column names, at this moment they are tupples
    for col in df_global.columns:
        if type(col) ==tuple:
            df_global = df_global.rename(columns={col:f'{col[0]}_{col[1]}'})

    #Splitting the data between two dataframes - with bureau or without bureau

    ## Dataframe with bureau data
    bureau_df = df_global[(df_global['CREDIT_TYPE_Credit card_mean_active'].notnull())]

    ## Dataframe without bureau data
    no_bureau_df = df_global[(df_global['CREDIT_TYPE_Credit card_mean_active'].isnull())]

    ## Cleaning the unnecessary columns from no_bureau_df
    ## We will clean now the df without bureau data from the columns of bureau data
    ### First we will define the columns to be erased
    bureau_columns = ['CREDIT_CURRENCY_currency 1_mean_active',
       'CREDIT_CURRENCY_currency 2_mean_active',
       'CREDIT_CURRENCY_currency 3_mean_active',
       'CREDIT_CURRENCY_currency 4_mean_active',
       'CREDIT_TYPE_Another type of loan_mean_active',
       'CREDIT_TYPE_Car loan_mean_active',
       'CREDIT_TYPE_Cash loan (non-earmarked)_mean_active',
       'CREDIT_TYPE_Consumer credit_mean_active',
       'CREDIT_TYPE_Credit card_mean_active',
       'CREDIT_TYPE_Interbank credit_mean_active',
       'CREDIT_TYPE_Loan for business development_mean_active',
       'CREDIT_TYPE_Loan for purchase of shares (margin lending)_mean_active',
       'CREDIT_TYPE_Loan for the purchase of equipment_mean_active',
       'CREDIT_TYPE_Loan for working capital replenishment_mean_active',
       'CREDIT_TYPE_Microloan_mean_active',
       'CREDIT_TYPE_Mobile operator loan_mean_active',
       'CREDIT_TYPE_Mortgage_mean_active',
       'CREDIT_TYPE_Real estate loan_mean_active',
       'CREDIT_TYPE_Unknown type of loan_mean_active',
       'AMT_CREDIT_SUM_OVERDUE_max_active',
       'AMT_CREDIT_SUM_OVERDUE_mean_active', 'BAD_STATUS_COUNT_mean_active',
       'BAD_STATUS_COUNT_max_active', 'AMT_CREDIT_MAX_OVERDUE_max_active',
       'AMT_CREDIT_MAX_OVERDUE_mean_active', 'CREDIT_DAY_OVERDUE_mean_active',
       'CREDIT_DAY_OVERDUE_max_active', 'CNT_CREDIT_PROLONG_mean_active',
       'CNT_CREDIT_PROLONG_max_active', 'AMT_CREDIT_SUM_DEBT_max_active',
       'AMT_CREDIT_SUM_DEBT_sum_active',
       'CREDIT_CURRENCY_currency 1_mean_closed',
       'CREDIT_CURRENCY_currency 2_mean_closed',
       'CREDIT_CURRENCY_currency 3_mean_closed',
       'CREDIT_CURRENCY_currency 4_mean_closed',
       'CREDIT_TYPE_Another type of loan_mean_closed',
       'CREDIT_TYPE_Car loan_mean_closed',
       'CREDIT_TYPE_Cash loan (non-earmarked)_mean_closed',
       'CREDIT_TYPE_Consumer credit_mean_closed',
       'CREDIT_TYPE_Credit card_mean_closed',
       'CREDIT_TYPE_Interbank credit_mean_closed',
       'CREDIT_TYPE_Loan for business development_mean_closed',
       'CREDIT_TYPE_Loan for purchase of shares (margin lending)_mean_closed',
       'CREDIT_TYPE_Loan for the purchase of equipment_mean_closed',
       'CREDIT_TYPE_Loan for working capital replenishment_mean_closed',
       'CREDIT_TYPE_Microloan_mean_closed',
       'CREDIT_TYPE_Mobile operator loan_mean_closed',
       'CREDIT_TYPE_Mortgage_mean_closed',
       'CREDIT_TYPE_Real estate loan_mean_closed',
       'CREDIT_TYPE_Unknown type of loan_mean_closed',
       'AMT_CREDIT_SUM_OVERDUE_max_closed',
       'AMT_CREDIT_SUM_OVERDUE_mean_closed', 'BAD_STATUS_COUNT_mean_closed',
       'BAD_STATUS_COUNT_max_closed', 'AMT_CREDIT_MAX_OVERDUE_max_closed',
       'AMT_CREDIT_MAX_OVERDUE_mean_closed', 'CREDIT_DAY_OVERDUE_mean_closed',
       'CREDIT_DAY_OVERDUE_max_closed', 'CNT_CREDIT_PROLONG_mean_closed',
       'CNT_CREDIT_PROLONG_max_closed', 'AMT_CREDIT_SUM_DEBT_max_closed',
       'AMT_CREDIT_SUM_DEBT_sum_closed']
    
    ### Droping the columns from the dataframe with no bureau balance
    no_bureau_df.drop(columns=bureau_columns, inplace=True)

    #Adding the previous application existance of information as a feature
    ## Bureau_df
    bureau_df['with_previous'] = np.where((bureau_df['SK_ID_PREV_nunique'].notnull()), 1, 0)
    ## No bureau_df
    no_bureau_df['with_previous'] = np.where((no_bureau_df['SK_ID_PREV_nunique'].notnull()), 1, 0)

    # Droping the last columns
    bureau_df.drop(columns=['SK_ID_PREV_nunique','index', 'SK_ID_CURR'], inplace=True)
    no_bureau_df.drop(columns=['SK_ID_PREV_nunique','index', 'SK_ID_CURR'], inplace=True)

    # Filling in the missing values based on the mean of the default or non-default subcategories
    
    ## Filling in the missing values for bureau_df
    for col in bureau_df.drop(columns=['TARGET']).columns:
        mask_filter_target_1 = bureau_df['TARGET'] == 1
        mask_filter_target_0 = bureau_df['TARGET'] == 0
        mean_value_0 = bureau_df[mask_filter_target_0][col].mean()
        mean_value_1 = bureau_df[mask_filter_target_1][col].mean()
        mask_filter_null = bureau_df[col].isnull()
        bureau_df.loc[mask_filter_target_0 & mask_filter_null,col] = mean_value_0
        bureau_df.loc[mask_filter_target_1 & mask_filter_null,col] = mean_value_1
    
    ## Filling in the missing values for no_bureau_df
    for col in no_bureau_df.drop(columns=['TARGET']).columns:
        mask_filter_target_1 = no_bureau_df['TARGET'] == 1
        mask_filter_target_0 = no_bureau_df['TARGET'] == 0
        mean_value_0 = no_bureau_df[mask_filter_target_0][col].mean()
        mean_value_1 = no_bureau_df[mask_filter_target_1][col].mean()
        mask_filter_null = no_bureau_df[col].isnull()
        no_bureau_df.loc[mask_filter_target_0 & mask_filter_null,col] = mean_value_0
        no_bureau_df.loc[mask_filter_target_1 & mask_filter_null,col] = mean_value_1

    # Loading bureau_df to GCP
    ## Create a local pickle file
    bureau_df.to_pickle('bureau_df.pkl')
    base_bucket = storage.Bucket("wagon-data-618-le-banq")

    ## Write pickle to GCS
    sample_item = base_bucket.item('bureau_df.pkl')
    with open('bureau_df.pkl', 'rb') as f:
        sample_item.write_to(bytearray(f.read()), 'model/bureau_df.pkl')

    # Loading no_bureau_df to GCP
    ## Create a local pickle file
    no_bureau_df.to_pickle('no_bureau_df.pkl')
    base_bucket = storage.Bucket("wagon-data-618-le-banq")

    ## Write pickle to GCS
    sample_item = base_bucket.item('no_bureau_df.pkl')
    with open('no_bureau_df.pkl', 'rb') as f:
        sample_item.write_to(bytearray(f.read()), 'model/no_bureau_df.pkl')

    return bureau_df, no_bureau_df

def split_dataset(df_global):
    #Split between dataset with previous application data and without previous applications
    # df1 is the dataframe where we have information from previous applications
    df1 = df_global[df_global['SK_ID_PREV_nunique'].notnull()]

    #df2 is the dataframe where we don't have information from previous applications
    df2 = df_global[df_global['SK_ID_PREV_nunique'].isnull()]

    #Drop all columns for dataframe without previous data
    df2 = df2.dropna(axis=1,thresh=1000)

    return df1, df2


def get_application():
    
    #importing the data
    file_path = 'raw_data/application_train.pkl'
    
    #Storing the data from google cloud storage
    master_df_input_files = {}
    master_df_input_files['application_train.pkl'] = get_pickle_gcp(bucket="wagon-data-618-le-banq", file_path=file_path)

    df_copy = master_df_input_files['application_train.pkl']
    
    #Make a safety copy so we can run the function several times
    df = df_copy.copy() 

    #Feature Engineering
    df['ANNUITY_OVER_CREDIT_RATIO'] = df.AMT_ANNUITY.div(df.AMT_CREDIT)
    df.loc[~np.isfinite(df['ANNUITY_OVER_CREDIT_RATIO']), 'ANNUITY_OVER_CREDIT_RATIO'] = np.nan
    
    #Columns to preserve
    current_applications_columns = ['SK_ID_CURR','TARGET','CNT_CHILDREN', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE', 'AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_YEAR','ANNUITY_OVER_CREDIT_RATIO']
    # Droping the columns from the previous filter
    df.drop(df.columns.difference(current_applications_columns), 1, inplace=True)

    #Data Cleaning
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)    #replace outlier

    #Defining categorical columns
    categorical_columns = ['NAME_INCOME_TYPE', 'OCCUPATION_TYPE']
   
    # Modifying DF one hot encoder for NAME_INCOME_TYPE & OCCUPATION_TYPE
    df, categorical_cols = one_hot_encoder(df,categorical_columns=categorical_columns)

    #Storing the base dataframe pre label encoding to assign then matching values
    df_base = df.copy()

    # Doing the label encoder for 'ORGANIZATION_TYPE' because it has too many categories
    df, encode_label_cols = label_encoder(df, categorical_columns=['ORGANIZATION_TYPE'])

    #Creating a master label reconciliation table
    

    #Resetting index
    df = df.reset_index()

    # Replacing nas by means 
    for col in df.drop(columns=['TARGET']).columns:
        mask_filter_target_1 = df['TARGET'] == 1
        mask_filter_target_0 = df['TARGET'] == 0
        mean_value_0 = df[mask_filter_target_0][col].mean()
        mean_value_1 = df[mask_filter_target_1][col].mean()
        mask_filter_null = df[col].isnull()
        df.loc[mask_filter_target_0 & mask_filter_null,col] = mean_value_0
        df.loc[mask_filter_target_1 & mask_filter_null,col] = mean_value_1

    return df

def get_prev_application():

    #import the data
    file_path = 'raw_data/previous_application.pkl'
    master_df_input_files = {}
    master_df_input_files['previous_application.pkl'] = get_pickle_gcp(bucket="wagon-data-618-le-banq", file_path=file_path)
    
    #Storing the data from google cloud storage
    df_copy = master_df_input_files['previous_application.pkl']
    
    #Make a safety copy so we can run the function several times
    df_prevapp = df_copy.copy() 

    #Get only last application for the previous contrat
    df_prevapp = df_prevapp[df_prevapp['FLAG_LAST_APPL_PER_CONTRACT'] == 'Y']

    #Feature Engineering
    df_prevapp['APPLICATION_CREDIT_DIFF'] = df_prevapp['AMT_APPLICATION'] - df_prevapp['AMT_CREDIT']
    #df_prevapp['CREDIT_TO_ANNUITY_RATIO'] = df_prevapp['AMT_CREDIT'] / df_prevapp['AMT_ANNUITY']
    #Dealing with 0/0 zerodivisionerror, infinity
    df_prevapp['CREDIT_TO_ANNUITY_RATIO'] = df_prevapp.AMT_CREDIT.div(df_prevapp.AMT_ANNUITY)
    df_prevapp.loc[~np.isfinite(df_prevapp['CREDIT_TO_ANNUITY_RATIO']), 'CREDIT_TO_ANNUITY_RATIO'] = np.nan
    
    #Drop Columns
    #Columns that we decided to keep for the first model
    prev_app_cols =['SK_ID_PREV','SK_ID_CURR','NAME_CONTRACT_TYPE','AMT_CREDIT',
                'AMT_ANNUITY','NAME_PORTFOLIO','NAME_CONTRACT_STATUS',
                'APPLICATION_CREDIT_DIFF','CREDIT_TO_ANNUITY_RATIO']
    
    df_prevapp = df_prevapp[prev_app_cols]

    #Filling missing values
    #Using the most common value
    df_prevapp = df_prevapp.fillna(df_prevapp['NAME_PORTFOLIO'].value_counts().index[0])
    df_prevapp = df_prevapp.fillna(df_prevapp['NAME_CONTRACT_TYPE'].value_counts().index[0])

    #One Hot Encoder categorical features
    df_prevapp, cat_cols = one_hot_encoder(df_prevapp, categorical_columns=['NAME_CONTRACT_TYPE','NAME_PORTFOLIO','NAME_CONTRACT_STATUS'],nan_as_category=False)

    #Because the functions one hot encode we are using changes the dtype of the columns
    #we are going to convert to float
    cols = df_prevapp.select_dtypes(exclude=['float', 'uint8']).columns
    df_prevapp[cols] = df_prevapp[cols].apply(pd.to_numeric, downcast='float', errors='coerce')


    #Count Number of refused & approved credits by SK_ID_CURR    
    df_prevapp['PREV_REFUSED_COUNT'] = df_prevapp.groupby(['SK_ID_CURR'])['NAME_CONTRACT_STATUS_Refused'].transform('sum')
    df_prevapp['PREV_APPROVED_COUNT'] = df_prevapp.groupby(['SK_ID_CURR'])['NAME_CONTRACT_STATUS_Approved'].transform('sum')

    #Make a dictionary of the features for the aggregation
    # Categorical features
    prev_app_cat_agg = {key: ['mean'] for key in cat_cols}
    #Numerical features
    prev_app_num_agg = {
    'SK_ID_PREV': ['nunique'],
    'AMT_ANNUITY': ['min', 'max', 'mean'],
    # Engineered features
    'CREDIT_TO_ANNUITY_RATIO': ['mean', 'max'],
    'APPLICATION_CREDIT_DIFF': ['min', 'max', 'mean'],
    'PREV_REFUSED_COUNT': ['mean'],
    'PREV_APPROVED_COUNT': ['mean']
    }

    #Group the two dictionaries
    PREV_AGG = prev_app_cat_agg.copy()
    PREV_AGG.update(prev_app_num_agg)

    #Transform the dataset into one single line per SK_ID_CURR

    df_prevapp = df_prevapp.groupby('SK_ID_CURR').agg(PREV_AGG).reset_index()
    return df_prevapp

def get_inst_payment():

    #import the data
    file_path = 'raw_data/installments_payments.pkl'
    master_df_input_files = {}
    master_df_input_files['installments_payments.pkl'] = get_pickle_gcp(bucket="wagon-data-618-le-banq", file_path=file_path)
    
    #Storing the data from google cloud storage
    df_copy = master_df_input_files['installments_payments.pkl']

    #Make a safety copy so we can run the function several times
    df_ip = df_copy.copy() 

    #Feature Engineering
    #LB - dataset transformation with aggregations to get with a smaller dataset

    #LB - Here want to know when a loan applicant (sk_id_curr) has for the same num_instalment_number 
    # and same sk_id_prev the max(days_entry_payment), sum AMT_PAYMENT and min days_installment. 
    # The AMT_INSTALMENT is always the same for the same num_instalment_number and SK_ID_PREV
    df_ip = df_ip.groupby(['SK_ID_PREV','SK_ID_CURR','NUM_INSTALMENT_NUMBER','AMT_INSTALMENT']).agg({'AMT_PAYMENT':'sum', 'DAYS_ENTRY_PAYMENT':'max', 'DAYS_INSTALMENT': 'min'}).reset_index()

    #LB - create two new columns: Late payments or ammount was missing
    df_ip['LATE_PAY'] = np.where(df_ip['DAYS_INSTALMENT'] < df_ip['DAYS_ENTRY_PAYMENT'],1,0)
    df_ip['AMT_MISS']= np.where(df_ip['AMT_PAYMENT'] < df_ip['AMT_INSTALMENT'],1,0)

    #LB - Now, that we have our new dataset, 
    #I want to know for each SK_ID_CURR the count of LATE_PAY and count AMT_MISS
    df_ip['COUNT_LATE_PAY'] = df_ip.groupby(['SK_ID_CURR'])['LATE_PAY'].transform('sum')
    df_ip['COUNT_AMT_MISS'] = df_ip.groupby(['SK_ID_CURR'])['AMT_MISS'].transform('sum')

    #Aggregate our dataset and SK_ID_CURR
    df_ip = df_ip.groupby(['SK_ID_CURR'])['AMT_PAYMENT','LATE_PAY','AMT_MISS' ].sum().reset_index()

    return df_ip

def get_bureau():
    #import the data
    file_path = 'raw_data/bureau.pkl'
    master_df_input_files = {}
    master_df_input_files['bureau.pkl'] = get_pickle_gcp(bucket="wagon-data-618-le-banq", file_path=file_path)
    
    #Storing the data from google cloud storage
    df_copy = master_df_input_files['bureau.pkl']
    
    #Make a safety copy so we can run the function several times
    df_bureau = df_copy.copy() 

    #Bring Bureau Balance dataset to merge with bureau
    df_bbalance = get_bureau_balance()

    #Merge bureau with bureau balance by SK_ID_BUREAU
    df_bureau = pd.merge(df_bureau, df_bbalance, on='SK_ID_BUREAU', how='left')

    # One-hot encoder,
    df_bureau, cat_cols = one_hot_encoder(df_bureau,  categorical_columns=['CREDIT_ACTIVE','CREDIT_CURRENCY','CREDIT_TYPE'],nan_as_category= False)

    # Because the functions one hot encode we are using changes the dtype of the columns
    #we are going to convert to float
    cols = df_bureau.select_dtypes(exclude=['float', 'uint8']).columns
    df_bureau[cols] = df_bureau[cols].apply(pd.to_numeric, downcast='float', errors='coerce')

    # Split the bureau data from Active/Closed
    active = df_bureau[df_bureau['CREDIT_ACTIVE_Active'] == 1]
    closed = df_bureau[df_bureau['CREDIT_ACTIVE_Closed'] == 1]

    ##### Active #####
    active_num_agg = {
                'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean'],
                'BAD_STATUS_COUNT': ['mean', 'max'],
                'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
                'CREDIT_DAY_OVERDUE':['mean', 'max'],
                'CNT_CREDIT_PROLONG': ['mean', 'max'],
                'AMT_CREDIT_SUM_DEBT': ['max', 'sum']
                }

    #Make a dictionary of the features for the aggregation
    # Categorical features
    active_cat_agg = {key: ['mean'] for key in cat_cols}
    #Numerical features
    #Group the two dictionaries
    ACTIVE_PREV_AGG = active_cat_agg.copy()
    ACTIVE_PREV_AGG.update(active_num_agg)

    df_active = active.groupby('SK_ID_CURR').agg(ACTIVE_PREV_AGG).reset_index()
    
    #Drop non-necessary columns
    df_active = df_active.drop(columns=['CREDIT_ACTIVE_Active','CREDIT_ACTIVE_Bad debt','CREDIT_ACTIVE_Closed','CREDIT_ACTIVE_Sold'])

    #Convert the multindex
    df_active.columns = ['_'.join(col) for col in df_active.columns.values]

    #add the suffix active to distinguish from the closed
    df_active = df_active.add_suffix('_active')

    #Make the SK_ID_CURR unique
    df_active = df_active.rename(columns={'SK_ID_CURR__active':'SK_ID_CURR'})

    ##### Closed #####
    closed_num_agg = {
                'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean'],
                'BAD_STATUS_COUNT': ['mean', 'max'],
                'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
                'CREDIT_DAY_OVERDUE':['mean', 'max'],
                'CNT_CREDIT_PROLONG': ['mean', 'max'],
                'AMT_CREDIT_SUM_DEBT': ['max', 'sum']
                }

    #Make a dictionary of the features for the aggregation
    # Categorical features
    closed_cat_agg = {key: ['mean'] for key in cat_cols}
    #Numerical features
    #Group the two dictionaries
    CLOSED_PREV_AGG = closed_cat_agg.copy()
    CLOSED_PREV_AGG.update(closed_num_agg)

    df_closed = closed.groupby('SK_ID_CURR').agg(CLOSED_PREV_AGG).reset_index()
    
    #Drop non-necessary columns
    df_closed = df_closed.drop(columns=['CREDIT_ACTIVE_Active','CREDIT_ACTIVE_Bad debt','CREDIT_ACTIVE_Closed','CREDIT_ACTIVE_Sold'])

    #Convert the multindex
    df_closed.columns = ['_'.join(col) for col in df_closed.columns.values]

    #add the suffix active to distinguish from the closed
    df_closed = df_closed.add_suffix('_closed')

    #Make the SK_ID_CURR unique
    df_closed = df_closed.rename(columns={'SK_ID_CURR__closed':'SK_ID_CURR'})

    ##Aggregate both closed and open to bureau dataset
    ###### CODE DOES NOT WORK FROM HERE DUE TO MULTINDEXING ######
    df_agg_bureau = pd.merge(df_active, df_closed, on='SK_ID_CURR', how='left')

    return df_agg_bureau

def get_bureau_balance():

    #import the data
    file_path = 'raw_data/bureau_balance.pkl'
    master_df_input_files = {}
    master_df_input_files['bureau_balance.pkl'] = get_pickle_gcp(bucket="wagon-data-618-le-banq", file_path=file_path)
    
    #Storing the data from google cloud storage
    df_copy = master_df_input_files['bureau_balance.pkl']
    
    #Make a safety copy so we can run the function several times
    df_bbalance = df_copy.copy() 

    #Calculate how many acumulated months does a person have good status, or bad status!!!!
    df_bbalance['Bad_Status'] = 0 
    df_bbalance['Bad_Status'] = np.where((df_bbalance['STATUS'] == '1') | (df_bbalance['STATUS'] == '2') | (df_bbalance['STATUS'] == '3') | (df_bbalance['STATUS'] == '4') | (df_bbalance['STATUS'] == '5'),1,0)
    
    #Group by SK_ID_BUREAU so it can be merged with bureau dataset
    df_bbalance = df_bbalance.groupby('SK_ID_BUREAU')['Bad_Status'].apply(lambda x: (x==1).sum()).reset_index(name='BAD_STATUS_COUNT')
    
    
    return df_bbalance