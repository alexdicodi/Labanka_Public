from datetime import datetime
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from LABANKA_PUBLIC.utils import get_pickle_gcp, get_joblib_gcp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return dict(greeting="hello")


@app.get("/predict")
def predict(with_previous,
            CNT_CHILDREN,               # 3
            AMT_INCOME_TOTAL,           # 50000
            DAYS_BIRTH,                 # 2530
            DAYS_EMPLOYED,              # 270
            ORGANIZATION_TYPE_TEXT,          # Electricity (Label encoding)
            EXT_SOURCE2,                # floating € [0,1]
            EXT_SOURCE3,                # floating € [0,1]
            DEF_60_CNT_SOCIAL_CIRCLE,   # Default (integer number)
            AMT_REQ_CREDIT_BUREAU_DAY,  # Default (integer number)
            AMT_REQ_CREDIT_BUREAU_YEAR, # Default (integer number)
            NAME_INCOME_TYPE,           # text, OHE
            CURR_AMT_ANNUITY,           # Only for current application, floating 3500.9
            CURR_AMT_CREDIT,            # Only for current applications floating 10000.5
            OCCUPATION_TYPE,            # text OHE
            NAME_CONTRACT_TYPE,         # if with_previous =1, value from params, if not default (OHE)
            NAME_PORTFOLIO,             # if with_previous =1, value from params, if not default (OHE)
            NAME_CONTRACT_STATUS,       # if with_previous =1, value from params, if not default (OHE)
            AMT_ANNUITY,                # if with_previous =1, value from params, if not default 
            AMT_CREDIT,                 # if with_previous =1, value from params, if not default 
            APPLICATION_CREDIT_DIF,     # if with_previous =1, value from params, if not default 
            PREV_REFUSED_COUNT,         # if with_previous =1, value from params, if not default 
            PREV_APPROVED_COUNT,        # if with_previous =1, value from params, if not default 
            AMT_PAYMENT,                # if with_previous =1, value from params, if not default (OHE)
            LATE_PAY,                   # if with_previous =1, value from params, if not default (OHE)
            AMT_MISS):
    
#Pending to calculate: ANNUITY_OVER_CREDIT_RATIO, CREDIT_TO_ANNUITY_RATIO (prev)

    # Feature engineering
    if int(CURR_AMT_ANNUITY) == 0 or  int(CURR_AMT_CREDIT) == 0:
        ANNUITY_OVER_CREDIT_RATIO = 0
    else:
        ANNUITY_OVER_CREDIT_RATIO = int(CURR_AMT_ANNUITY) / int(CURR_AMT_CREDIT)
    
    if int(CURR_AMT_ANNUITY) == 0 or  int(CURR_AMT_CREDIT) == 0:
        CREDIT_TO_ANNUITY_RATIO = 0
    else:
        CREDIT_TO_ANNUITY_RATIO = int(CURR_AMT_CREDIT) / int(CURR_AMT_ANNUITY)

    """not used in the original preparation file
    #Loading the encoders
    get_current_app_OHE_processor = get_processor_gcp(bucket="wagon-data-618-le-banq", file_path='encoders/get_app_OHE_pipeline.pkl')
    get_current_app_ORD_processor = get_processor_gcp(bucket="wagon-data-618-le-banq", file_path='encoders/get_app_ORD_pipeline.pkl')
    prev_app_OHE_processor = get_processor_gcp(bucket="wagon-data-618-le-banq", file_path='encoders/get_prev_app_OHE_pipeline.pkl')
    prev_app_OHE_processor = get_processor_gcp(bucket="wagon-data-618-le-banq", file_path='encoders/get_prev_app_OHE_pipeline.pkl')
    """

    #Loading the master label encoder

    master_label_encoder_df = get_pickle_gcp(file_path='encoders/master_label_encoding.pkl')

    #Loading the dataframe column structure

    columns_df =  ['with_previous','CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH',
   'DAYS_EMPLOYED', 'ORGANIZATION_TYPE_TEXT', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
   'DEF_60_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_DAY',
   'AMT_REQ_CREDIT_BUREAU_YEAR', 'ANNUITY_OVER_CREDIT_RATIO',
   'NAME_INCOME_TYPE_Businessman', 'NAME_INCOME_TYPE_Commercial associate',
   'NAME_INCOME_TYPE_Maternity leave', 'NAME_INCOME_TYPE_Pensioner',
   'NAME_INCOME_TYPE_State servant', 'NAME_INCOME_TYPE_Student',
   'NAME_INCOME_TYPE_Unemployed', 'NAME_INCOME_TYPE_Working',
   'NAME_INCOME_TYPE_nan', 'OCCUPATION_TYPE_Accountants',
   'OCCUPATION_TYPE_Cleaning staff', 'OCCUPATION_TYPE_Cooking staff',
   'OCCUPATION_TYPE_Core staff', 'OCCUPATION_TYPE_Drivers',
   'OCCUPATION_TYPE_HR staff', 'OCCUPATION_TYPE_High skill tech staff',
   'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Laborers',
   'OCCUPATION_TYPE_Low-skill Laborers', 'OCCUPATION_TYPE_Managers',
   'OCCUPATION_TYPE_Medicine staff',
   'OCCUPATION_TYPE_Private service staff',
   'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Sales staff',
   'OCCUPATION_TYPE_Secretaries', 'OCCUPATION_TYPE_Security staff',
   'OCCUPATION_TYPE_Waiters/barmen staff', 'OCCUPATION_TYPE_nan',
   'NAME_CONTRACT_TYPE_Cash loans_mean',
   'NAME_CONTRACT_TYPE_Consumer loans_mean', 'NAME_CONTRACT_TYPE_POS_mean',
   'NAME_CONTRACT_TYPE_Revolving loans_mean', 'NAME_PORTFOLIO_Cards_mean',
   'NAME_PORTFOLIO_Cars_mean', 'NAME_PORTFOLIO_Cash_mean',
   'NAME_PORTFOLIO_POS_mean', 'NAME_CONTRACT_STATUS_Approved_mean',
   'NAME_CONTRACT_STATUS_Canceled_mean',
   'NAME_CONTRACT_STATUS_Refused_mean',
   'NAME_CONTRACT_STATUS_Unused offer_mean', 'AMT_ANNUITY_min',
   'AMT_ANNUITY_max', 'AMT_ANNUITY_mean', 'CREDIT_TO_ANNUITY_RATIO_mean',
   'CREDIT_TO_ANNUITY_RATIO_max', 'APPLICATION_CREDIT_DIFF_min',
   'APPLICATION_CREDIT_DIFF_max', 'APPLICATION_CREDIT_DIFF_mean',
   'PREV_REFUSED_COUNT_mean', 'PREV_APPROVED_COUNT_mean', 'AMT_PAYMENT',
   'LATE_PAY', 'AMT_MISS']

    #Creating an empty dataframe

    input_df = pd.DataFrame(columns=columns_df)

    # Preparing the dictionary to append to the empty dataframe
    
    df_input_append = {'with_previous': with_previous, 'CNT_CHILDREN': CNT_CHILDREN, 'AMT_INCOME_TOTAL': AMT_INCOME_TOTAL, 'DAYS_BIRTH': DAYS_BIRTH,
       'DAYS_EMPLOYED': DAYS_EMPLOYED, 'ORGANIZATION_TYPE_TEXT': ORGANIZATION_TYPE_TEXT,'EXT_SOURCE_2': EXT_SOURCE2, 'EXT_SOURCE_3': EXT_SOURCE3,
       'DEF_60_CNT_SOCIAL_CIRCLE': DEF_60_CNT_SOCIAL_CIRCLE, 'AMT_REQ_CREDIT_BUREAU_DAY': AMT_REQ_CREDIT_BUREAU_DAY,
       'AMT_REQ_CREDIT_BUREAU_YEAR': AMT_REQ_CREDIT_BUREAU_YEAR, 'ANNUITY_OVER_CREDIT_RATIO': ANNUITY_OVER_CREDIT_RATIO,
       'NAME_INCOME_TYPE_Businessman':0, 'NAME_INCOME_TYPE_Commercial associate':0,
       'NAME_INCOME_TYPE_Maternity leave':0, 'NAME_INCOME_TYPE_Pensioner':0,
       'NAME_INCOME_TYPE_State servant':0, 'NAME_INCOME_TYPE_Student':0,
       'NAME_INCOME_TYPE_Unemployed':0, 'NAME_INCOME_TYPE_Working':0,
       'NAME_INCOME_TYPE_nan':0, 'OCCUPATION_TYPE_Accountants':0,
       'OCCUPATION_TYPE_Cleaning staff':0, 'OCCUPATION_TYPE_Cooking staff':0,
       'OCCUPATION_TYPE_Core staff':0, 'OCCUPATION_TYPE_Drivers':0,
       'OCCUPATION_TYPE_HR staff':0, 'OCCUPATION_TYPE_High skill tech staff':0,
       'OCCUPATION_TYPE_IT staff':0, 'OCCUPATION_TYPE_Laborers':0,
       'OCCUPATION_TYPE_Low-skill Laborers':0, 'OCCUPATION_TYPE_Managers':0,
       'OCCUPATION_TYPE_Medicine staff':0,
       'OCCUPATION_TYPE_Private service staff':0,
       'OCCUPATION_TYPE_Realty agents':0, 'OCCUPATION_TYPE_Sales staff':0,
       'OCCUPATION_TYPE_Secretaries':0, 'OCCUPATION_TYPE_Security staff':0,
       'OCCUPATION_TYPE_Waiters/barmen staff':0, 'OCCUPATION_TYPE_nan':0,
       'NAME_CONTRACT_TYPE_Cash loans_mean':0,
       'NAME_CONTRACT_TYPE_Consumer loans_mean':0, 'NAME_CONTRACT_TYPE_POS_mean':0,
       'NAME_CONTRACT_TYPE_Revolving loans_mean':0, 'NAME_PORTFOLIO_Cards_mean':0,
       'NAME_PORTFOLIO_Cars_mean':0, 'NAME_PORTFOLIO_Cash_mean':0,
       'NAME_PORTFOLIO_POS_mean':0, 'NAME_CONTRACT_STATUS_Approved_mean':0,
       'NAME_CONTRACT_STATUS_Canceled_mean':0,
       'NAME_CONTRACT_STATUS_Refused_mean':0,
       'NAME_CONTRACT_STATUS_Unused offer_mean': 0, 'AMT_ANNUITY_min': AMT_ANNUITY,
       'AMT_ANNUITY_max': AMT_ANNUITY, 'AMT_ANNUITY_mean': AMT_ANNUITY, 'CREDIT_TO_ANNUITY_RATIO_mean': CREDIT_TO_ANNUITY_RATIO,
       'CREDIT_TO_ANNUITY_RATIO_max': CREDIT_TO_ANNUITY_RATIO, 'APPLICATION_CREDIT_DIFF_min': APPLICATION_CREDIT_DIF,
       'APPLICATION_CREDIT_DIFF_max': APPLICATION_CREDIT_DIF, 'APPLICATION_CREDIT_DIFF_mean': APPLICATION_CREDIT_DIF,
       'PREV_REFUSED_COUNT_mean': PREV_REFUSED_COUNT, 'PREV_APPROVED_COUNT_mean': PREV_APPROVED_COUNT, 'AMT_PAYMENT': AMT_PAYMENT,
       'LATE_PAY': LATE_PAY, 'AMT_MISS': AMT_MISS}

    df = input_df.append(df_input_append, ignore_index = True)

    # We will now iterate over each of the OHE elements and assign 1 to the one that matches clients input

    ## Name Income Type

    name_income_type_cols = [col for col in df.columns if 'NAME_INCOME_TYPE_' in col]

    for i in name_income_type_cols:
        if NAME_INCOME_TYPE in i: 
            df[i]=1
        else: 
            None

    ## OCCUPATION_TYPE

    occupation_type_cols = [col for col in df.columns if 'OCCUPATION_TYPE_' in col]

    for i in occupation_type_cols:
        if OCCUPATION_TYPE in i: 
            df[i]=1
        else: 
            None

    ## NAME_CONTRACT_TYPE

    name_contract_type_cols = [col for col in df.columns if 'NAME_CONTRACT_TYPE_' in col]

    for i in name_contract_type_cols:
        if NAME_CONTRACT_TYPE in i: 
            df[i]=1
        else: 
            None
        
    ## NAME_PORTFOLIO

    name_portfolio_cols = [col for col in df.columns if 'NAME_PORTFOLIO_' in col]

    for i in name_portfolio_cols:
        if NAME_PORTFOLIO in i: 
            df[i]=1
        else: 
            None

    ## NAME_CONTRACT_STATUS

    name_contract_status_cols = [col for col in df.columns if 'NAME_CONTRACT_STATUS_' in col]

    for i in name_contract_status_cols:
        if NAME_CONTRACT_STATUS in i: 
            df[i]=1
        else: 
            None
    
    ## Merging with the master label encoding df

    clean_df = df.merge(master_label_encoder_df,on='ORGANIZATION_TYPE_TEXT',how='inner')
    clean_df.drop(columns='ORGANIZATION_TYPE_TEXT', inplace=True)

    # pipeline = get_model_from_gcp() OKEI
    model = get_joblib_gcp(bucket="wagon-data-618-le-banq", file_path='model/no_bureau_model_lgbm.sav')

    # make prediction
    results = model.predict(clean_df)

    # convert response from numpy to python type
    pred = float(results[0])

    return dict(prediction=pred)
    
# $DELETE_END