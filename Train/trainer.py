from datetime import datetime
from re import X
import pytz
import pandas as pd
import joblib
import pickle
from data_engine.preparation import main_dataset, get_bureau, get_prev_application
import os
from datalab.context import Context
import datalab.storage as storage
from io import BytesIO
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score
from lightgbm import LGBMClassifier
from LABANKA_PUBLIC.utils import   one_hot_encoder, label_encoder, get_pickle_gcp, missing_data, get_joblib_gcp, get_processor_gcp, storage_upload
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

class Trainer(object):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
    
    def run(self):
        ##Instantiating the model
        self.lgbm = LGBMClassifier(boosting_type='goss', objective='binary', n_estimators= 10000, random_state=5, is_unbalance =True, verbose=-1)
        self.lgbm.fit(self.X, self.y)
    
    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.lgbm.predict(X_test)
        auc_score = roc_auc_score(y_test, y_pred) 
        return round(auc_score, 2)
    
    def save_model_locally(self, model_name = "no_bureau_model_lgbm.sav"):
        """Save the model into a .sav format
        :param model_name: Name of the model
        """
        joblib.dump(self.lgbm, model_name)
        print(f"{model_name} saved locally")
    


if __name__ == "__main__":
    
    #Data Loading & Preparation
    ## Get the clean and preprocessed data
    no_bureau_df = get_pickle_gcp(file_path='dataframes/no_bureau_df.pkl')
    bureau_df = get_pickle_gcp(file_path='dataframes/bureau_df.pkl')

    ## Data Prep for the model
    ### Preparing the independent and dependent variables
    #### No bureau_df
    X_nb = no_bureau_df.drop(columns='TARGET')
    y_nb = no_bureau_df['TARGET']
    #### bureau_df
    X_b = bureau_df.drop(columns='TARGET')
    y_b = bureau_df['TARGET']

    ## Train Test Split
    ## No Bureau
    X_nb_train, X_nb_test, y_nb_train, y_nb_test = train_test_split(X_nb, y_nb, test_size=0.3)
    ## Bureau
    X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_b, y_b, test_size=0.3)

    #Model
    ## No Bureau
    no_bureau_trainer = Trainer(X=X_nb_train, y=y_nb_train)
    no_bureau_trainer.run()
    no_bureau_auc = no_bureau_trainer.evaluate(X_nb_test, y_nb_test)
    print(f"no bureau auc: {no_bureau_auc}")
    no_bureau_trainer.save_model_locally(model_name = "no_bureau_model_lgbm.sav")
    storage_upload(rm=False, BUCKET_NAME="wagon-data-618-le-banq", MODEL_NAME = "no_bureau_model_lgbm", MODEL_VERSION = "v1", local_model_file_path='no_bureau_model_lgbm.sav')

    ## Bureau
    bureau_trainer = Trainer(X=X_b_train, y=y_b_train)
    bureau_trainer.run()
    bureau_auc = bureau_trainer.evaluate(X_b_test, y_b_test)
    print(f"bureau auc: {bureau_auc}")
    bureau_trainer.save_model_locally(model_name = "bureau_model_lgbm.sav")
    storage_upload(rm=False, BUCKET_NAME="wagon-data-618-le-banq", MODEL_NAME = "bureau_model_lgbm", MODEL_VERSION = "v1", local_model_file_path='bureau_model_lgbm.sav')