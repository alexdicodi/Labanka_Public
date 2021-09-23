from os import EX_SOFTWARE
import pandas as pd
import math
import numpy as np
from pathlib import Path
import os
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datalab.context import Context
from datetime import date, datetime
import datalab.storage as storage
import pandas as pd
from io import BytesIO
import pickle
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Beta version - Custom OHE and CustomColunTransformer

class CustomOHE1(OneHotEncoder):
    def transform(self, *args, **kwargs):
        return pd.DataFrame(super().transform(*args, **kwargs), columns=self.get_feature_names())

class CustomOrdinalEncoder(OrdinalEncoder):
    def transform(self, *args, **kwargs):
        return pd.DataFrame(super().transform(*args, **kwargs), columns=self.categories_)

class CustomColumnTransformer(ColumnTransformer):
    def transform(self, *args, **kwargs):
        return pd.DataFrame(super().transform(*args, **kwargs), columns=self.get_feature_names())
    def fit_transform(self, *args, **kwargs):
        return pd.DataFrame(super().fit_transform(*args, **kwargs), columns=self.get_feature_names())

class PandasColumnTransformer(ColumnTransformer):
    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=X.columns, index=X.index)

    def fit_transform(self, X, y=None):
        return pd.DataFrame(super().fit_transform(X), columns=X.columns, index=X.index)

## Original 

def one_hot_encoder(df, categorical_columns=None, nan_as_category=True):
    """Create a new column for each categorical value in categorical columns. """
    original_columns = list(df.columns)
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    categorical_columns = [c for c in df.columns if c not in original_columns]
    return df, categorical_columns

def label_encoder(df, categorical_columns=None):
    """Encode categorical values as integers (0,1,2,3...) with pandas.factorize. """
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in categorical_columns:
        df[col], uniques = pd.factorize(df[col])
    return df, categorical_columns


# DF loading, optimization and cleaning

def df_optimized(df, verbose=True, **kwargs):
    """
    Reduces size of dataframe by downcasting numeircal columns
    :param df: input dataframe
    :param verbose: print size reduction if set to True
    :param kwargs:
    :return: df optimized
    """
    in_size = df.memory_usage(index=True).sum()
    # Optimized size here
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)
            if type == "float":
                df[col] = pd.to_numeric(df[col], downcast="integer")
    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100
    GB = out_size / 1000000000
    if verbose:
        print("optimized size by {} % | {} GB".format(ratio, GB))
    return df

# GCP Utils

## Download

def get_pickle_gcp(bucket="wagon-data-618-le-banq", file_path='raw_data/HomeCredit_columns_description.pkl'):
    """
    Transforms pickle files stored in GCP Storage into Dataframes
    :param bucket: Name of the wished GCP bucket
    :param file_path: File path inside the GCP Bucket
    :return: dataframe
    """
    base_bucket = storage.Bucket(bucket)
    remote_pickle = base_bucket.item(file_path).read_from()
    df = pd.read_pickle(BytesIO(remote_pickle))
    return df

def get_joblib_gcp(bucket="wagon-data-618-le-banq", file_path='model/no_bureau_model.sav'):
    """
    Loads joblib files stored in GCP Storage
    :param bucket: Name of the wished GCP bucket
    :param file_path: File path inside the GCP Bucket
    :return: model
    """
    base_bucket = storage.Bucket(bucket)
    remote_pickle = base_bucket.item(file_path).read_from()
    model = joblib.load(BytesIO(remote_pickle))
    return model

def get_processor_gcp(bucket="wagon-data-618-le-banq", file_path='get_app_OHE_pipeline.pkl'):
    """
    Loads processor joblib files stored in GCP Storage
    :param bucket: Name of the wished GCP bucket
    :param file_path: File path inside the GCP Bucket
    :return: model
    """
    base_bucket = storage.Bucket(bucket)
    remote_pickle = base_bucket.item(file_path).read_from()
    processor = joblib.load(BytesIO(remote_pickle))
    return processor

## Upload

def storage_upload(rm=False, BUCKET_NAME="wagon-data-618-le-banq", MODEL_NAME = "no_bureau_model", MODEL_VERSION = "v1", local_model_file_path='no_bureau_model.sav'):
    """
    Uploads the local model to GCP Storage
    :param rm: Delete local model from local storage
    :param BUCKET_NAME: Name of the wished GCP bucket
    :param MODEL_VERSION: Version of the model
    :param local_model_file_path: File path of the local model
    :return: model
    """
    client = storage.Client().bucket(BUCKET_NAME)
    storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_file_path}"
    blob = client.blob(storage_location)
    blob.upload_from_filename(local_model_file_path)
    print(f"=> model.joblib uploaded to bucket {BUCKET_NAME} inside {storage_location}")
    if rm:
        os.remove(local_model_file_path)

# DF cleaning

def none_norm(source):
    res = pd.DataFrame()
    mul = 100000
    for i in range(1, math.ceil(source.shape[0]/mul)+1):
        df = source[(i-1)*mul:i*mul]
        for col, dtype in dict(df.dtypes).items():
            if dtype is np.dtype('O') or dtype == 'object':
                df = df.astype({col: 'str'})
                df.loc[df[col].str.lower().isin(['nan','nat', 'none', 'null','XNA', 'XAP']), col] = None
            if dtype is np.dtype('float64') or dtype is np.dtype('int64'):
                df.loc[df[col].isnull(), col] = None
        res = res.append(df)
    return res

def replace_none_values(df, string_values_to_replace, numeric_values_to_replace, **kwargs):
    """
    Replaces nan-like values for nan
    :param df: input dataframe
    :param string_values_to_replace: list of string elements to be replaced by nan
    :param numeric_values_to_replace: list of numeric elements to be replaced by nan
    :param kwargs:
    :return: clean df
    """
    #Replacing string values
    df.replace(string_values_to_replace, np.nan, inplace= True)
    #Replacing numeric values
    df.replace(numeric_values_to_replace, np.nan, inplace= True)
    return df

def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total_Null', 'Missing_Percent'])

def drop_columns(dataframe, column_list):
    """Will drop the columns given in the column list"""
    dataframe.drop(columns=column_list, inplace=True)
    return dataframe

# File imports to dataframes

def get_file_names(levels_above_current_dir = 1, data_folder_name = 'raw_data', extension='xlsx'):
        """
        This function returns a dict with the path of the data folder and the names of the files that contain a given extension
        """
        # Hints: Build csv_path as "absolute path" in order to call this method from anywhere.
        # Do not hardcode your path as it only works on your machine ('Users/username/code...')
        # Use __file__ as absolute path anchor independant of your computer
        # Make extensive use of `import ipdb; ipdb.set_trace()` to investigate what `__file__` variable is really
        # Use os.path library to construct path independent of Unix vs. Windows specificities
        
        file_names_dict = {}
        """ 
        We will find the current working directory, 
        then go back to the main directory and finally add the data folder name
        """
        p = Path(os.path.abspath('')).parents[levels_above_current_dir-1]
        filename = os.path.abspath(os.path.join(p, data_folder_name))
        """ 
        We will now create the list
        """
        ## We will store the names in the list file_names
        file_names = [file for file in listdir(filename) if file.endswith('.{extension}'.format(extension=extension))]
        file_names_dict[filename]=file_names
        return file_names_dict
    
def get_dfs(file_names_dict, extension='xlsx'):
        """
        We will loop through the keys to create the 
        data dictionary with the keys and the df loaded from the csv files
        """
        folder_path =  list(file_names_dict.keys())[0]
        # initialize dictionary
        data = {}

        #Iterate through the dictionary and assign the df
        if extension == 'csv':
            for i in list(file_names_dict.values())[0]:
                data[i] = pd.read_csv(os.path.join(folder_path, i))
        
        elif extension == 'xlsx' or extension == 'xls':
            for i in list(file_names_dict.values())[0]:
                data[i] = pd.read_csv(os.path.join(folder_path, i))
        
        elif extension == 'json':
            for i in list(file_names_dict.values())[0]:
                data[i] = pd.read_json(os.path.join(folder_path, i))
        elif extension == 'pkl':
            for i in list(file_names_dict.values())[0]:
                data[i] = pd.read_pickle(os.path.join(folder_path, i))
        else:
            None
        return data
    
# Graphical Tools

def plot_stats(dataframe, feature,label_rotation=False,horizontal_layout=True):
    temp = dataframe[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = dataframe[[feature, 'TARGET']].groupby([feature],as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(28,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(28,14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x = feature, y="Number of contracts",data=df1)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    s = sns.barplot(ax=ax2, x = feature, y='TARGET', order=cat_perc[feature], data=cat_perc)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    return plt.show()
   
def plot_distribution(var, df):
    
    i = 0
    t1 = df.loc[df['TARGET'] != 0]
    t0 = df.loc[df['TARGET'] == 0]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(2,2,figsize=(12,12))

    for feature in var:
        i += 1
        plt.subplot(2,2,i)
        sns.kdeplot(t1[feature], bw=0.5,label="TARGET = 1")
        sns.kdeplot(t0[feature], bw=0.5,label="TARGET = 0")
        plt.ylabel('Density plot', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.show()

# Date conversion

def convert_date(somedate):
    today = date.today()
    converted_date = datetime.strptime(str(somedate),'%Y-%m-%d').date()
    diff = converted_date - today
    return diff.days