
from lz4 import frame

import os
import matplotlib.pylab as plt
from io import BytesIO
import pandas as pd
from const import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

def read_data(filename):
    chunk_list=[]#init
    #we will read the file chunk by chunk because it's too big
    with frame.open(filename, mode='r') as fp:
        chunk_size=1024*28*28
        output_data = fp.read(size=chunk_size)
        df_chunk=pd.read_csv(BytesIO(output_data))
        chunk_list.append(df_chunk)
    df = pd.concat(chunk_list)
    return df






def prepare_data(df_train,df_test):
  
    # define data
  scaler = MinMaxScaler()
    # transform data
  #training data
  df_train=df_train.drop(columns=['dataset_type'])
  df_train = df_train.dropna()
  df_train = df_train.drop_duplicates()


  from sklearn.datasets import make_regression
  from sklearn.feature_selection import SelectKBest
  from sklearn.feature_selection import f_regression

  #test set 
  df_test = df_test.dropna()
  df_test = df_test.drop_duplicates()
  df_valid=df_test.loc[df_test['sale_day_x'].isin(test_set)]
  y_test = df_test['demand']
  X_df_test=df_test[['sale_day_x','price','origin_station_name','destination_station_name']]
  X_df_test=X_df_test.dropna()

  #valdation set  
  y_valid=df_valid['demand']
  X_df_valid=df_valid[['sale_day_x','price','origin_station_name','destination_station_name']]

  X_df_valid=X_df_valid.dropna()

  #training set 
  X_df_train = df_train[['sale_day_x','price','origin_station_name','destination_station_name']]
  X_df_train= X_df_train.dropna()
  y_train = df_train['demand']

  #One-hot-encode categorical variables 
  cols = X_df_test.columns
  num_cols = X_df_test._get_numeric_data().columns
  cat_cols=list(set(cols) - set(num_cols))
  from sklearn.compose import ColumnTransformer
  ct = ColumnTransformer([('name', OneHotEncoder(sparse=False, handle_unknown='ignore'), cat_cols)], remainder = 'passthrough')
  X_df_train = ct.fit_transform(X_df_train)
  X_df_test = ct.transform(X_df_test)
  X_df_valid=ct.transform(X_df_valid)

  



  
  return X_df_train,X_df_test,y_train,y_test,X_df_valid,y_valid








