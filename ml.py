from lightgbm import LGBMRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.svm import LinearSVR


import pandas as pd
import seaborn as sns

from lz4 import frame

import os
import lz4
import matplotlib.pylab as plt
from io import BytesIO




def read_data(filename):
    chunk_list=[]#init
    #we will read the file chunk by chunk because it's too big
    with frame.open('train.lz4', mode='r') as fp:
        chunk_size=1024*28*28
        output_data = fp.read(size=chunk_size)
        df_chunk=pd.read_csv(BytesIO(output_data))
        chunk_list.append(df_chunk)
    df = pd.concat(chunk_list)
    return df



def prepare_data(df_train,df_test):
  
  #training data
  df_train=df_train.drop(columns=['dataset_type','sale_year','od_origin_year','origin_days_to_next_school_holiday'])
  df_train = df_train.dropna()

  df_train['sale_date'] = pd.to_datetime(df_train['sale_date'])
  df_train['departure_date'] = pd.to_datetime(df_train['departure_date'])
  df_train['origin_station_name'] = df_train['origin_station_name'].astype("|S")
  df_train['destination_station_name'] = df_train['destination_station_name'].astype("|S")
  df_train = df_train.drop_duplicates()
  X_df_train = df_train.drop(columns=['demand'])
  y_train = df_train['demand'].values
  #testing data
  df_test=df_test.drop(columns=['dataset_type','sale_year','od_origin_year','origin_days_to_next_school_holiday'])
  df_test['sale_date'] = pd.to_datetime(df_train['sale_date'])
  df_test['departure_date'] = pd.to_datetime(df_train['departure_date'])
  df_test['origin_station_name'] = df_test['origin_station_name'].astype("|S")
  df_test['destination_station_name'] = df_test['destination_station_name'].astype("|S")
  df_test = df_test.dropna()
  df_test = df_test.drop_duplicates()
  y_test = df_test['demand'].values
  print('df_test before:',df_test.shape)
  X_df_test = df_test.drop(columns=['demand'])
  print('df_test after:',X_df_test.shape)

  print('before:',df_test['demand'].shape)
  print(y_test.shape)
  return X_df_train,X_df_test,y_train,y_test


def compute_mse(model, X, y_true, name):
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    print(f'Mean Squared Error for {name}: {mse}')


def build_evaluate_fn(X_train, y_train, X_test, y_test):
    def evaluate(model):
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print("Train Score:", train_score)
        print("Test Score:", test_score)
        print()    
        compute_mse(model, X_train, y_train, 'training set')
        compute_mse(model, X_test, y_test, 'test set')
    return evaluate




df_train=read_data('train.lz4')
df_test=read_data('test.lz4')


X_df_train,X_df_test,y_train,y_test=prepare_data(df_train,df_test)

print(X_df_train)
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(X_df_train.values)
X_train=encoder.transform(X_df_train.values)


X_test = encoder.transform(X_df_test.values)





from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
fs2 = SelectKBest(score_func=chi2, k=340)
fs2.fit(X_train, y_train)
X_train_fs = fs2.transform(X_train)
X_test_fs = fs2.transform(X_test)




#@title
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
xgb=XGBClassifier(n_estimators=25)
xgb.fit(X_train_fs,y_train)
y_pred=xgb.predict(X_test_fs)
print('test accuracy:',round(accuracy_score(y_test,y_pred),3))



from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_jobs=3,random_state=40)
rf.fit(X_train_fs,y_train)
y_pred=rf.predict(X_test_fs)
print('test accuracy:',round(accuracy_score(y_test,y_pred),3))


evaluate=build_evaluate_fn(X_train_fs, y_train, X_test_fs, y_test)
evaluate(rf)


#plots preparations

print(df_test)

data = {'sale_day_x':X_df_test['sale_day_x'],'price':X_df_test['price'],
        'true_demand':y_test}

df_result_true = pd.DataFrame(data)
df_result_true.reset_index(inplace=True)





data = {'sale_day_x':X_df_test['sale_day_x'],'price':X_df_test['price'],
        'demand_predicted':y_pred}

df_result_predicted = pd.DataFrame(data)
df_result_predicted.reset_index(inplace=True)





#plot evaluations
plt.figure(figsize=(30, 6)) 
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='small')
plt.gca().xaxis.grid(True)

ax=sns.lineplot(x = 'sale_day_x',y = 'true_demand', data=df_result_true)
ax.legend('true_demand')
#sns.lineplot(x = 'sale_day_x',y = 'price', data=demand_by_day).set(title='Demand and price evolutions as a function of sale_day_x')
ax2 = plt.twinx()
ax3=sns.lineplot(x='sale_day_x',y='demand_predicted',data=df_result_predicted, color="b", ax=ax2)
ax2.legend('predicted_demand')

plt.show()

#demand as a function of price
plt.figure(figsize=(30, 6)) 
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='small')
plt.gca().xaxis.grid(True)

ax=sns.lineplot(x = 'price',y = 'true_demand', data=df_result_true)
ax.legend('true demand')
#sns.lineplot(x = 'sale_day_x',y = 'price', data=demand_by_day).set(title='Demand and price evolutions as a function of sale_day_x')
ax2 = plt.twinx()
ax3=sns.lineplot(x='price',y='demand_predicted',data=df_result_predicted, color="g", ax=ax2)
ax2.legend('predicted price')

plt.show()




'''

#svr part 
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error, explained_variance_score, max_error

print(f'Model fit results:\n'
      f'r2_score {r2_score(Y_test, predictions)} \t MSE {mean_squared_error(Y_test, predictions)}'
      f'\tEVS {explained_variance_score(y_test, predictions)} \n MAE {mean_absolute_error(Y_test, predictions)}'
      f'\tMAD {median_absolute_error(y_test, predictions)}\t ME {max_error(y_test, predictions)}')





data = {'sale_day_x':df_test['sale_day_x'],'price':df_test['price'],
        'demand_predicted':predictions}

df_result_svr = pd.DataFrame(data)
df_result_svr.reset_index(inplace=True)



plt.figure(figsize=(30, 6)) 
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='small')
plt.gca().xaxis.grid(True)

ax=sns.lineplot(x = 'sale_day_x',y = 'true_demand', data=df_result_true)
ax.legend('true demand')
#sns.lineplot(x = 'sale_day_x',y = 'price', data=demand_by_day).set(title='Demand and price evolutions as a function of sale_day_x')
ax2 = plt.twinx()
ax3=sns.lineplot(x='sale_day_x',y='demand_predicted',data=df_result_svr, color="b", ax=ax2)
ax2.legend('predicted price')


evaluate = build_evaluate_fn(X_train_fs, y_train, X_test_fs, Y_test)

evaluate(gbr)
'''