from  data_prep import *
from evaluate import *
from model import *
from train import *
from validation import *





df_train=read_data('train.lz4')
df_test=read_data('test.lz4')
df_train = df_train.dropna()
df_train = df_train.drop_duplicates()
df_test = df_test.dropna()
df_test = df_test.drop_duplicates()

X_train,X_test,y_train,y_test,X_df_valid,y_valid=prepare_data(df_train,df_test)
ann_model=launch_training(X_train,y_train,X_test,y_test)
model_validation(ann_model,X_df_valid,y_valid)





