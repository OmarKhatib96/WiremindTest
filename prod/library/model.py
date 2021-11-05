

#----------------------------------------------------------------------------
# Created By  : Omar KHATIB for Wiremind   Line 3
# Created Date: 05/11/2021 
# version ='1.0'
#This module contains the model class and all its method
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------













from datetime import date
from library.const import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf 
#custome mean absolute error
def mae( g,col1,col2 ):
        mae = mean_absolute_error( g[col1], g[col2] )
        return pd.Series( dict(  mae = mae ) )

class model:

    def __init__(self,shape,ct):
        self.init_model(shape)
        self.ct=ct

    def init_model(self,shape):
        self.deep_model= Sequential()
        self.deep_model.add(Dense(units=51, input_shape=(shape[1],)))
        self.deep_model.add(Dropout(DO))
        self.deep_model.add(Dense(units=18,activation='relu'))
        self.deep_model.add(Dense(units=1))
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.deep_model.compile(optimizer = opt, loss = 'mse')

    #this method will launch the training (& and the testing as well) the plot of the loss and val_loss is saved as .png file
    def launch_training(self,X_train,y_train,X_test,y_test):
        history = self.deep_model.fit(
            X_train,
            y_train,
            batch_size=batchsize,
            epochs=nbr_epochs,
            validation_data=(X_test, y_test),
        )

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        plt.savefig(plots+'/training-test.png')

        #performance measurement (mse,mae during training/test)
        self.build_evaluate_fn(X_train, y_train, X_test, y_test)

    def compute_mse(self, X, y_true, name):
        y_pred = self.deep_model.predict(X)
        mse = mean_squared_error(y_true, y_pred)
        mae=mean_absolute_error(y_true, y_pred)
        
        print(f'Mean Squared Error for {name}: {mse}')
        print(f'Mean Absolute Error for {name}: {mae}')
        f = open(plots+"./training_results.txt", "w")
        f.write(f'Mean Squared Error for {name}: {mse}')
        f.write(f'Mean Absolute Error for {name}: {mae}')
        f.close()        

    #taken from stackoverflow
    def build_evaluate_fn(self,X_train, y_train, X_test, y_test):
        self.compute_mse( X_train, y_train, 'training set')
        self.compute_mse( X_test, y_test, 'test set')


    #This method predicts a demand for the tuple (sale_day_x,price,origin_station_name,destination_station_name)
    def predict_demand(self,sale_day_x,price,origin_station_name,destination_station_name):
        input=np.array([sale_day_x,price,origin_station_name,destination_station_name])
        df = pd.DataFrame(data=[input], columns=["sale_day_x", "price",'origin_station_name','destination_station_name'])
        #One-hot-encode categorical variables 
        df_encoded=self.ct.transform(df)
        demand_predicted=self.deep_model.predict(df_encoded)
        return round(max(0,demand_predicted[0][0]))



    #this method will evaluate the model by given for each train the mean average error of the accumulated demand over the last -90 days
    def evaluation_model(self,df_validation,y_valid):
        predicted=[]
        for i  in df_validation.index:
            demand_predicted=self.predict_demand(df_validation.loc[i, "sale_day_x"], df_validation.loc[i, "price"],df_validation.loc[i, "origin_station_name"],df_validation.loc[i, "destination_station_name"])
            #print('the predicted demand for the traing going from '+df_validation.loc[i, "origin_station_name"]+' to '+df_validation.loc[i, "destination_station_name"]+ 'is ',demand_predicted)
            predicted.append(demand_predicted)

        df_predicted = pd.DataFrame(predicted, columns=['predicted_demand'])
        df_true_demand=pd.DataFrame(y_valid.values, columns=['true_demand'])
        df_true_demand.reset_index(drop=True, inplace=True)
        df_predicted.reset_index(drop=True, inplace=True)
        df_validation.reset_index(drop=True, inplace=True)

        concatenated = pd.concat([df_validation,df_predicted, df_true_demand],axis=1)
        concatenated[['predicted_demand_cumul','true_demand_cumul']]=concatenated.groupby(['od_origin_time','departure_date','origin_station_name','destination_station_name'])[['predicted_demand','true_demand']].apply(lambda x: x.cumsum())

        self.full_stat=concatenated
        evaluation_df=concatenated.groupby(['od_origin_time','departure_date','origin_station_name','destination_station_name'])[['predicted_demand_cumul','true_demand_cumul']].apply( mae,'predicted_demand_cumul','true_demand_cumul' ).reset_index()
        print(evaluation_df)
        return evaluation_df
        

    #This method will plot the cumulated demand for a particular train over time(day_x)
    def plot_accumul(self,df_validation,y_valid,date_depart,origin_station,destination_station,od_origin_time):
        evaluation_df=self.evaluation_model(df_validation,y_valid)
         #plot evaluations
        plt.figure(figsize=(60, 10)) 
        plt.xticks(
            rotation=45, 
            horizontalalignment='right',
            fontweight='light',
            fontsize='small')
        plt.gca().xaxis.grid(True)
        evaluation_df = self.full_stat.loc[(self.full_stat['destination_station_name'] ==destination_station ) & (self.full_stat['origin_station_name'] ==origin_station ) &  (self.full_stat['departure_date'] ==date_depart) & (self.full_stat['od_origin_time'] ==od_origin_time)]
        plt.plot( evaluation_df['sale_day_x'],evaluation_df['true_demand_cumul'],label='observed')
        plt.plot(evaluation_df['sale_day_x'],np.round( evaluation_df['predicted_demand_cumul']),label="predicted")
        plt.title('Accumulated demand over time for the train going from '+origin_station+' to '+destination_station+' at the date '+date_depart+' at '+str(int(od_origin_time/60))+'h'+str(od_origin_time%60))
        plt.legend(loc="upper left")
        plt.show()
        plt.savefig(plots+'/cumulated_predicted_demand'+date_depart+origin_station+destination_station+str(od_origin_time)+'.png')




