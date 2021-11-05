
#----------------------------------------------------------------------------
# Created By  : Omar KHATIB for Wiremind   Line 3
# Created Date: 05/11/2021 
# version ='1.0'
#This module can be used to instanciate the model, launch the training/testing , and try the predict_demand method
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------







from  data_prep import *
from model import *
import logging

#To log events in the hist.log
logging.basicConfig(filename=log_file,
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


try:
    
    #To prepare the datasets for the training process
    X_train,X_test,y_train,y_test,X_df_valid,y_valid,df_validation,ct=prepare_data(training_path,test_path)
    #instanciate the model
    ann_model=model(X_train.shape,ct)
    logging.info('Model created')
    #Launch the training of the model
    ann_model.launch_training(X_train,y_train,X_test,y_test)
    logging.info('training is over')
    #Evaluate the model with the [-89,-60,-30,-20,-15,-10,-7,-6,-5,-3,-2,-1]
    result=ann_model.evaluation_model(df_validation,y_valid)
    logging.info('model evaluated')
    result.to_csv(output_eval)

    #obtain a prediction of a demand for (day_x,price) for a train going from a station to another.
    predicted_demand=ann_model.predict_demand(-40,120,'rb','bb')
    print(predicted_demand)
    #[Optional] To visualize the accumulated demand for a particular train (use the data in eval.csv)
    ann_model.plot_accumul(df_validation,y_valid,'2020-10-18','rb','bb',360)

except ValueError as err:
    logging.info(err.args)





