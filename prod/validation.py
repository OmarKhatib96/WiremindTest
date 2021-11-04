



from train import *
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


def model_validation(ann_model,X_df_valid,y_valid):
    y_pred=ann_model.predict(X_df_valid)
    data = {'sale_day_x':X_df_valid[:len(test_set),X_df_valid.shape[1]-2],'price':X_df_valid[:len(test_set),X_df_valid.shape[1]-1],
            'demand_predicted':y_pred[:len(test_set)].flatten()}

    df_result_predicted = pd.DataFrame(data)
    df_result_predicted.reset_index(inplace=True)

    data ={'sale_day_x':X_df_valid[:len(test_set),X_df_valid.shape[1]-2],'price':X_df_valid[:len(test_set),X_df_valid.shape[1]-1],
            'true_demand':y_valid[:len(test_set)]}

    df_result_true = pd.DataFrame(data)
    df_result_true.reset_index(inplace=True)

    import numpy as np

    #cumulated demandes
    df_result_true['cumulated_true_demand'] = np.cumsum(df_result_true['true_demand'].values)
    df_result_predicted['cumulated_predicted_demand'] =np.cumsum(df_result_predicted['demand_predicted'].values)

    print("les perf  des demandes accumulées sont:")
    print('mae= ',mean_absolute_error(df_result_true['cumulated_true_demand'], df_result_predicted['cumulated_predicted_demand']))
    #print(mean_absolute_percentage_error(df_result_true['cumulated_true_demand'],np.round( df_result_predicted['cumulated_predicted_demand'])))

    #plot evaluations
    plt.figure(figsize=(60, 10)) 
    plt.xticks(
        rotation=45, 
        horizontalalignment='right',
        fontweight='light',
        fontsize='small')
    plt.gca().xaxis.grid(True)
    plt.plot( test_set,df_result_true['cumulated_true_demand'],label='observed')
    plt.plot(test_set,np.round( df_result_predicted['cumulated_predicted_demand']),label="predicted")
    plt.legend(loc="upper left")
    plt.show()
    plt.savefig('./cumulated_predicted_demand.png')
