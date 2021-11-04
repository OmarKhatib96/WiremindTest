from model import *
from evaluate import *
from const import *

def launch_training(X_train,y_train,X_test,y_test):
    

    #instanciation of the model 
    ann_model=model(X_train)


    history = ann_model.fit(
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
    plt.savefig('./training-test.png')

    #performance measurement (mse,mae during training/test)
    perf_evaluator=build_evaluate_fn(X_train, y_train, X_test, y_test)
    perf_evaluator(ann_model)
    return ann_model
