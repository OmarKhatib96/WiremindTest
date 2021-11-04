
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def compute_mse(model, X, y_true, name):
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    mae=mean_absolute_error(y_true, y_pred)
    print(f'Mean Squared Error for {name}: {mse}')
    print(f'Mean Absolute Error for {name}: {mae}')


def build_evaluate_fn(X_train, y_train, X_test, y_test):
    def evaluate(model):
        compute_mse(model, X_train, y_train, 'training set')
        compute_mse(model, X_test, y_test, 'test set')
    return evaluate


