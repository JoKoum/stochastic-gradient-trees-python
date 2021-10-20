from StochasticGradientTree import StochasticGradientTreeRegressor

import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

if __name__ == '__main__':

    X, y = load_linnerud(return_X_y=True)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    estimator = StochasticGradientTreeRegressor()

    clf = MultiOutputRegressor(estimator)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34)

    start = time.process_time()
    clf.fit(X_train,y_train)
    print('Time taken: {}s'.format(time.process_time() - start))
    
    y_pred = clf.predict(X_test)

    print('MSE: ', mean_squared_error(y_test, y_pred))
    print('MAE: ', mean_absolute_error(y_test, y_pred))
