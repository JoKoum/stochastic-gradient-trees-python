from StochasticGradientTree import StochasticGradientTreeRegressor

import time
import math

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34)
        
    tree = StochasticGradientTreeRegressor()

    start = time.process_time()
    tree.fit(X_train, y_train)
    print('Time taken: ', time.process_time() - start)
    
    y_pred = tree.predict(X_test)        

    r2_test = r2_score(y_test, y_pred)

    print('MSE: ', mean_squared_error(y_test, y_pred))
    print('MAE: ', mean_absolute_error(y_test, y_pred))
    print('R2 test: ', r2_test)
    print('Correlation coefficient :', math.sqrt(abs(r2_test)))

    return tree, r2_test

if __name__ == "__main__":

    diabetes = load_diabetes(as_frame=True)

    X = diabetes.frame.copy()
    y = diabetes.frame['target']   
        
    X.drop(['target'], axis=1, inplace=True) 

    tree, _ = train(X, y)