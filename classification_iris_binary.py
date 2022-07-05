import numpy as np

from pysgt.StochasticGradientTree import StochasticGradientTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss

def train(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34)
        
    tree = StochasticGradientTreeClassifier()

    tree.fit(X_train, y_train)
    
    y_pred = tree.predict(X_test)
    proba = tree.predict_proba(X_test)        

    acc_test = accuracy_score(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))
    print('Acc test: ', acc_test)
    print('Cross entropy loss: ', log_loss(y_test, proba))

    return tree, acc_test

if __name__ == "__main__":

    iris = load_iris(as_frame=True)
    
    iris.frame.drop(list(np.where(iris.frame.target == 2))[0], inplace=True)

    X = iris.frame.copy()
    y = iris.frame.target
        
    X.drop(['target'], axis=1, inplace=True) 

    tree, _ = train(X, y)