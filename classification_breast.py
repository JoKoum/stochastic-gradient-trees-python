from pysgt.StochasticGradientTree import StochasticGradientTreeClassifier
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss

def train(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34)
        
    tree = StochasticGradientTreeClassifier()

    start = time.process_time()
    tree.fit(X_train, y_train)
    print('Time taken: {}s'.format(time.process_time() - start))
    
    y_pred = tree.predict(X_test)

    proba = tree.predict_proba(X_test)        

    acc_test = accuracy_score(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))
    print('Acc test: ', acc_test)
    print('Cross entropy loss: ', log_loss(y_test, proba))

    return tree, acc_test

if __name__ == "__main__":

    breast = load_breast_cancer(as_frame=True)

    X = breast.frame.copy()
    y = breast.frame.target
        
    X.drop(['target'], axis=1, inplace=True) 

    tree, _ = train(X, y)