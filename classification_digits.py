from StochasticGradientTree import StochasticGradientTreeClassifier

import time

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

if __name__ == '__main__':

    digits = load_digits(as_frame=True)

    X = digits.frame.copy()
    y = digits.frame.target
        
    X.drop(['target'], axis=1, inplace=True)

    estimator = StochasticGradientTreeClassifier()

    clf = OneVsRestClassifier(estimator)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34)

    start = time.process_time()
    clf.fit(X_train,y_train)
    print('Time taken: {}s'.format(time.process_time() - start))

    pred = clf.predict(X_test)

    print(accuracy_score(y_test, pred))
    print(confusion_matrix(y_test, pred))