import numpy as np
from sklearn.preprocessing import LabelBinarizer
from copy import copy

class OneVsRestClassifier:
    def __init__(self, estimator):
        self.estimator = estimator
        self.label_binarizer = LabelBinarizer(sparse_output=False)
        self._isFit = False

    def fit(self, X, y):
        y = self.label_binarizer.fit_transform(y.values.reshape(-1, 1))
        if not self._isFit:
            self.classifiers = [copy(self.estimator) for _ in range(y.shape[1])]
        
        for i, classifier in enumerate(self.classifiers):
            classifier.fit(X, y[:, i])
        
        self._isFit = True

    def predict(self, X):      
        maxima = np.empty(len(X), dtype=float)
        maxima.fill(-np.inf)
        argmaxima = np.zeros(len(X), dtype=int)

        for i, classifier in enumerate(self.classifiers):
            
            pred = classifier.predict_proba(X)[:, 1]
            np.maximum(maxima, pred, out=maxima)
            argmaxima[maxima == pred] = i
        
        return self.label_binarizer.classes_[argmaxima]
        