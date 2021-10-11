Stochastic Gradient Trees - Python
=========================

[Stochastic Gradient Trees](https://arxiv.org/abs/1901.07777)[^1] by Henry Gouk, Bernhard Pfahringer, and Eibe Frank implementation in Python. Based on the [parer's accompanied repository](https://github.com/henrygouk/stochastic-gradient-trees) code.

### Python Version 3.7

### Used Python libraries:
* numpy==1.20.2
* scikit-learn==0.24.2

### Usage:

    from StochasticGradientTree import StochasticGradientTreeClassifier

    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer
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

        breast = load_breast_cancer(as_frame=True)

        X = breast.frame.copy()
        y = breast.frame.target
        
        X.drop(['target'], axis=1, inplace=True) 

        tree, _ = train(X, y)

[^1]: Gouk, H., Pfahringer, B., and Frank, E. Stochastic gradient trees. In Proceedings of The Eleventh Asian Conference on Machine Learning, volume 101 of Proceedings of Machine Learning Research, pp. 1094–1109. PMLR, 2019.