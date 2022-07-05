import numpy as np
from .utils.StreamingGradientTree import StreamingGradientTree
from .utils.StreamingGradientTreeOptions import StreamingGradientTreeOptions
from .utils.FeatureInfo import FeatureInfo
from .utils.SoftmaxCrossEntropy import SoftmaxCrossEntropy
from .utils.SquaredError import SquaredError
from sklearn.base import BaseEstimator

class SGT:
    def __init__(self, objective=None, bins=64, batch_size=200, epochs=20, m_lambda=0.1, gamma=1.0, upper_bounds=[], lower_bounds=[], learning_rate=1):

        self.objective = objective

        self.bins = bins
        self.batch_size = batch_size
        self.epochs = epochs

        self.m_lambda = m_lambda
        self.gamma = gamma
        
        self.options = StreamingGradientTreeOptions()
        self.options.gracePeriod = self.batch_size
        self.options.mLambda = self.m_lambda
        self.options.gamma = self.gamma

        if len(upper_bounds) == 0 and len(lower_bounds) == 0:
            self.MinMaxProvided = False
        else:
            self.MinMaxProvided = True
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds

        self._isFit = False
        self._samplesSeen = 0

        self.lr = learning_rate
    
    def getEpochs(self):
        return self.epochs
    
    def setEpochs(self, epochs):
        self.epochs = epochs
    
    def setBins(self, bins):
        self.bins = bins
    
    def getBins(self):
        return self.bins
    
    def setTrainBatchSize(self, bs):
        self.batch_size = bs
    
    def getTrainBatchSize(self):
        return self.batch_size
    
    def setLambda(self, l):
        self.m_lambda = l
    
    def getLambda(self):
        return self.m_lambda

    def setGamma(self, l):
        self.gamma = l
    
    def getGamma(self):
        return self.gamma
    
    def get_depth(self):
        if not hasattr(self, 'tree'):
            return 0
        return self.tree.getDepth()

    def get_total_nodes(self):
        if not hasattr(self, 'tree'):
            return 1
        return self.tree.getNumNodes()

    def set_learning_rate(self, lr):
        self.lr = lr

    def createFeatures(self, X):
        if hasattr(X, "dtypes") and hasattr(X, "__array__"):
            self.dtypes = list(X.dtypes)
            X = X.to_numpy()
        else:
            self.dtypes = [X.dtype for _ in range(X.shape[1])]
        
        if self._samplesSeen < 1000:
            if not self._isFit:
                self.features = X
            else: 
                self.features = np.append(self.features, X, axis=0)    
            self._samplesSeen += X.shape[0]

            if not self.MinMaxProvided:
                self.upper_bounds = np.max(self.features, axis=0)
                self.lower_bounds = np.min(self.features, axis=0)
        
        if not self._isFit:
            self.featureInfo = [FeatureInfo('nominal', len(np.unique(X[:,i]))) if 'int' in self.dtypes[i].name else FeatureInfo('ordinal', self.bins) for i in range(X.shape[1])]
            self.buckets = [len(np.unique(X[:,i])) if 'int' in self.dtypes[i].name else self.bins for i in range(X.shape[1])]
        
        discretized = np.zeros_like(X)

        for i, observation in enumerate(X):
            discretized[i,:] = self._discretize(observation)
    
        return discretized.astype(int)

    def _discretize(self, observations):
        '''Dicretize obervations based on the created buckets'''
        discretized = []
        for i in range(len(observations)):
            if self.upper_bounds[i] == self.lower_bounds[i]:
                scaling = 1
            else:
                scaling = ((observations[i] - self.lower_bounds[i]) / (self.upper_bounds[i] - self.lower_bounds[i]))
            scaled_observations = int(self.buckets[i] * scaling)
            scaled_observations = min(self.buckets[i] - 1, max(0, scaled_observations))
            discretized.append(scaled_observations)
        return np.array(discretized)

    def _train(self, x, y):
        pred = [self.tree.predict(x)]
        gradHess = self.objective.computeDerivatives([y], pred)
        gradHess[0].gradient, gradHess[0].hessian = self.lr * gradHess[0].gradient, self.lr * gradHess[0].hessian
        self.tree.update(x, gradHess[0])

    def fit(self, X, y):        
        X = self.createFeatures(X)        

        if not self._isFit:
            self.tree = StreamingGradientTree(self.featureInfo, self.options)

        if 'pandas' in str(type(y)):
            y = y.to_numpy()

        [[self._train(x, yi) for x, yi in zip(X,y)] for _ in range(self.epochs)]

        self._isFit = True

class SGTClassifier(SGT):
    def __init__(self, objective=SoftmaxCrossEntropy(), bins=64, batch_size=200, epochs=20, m_lambda=0.1, gamma=1.0, upper_bounds=[], lower_bounds=[], learning_rate=1):
        super().__init__(
            objective=objective,
            bins=bins,
            batch_size=batch_size,
            epochs=epochs,
            m_lambda=m_lambda,
            gamma=gamma,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            learning_rate=learning_rate
            )
        self._estimator_type = 'classifier'
    
    def predict(self, X):        
        y_pred = self.predict_proba(X)

        return [np.argmax(pred) for pred in y_pred]
    
    def predict_proba(self, X):
        X = self.createFeatures(X)

        if not self._isFit:
            self.tree = StreamingGradientTree(self.featureInfo, self.options)

        logits = [self.tree.predict(X[i]) for i in range(len(X))]
        
        probs = [self.objective.transfer([logit]) for logit in logits]

        proba = [[prob[1], prob[0]] for prob in probs]

        return np.array(proba)

class SGTRegressor(SGT):
    def __init__(self, objective=SquaredError(), bins=64, batch_size=200, epochs=20, m_lambda=0.1, gamma=1.0, upper_bounds=[], lower_bounds=[], learning_rate=1):
        super().__init__(
            objective=objective,
            bins=bins,
            batch_size=batch_size,
            epochs=epochs,
            m_lambda=m_lambda,
            gamma=gamma,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            learning_rate = learning_rate
            )
        self._estimator_type = 'regressor'
    
    def predict(self, X):        
        X = self.createFeatures(X)

        if not self._isFit:
            self.tree = StreamingGradientTree(self.featureInfo, self.options)

        y_pred = [self.tree.predict(X[i]) for i in range(len(X))]
        
        return y_pred
    
    def predict_proba(self, X):
        print("Not supported Regression method")

class StochasticGradientTree(SGT, BaseEstimator):
    def __init__(self, objective=None, bins=64, batch_size=200, epochs=20, m_lambda=0.1, gamma=1, upper_bounds=[], lower_bounds=[]):
        super().__init__(
            objective=objective,
            bins=bins,
            batch_size=batch_size,
            epochs=epochs, 
            m_lambda=m_lambda,
            gamma=gamma,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds
            )

class StochasticGradientTreeClassifier(StochasticGradientTree, SGTClassifier):
    def __init__(self, objective=SoftmaxCrossEntropy(), bins=64, batch_size=200, epochs=20, m_lambda=0.1, gamma=1, upper_bounds=[], lower_bounds=[]):
        super().__init__(
            objective=objective,
            bins=bins,
            batch_size=batch_size,
            epochs=epochs,
            m_lambda=m_lambda,
            gamma=gamma,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds
            )

class StochasticGradientTreeRegressor(StochasticGradientTree, SGTRegressor):
    def __init__(self, objective=SquaredError(), bins=64, batch_size=200, epochs=20, m_lambda=0.1, gamma=1, upper_bounds=[], lower_bounds=[]):
        super().__init__(
            objective=objective,
            bins=bins,
            batch_size=batch_size,
            epochs=epochs,
            m_lambda=m_lambda,
            gamma=gamma,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds
            )
    