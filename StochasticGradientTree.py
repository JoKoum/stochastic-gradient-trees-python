import numpy as np
from utils.StreamingGradientTree import StreamingGradientTree
from utils.StreamingGradientTreeOptions import StreamingGradientTreeOptions
from utils.FeatureInfo import FeatureInfo
from utils.SoftmaxCrossEntropy import SoftmaxCrossEntropy
from utils.SquaredError import SquaredError

from sklearn.base import BaseEstimator
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
    
class StochasticGradientTreeClassifier(BaseEstimator):
    def __init__(self, bins=64, batch_size=200, epochs=20, m_lambda=0.1, gamma=1.0):
        self.bins = bins
        self.batch_size = batch_size
        self.epochs = epochs

        self.m_lambda = m_lambda
        self.gamma = gamma
        
        self.options = StreamingGradientTreeOptions()
        self.options.gracePeriod = self.batch_size
        self.options.mLambda = self.m_lambda
        self.options.gamma = self.gamma

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

    def createFeatures(self, X):
        fx = X.copy()

        featureInfo = []
        self.discretizers = {}
        self.encoders = {}
    
        for i in range(fx.values.shape[1]):
            if fx[fx.columns[i]].dtype == 'object':
                feat = FeatureInfo('nominal', len(fx[fx.columns[i]].unique()))
                encoder = OrdinalEncoder(dtype=np.int64)
                fx[fx.columns[i]] = encoder.fit_transform(fx[fx.columns[i]].values.reshape(-1,1))
                self.encoders[i] = encoder

            else:
                feat = FeatureInfo('ordinal', self.bins)
                ds = KBinsDiscretizer(self.bins, encode='ordinal', strategy='uniform')
                fx[fx.columns[i]] = ds.fit_transform(fx[fx.columns[i]].values.reshape(-1,1))
                fx[fx.columns[i]] = np.array(fx[fx.columns[i]].values, dtype=np.int64)
                self.discretizers[i] = ds
            
            featureInfo.append(feat)
    
        return fx, featureInfo
    
    def transformFeatures(self, X):
        fx = X.copy()
    
        for i in range(fx.values.shape[1]):
            if fx[fx.columns[i]].dtype == 'object':
                fx[fx.columns[i]] = self.encoders[i].transform(fx[fx.columns[i]].values.reshape(-1,1))
            else:
                fx[fx.columns[i]] = self.discretizers[i].transform(fx[fx.columns[i]].values.reshape(-1,1))
                fx[fx.columns[i]] = np.array(fx[fx.columns[i]].values, dtype=np.int64)    
        return fx
        

    def fit(self, X, y):

        X, featureInfo = self.createFeatures(X)
        self.tree = StreamingGradientTree(featureInfo, self.options)

        self.mObjective = SoftmaxCrossEntropy()

        X = X.values
        
        try:
            y = y.values
        except:
            pass

        for _ in range(self.epochs):
        
            for i, x in enumerate(X):
            
                pred = [self.tree.predict(x)]         
                target = [np.float64(y[i])]
            
                gradHess = self.mObjective.computeDerivatives(target,pred)
                
                self.tree.update(x, gradHess[0])
    
    def predict(self, X):
        
        y_pred = self.predict_proba(X)

        return [np.argmax(pred) for pred in y_pred]
    
    def predict_proba(self, X):

        X, _ = self.createFeatures(X)

        X = X.values

        logits = [self.tree.predict(X[i]) for i in range(len(X))]
        
        probs = [self.mObjective.transfer([logit]) for logit in logits]

        proba = [[prob[1], prob[0]] for prob in probs]

        return np.array(proba)

class StochasticGradientTreeRegressor(StochasticGradientTreeClassifier):
    def __init__(self, bins=64, batch_size=200, epochs=20, m_lambda=0.1, gamma=1.0):
        super().__init__( bins, batch_size, epochs, m_lambda, gamma)

    def fit(self, X, y):

        X, featureInfo = self.createFeatures(X)
        self.tree = StreamingGradientTree(featureInfo, self.options)

        self.mObjective = SquaredError()

        X = X.values
        y = y.values

        for _ in range(self.epochs):
        
            for i, x in enumerate(X):
            
                pred = [self.tree.predict(x)]         
                target = [np.float64(y[i])]
            
                gradHess = self.mObjective.computeDerivatives(target,pred)
                
                self.tree.update(x, gradHess[0])
    
    def predict(self, X):

        X = self.transformFeatures(X)

        X = X.values

        y_pred = [self.tree.predict(X[i]) for i in range(len(X))]
        
        return y_pred
    
    def predict_proba(self, X):
        print("Not supported Regression method")