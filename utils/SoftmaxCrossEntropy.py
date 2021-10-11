import numpy as np
from utils.GradHess import GradHess

class SoftmaxCrossEntropy:
    def computeDerivatives(self, groundTruth, raw):
        result = [GradHess() for _ in range(len(raw))]
        predictions = self.transfer(raw)

        for i in range(len(result)):
           result[i] = GradHess(predictions[i] - groundTruth[i], predictions[i] * (1.0 - predictions[i]))
        
        return result
    
    @staticmethod
    def transfer(raw):
        result = np.zeros(len(raw)+1)
        
        for i in range(len(raw)):
            result[i] = raw[i]
        
        Max = -np.inf
        Sum = 0.0

        for i in range(len(result)):
            Max = max(Max, result[i])
        
        for i in range(len(result)):
            result[i] = np.exp(result[i] - Max)
            Sum += result[i]
        
        for i in range(len(result)):
            result[i] /= Sum

        return result