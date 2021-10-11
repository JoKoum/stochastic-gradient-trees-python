import numpy as np
from utils.GradHess import GradHess
import warnings
warnings.filterwarnings("ignore")

class GradHessStats:
    def __init__(self):
        self.mObservations = 0
        self.mSum = GradHess()
        self.mScaledVariance = GradHess()
        self.mScaledCovariance = 0.0
    
    def add(self, stats):
        if stats.mObservations == 0:
            return
        
        if self.mObservations == 0:
            self.mSum = GradHess(gradHess=stats.mSum)
            self.mScaledVariance = GradHess(gradHess=stats.mScaledVariance)
            self.mScaledCovariance = stats.mScaledCovariance
            self.mObservations = stats.mObservations
            return
        
        meanDiff = stats.getMean()
        meanDiff.sub(self.getMean())
        n1 = self.mObservations
        n2 = stats.mObservations

        # Do scaled variance bit (see Wikipedia page on "Algorithms for calculating variance", section about parallel calculation)
        self.mScaledVariance.gradient += stats.mScaledVariance.gradient + np.power(meanDiff.gradient, 2.0) * (n1 * n2) / (n1 + n2)
        self.mScaledVariance.hessian += stats.mScaledVariance.hessian + np.power(meanDiff.hessian, 2.0) * (n1 * n2) / (n1 + n2)
        
        # Do scaled covariance bit (see "Numerically Stable, Single-Pass, Parallel Statistics Algorithms" (Bennett et al, 2009))
        self.mScaledCovariance += stats.mScaledCovariance + meanDiff.gradient * meanDiff.hessian * (n1 * n2) / (n1 + n2)

        # Do the other bits
        self.mSum.add(stats.mSum)
        self.mObservations += stats.mObservations
    
    def addObservation(self, gradHess):
        oldMean = self.getMean()
        self.mSum.add(gradHess)
        self.mObservations += 1
        newMean = self.getMean()


        self.mScaledVariance.gradient += (gradHess.gradient - oldMean.gradient) * (gradHess.gradient - newMean.gradient)
        self.mScaledVariance.hessian += (gradHess.hessian - oldMean.hessian) * (gradHess.hessian - newMean.hessian)

        self.mScaledCovariance += (gradHess.gradient - oldMean.gradient) * (gradHess.hessian - newMean.hessian)
    
    def getMean(self):
        if self.mObservations == 0:
            return GradHess(0.0, 0.0)
        else:
            return GradHess(self.mSum.gradient / self.mObservations, self.mSum.hessian / self.mObservations)
    
    def getVariance(self):
        if self.mObservations < 2:
            return GradHess(np.inf, np.inf)
        else:
            return GradHess(self.mScaledVariance.gradient / (self.mObservations - 1), self.mScaledVariance.hessian / (self.mObservations - 1))
    
    def getCovariance(self):
        if self.mObservations < 2:
            return np.inf
        else:
            return self.mScaledCovariance / (self.mObservations - 1)
    
    def getObservationCount(self):
        return self.mObservations
    
    def getDeltaLossMean(self, deltaPrediction):
        mean = self.getMean()
        return deltaPrediction * mean.gradient + 0.5 * mean.hessian * np.power(deltaPrediction, 2.0)
    
    # This method ignores correlations between deltaPrediction and the gradients/hessians! Considering
    # deltaPredicions is derived from the gradient and hessian sample, this assumption is definitely violated.
    def getDeltaLossVariance(self, deltaPrediction):
                
        variance = self.getVariance()
        covariance = self.getCovariance()

        gradTermVariance = np.power(deltaPrediction, 2.0) * variance.gradient
        hessTermVariance = 0.25 * variance.hessian * np.power(deltaPrediction, 4.0)

        return max(0.0, gradTermVariance + hessTermVariance + np.power(deltaPrediction, 3.0) * covariance)
    
    @staticmethod
    def combineMean(m1, n1, m2, n2):
        if n1 == 0:
            return m2
        if n2 == 0:
            return m1

        return (m1 * n1 + m2 * n2) / (n1 + n2)
    
    @staticmethod
    def combineVariance(m1, s1, n1, m2, s2, n2):
        # Some special cases, just to be safe
        if n1 == 0:
            return s2
        if n2 == 0:
            return s1

        n = n1 + n2

        m = GradHessStats().combineMean(m1, n1, m2, n2)

        # First we have to bias the sample variances (we'll unbias this later)
        s1 = ((n1 - 1) / n1) * s1
        s2 = ((n2 - 1) / n2) * s2

        # Compute the sum of squares of all the datapoints
        t1 = n1 * (s1 + m1 * m1)
        t2 = n2 * (s2 + m2 * m2)
        t = t1 + t2

        # Now get the full (biased) sample variance
        s = t / n - m

        # Apply Bessel's correction
        s = (n / (n - 1)) * s

        return s