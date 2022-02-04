import numpy as np
from utils.GradHess import GradHess
from utils.GradHessStats import GradHessStats

def testGetDeltaLossMean():
    stats = GradHessStats()
    stats.addObservation(GradHess(0.5, 1.0))
    stats.addObservation(GradHess(1.0, 1.0))
    stats.addObservation(GradHess(1.5, 1.0))
    deltaPrediction = -1.0
    deltaLossMean = stats.getDeltaLossMean(deltaPrediction)

    np.testing.assert_allclose(deltaLossMean, -0.5, 1E-12)
    
def testGetDeltaLossVariance():
    stats = GradHessStats()
    stats.addObservation(GradHess(0.5, 1.0))
    stats.addObservation(GradHess(1.0, 1.0))
    stats.addObservation(GradHess(1.5, 1.0))
    deltaPrediction = -1.0
    deltaLossVariance = stats.getDeltaLossVariance(deltaPrediction)
    np.testing.assert_allclose(deltaLossVariance, 0.25, 1E-12)

def testAdd():
    stats1 = GradHessStats()
    stats1.addObservation(GradHess(0.5, 0.8))
    stats1.addObservation(GradHess(1.0, 1.0))
    stats1.addObservation(GradHess(1.5, 1.2))

    stats2 = GradHessStats()
    stats2.addObservation(GradHess(0.5, 1.2))
    stats2.addObservation(GradHess(1.0, 1.0))
    stats2.addObservation(GradHess(1.5, 0.9))

    stats1 += stats2

    expected = GradHessStats()
    expected.addObservation(GradHess(0.5, 0.8))
    expected.addObservation(GradHess(1.0, 1.0))
    expected.addObservation(GradHess(1.5, 1.2))
    expected.addObservation(GradHess(0.5, 1.2))
    expected.addObservation(GradHess(1.0, 1.0))
    expected.addObservation(GradHess(1.5, 0.9))
        
    np.testing.assert_allclose(expected.getVariance().gradient, stats1.getVariance().gradient, 1E-12)
    np.testing.assert_allclose(expected.getCovariance(), stats1.getCovariance(), 1E-12)

def testCombineMean():
    m1 = 2.0
    n1 = 5
    m2 = 3.0
    n2 = 3
    m = 19.0 / 8.0

    np.testing.assert_allclose(m, GradHessStats().combineMean(m1, n1, m2, n2), 1E-12)

def testCombineVariance():
    m1 = 2.0
    s1 = 1.0
    n1 = 5
    m2 = 3.0
    s2 = 4.0
    n2 = 3
    #m = 19.0 / 8.0
    
    np.testing.assert_allclose(5.7142857142857135, GradHessStats().combineVariance(m1, s1, n1, m2, s2, n2), 1E-12)
    #np.testing.assert_allclose(1.9821, stats.combineVariance(m1, s1, n1, m2, s2, n2), 1E-12)

if __name__ == '__main__':
    testGetDeltaLossMean()
    testGetDeltaLossVariance()
    testAdd()
    testCombineMean()
    testCombineVariance()