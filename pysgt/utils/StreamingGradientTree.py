import numpy as np
from scipy.special import betainc
from .GradHessStats import GradHessStats

class StreamingGradientTree:
    def __init__(self, featureInfo, options):
        """
        mFeatureInfo: List of data features
        """
        self.mFeatureInfo = featureInfo
        self.mOptions = options
        
        self.mNumNodes = 0
        self.mNumNodeUpdates = 0
        self.mNumSplits = 0
        self.mMaxDepth = 0

        self.hasSplit = [False for _ in range(len(self.mFeatureInfo))]
        
        self.mRoot = Node(self.mOptions.initialPrediction, 1, self.hasSplit, self)
    
    def getNumNodes(self):
        return self.mNumNodes
    
    def getNumNodeUpdates(self):
        return self.mNumNodeUpdates
    
    def getNumSplits(self):
        return self.mNumSplits
    
    def getDepth(self):
        return self.mMaxDepth

    def randomlyInitialize(self, predBound):
        fid = np.random.randint(len(self.mFeatureInfo))
        self.mRoot.mSplit = Split()
        self.mRoot.mSplit.feature = fid
        self.mNumSplits += 1

        if self.mFeatureInfo[fid].type == 'nominal':
            self.mRoot.mChildren = Node(self.mFeatureInfo[fid].categories, self)

            for i in range(len(self.mRoot.mChildren)):
                self.mRoot.mChildren[i] = Node(predBound * (2.0 * np.random.rand() - 1.0), 2, self.hasSplit, self)

        elif self.mFeatureInfo[fid].type == 'ordinal':
            self.mRoot.mSplit.index = np.random.randint(self.mFeatureInfo[fid].categories / 2) + self.mFeatureInfo[fid].categories / 4

            self.mRoot.mChildren = []
            self.mRoot.mChildren.append(Node(predBound * (2.0 * np.random.rand() - 1.0), 2, self.hasSplit, self))
            self.mRoot.mChildren.append(Node(predBound * (2.0 * np.random.rand() - 1.0), 2, self.hasSplit, self))
        
    def update(self, features, gradHess):
        leaf = self.mRoot.getLeaf(features)

        leaf.update(features, gradHess)

        if leaf.mInstances % self.mOptions.gracePeriod != 0:
            return

        bestSplit = leaf.findBestSplit()
        leaf.setDeltaPredictions(bestSplit.deltaPredictions)

        p = self.computePValue(bestSplit, leaf.mInstances)
        
        if (p < self.mOptions.delta) and (bestSplit.lossMean < 0.0):
            leaf.applySplit(bestSplit)
    
    def predict(self, features):
        return self.mRoot.getLeaf(features).predict()

    def explain(self, features):
        self.nodePath = [self.mRoot]
        self.mRoot.traverseToLeaf(features)
        return {f'Node {node.mDepth}':
            {'Threshold': node.getDeltaPredictions(),
             'Prediction': node.predict()
             }
            for node in self.nodePath
        }
    
    def computePValue(self, split, instances):
        # H0: the expected loss is zero
        # HA: the expected loss is not zero
        try:
            F = instances * split.lossMean ** 2.0 / split.lossVariance
            return self._FProbability(F, 1, instances-1)
        except ArithmeticError:
            return 1.0
    
    @staticmethod
    def _FProbability(F, df1, df2):
        return betainc(df2/2.0, df1/2.0, df2/(df2+df1*F))

class Node:
    def __init__(self, prediction, depth, hasSplit, tree):
        
        self.mChildren = []
        self.tree = tree
        self.tree.mNumNodes += 1
            
        self.mPrediction = prediction
        self.mDepth = depth
        self.tree.mMaxDepth = max(self.tree.mMaxDepth, self.mDepth)
            
        self.mHasSplit = hasSplit.copy()
        self.mDeltaPrediction = []

        self.reset()
    
    def reset(self):
        self.mUpdateStats = GradHessStats()
        self.mInstances = 0
        self.mSplitStats = [[GradHessStats() for _ in range(self.tree.mFeatureInfo[i].categories)] for i in range(len(self.tree.mFeatureInfo))]
    
    def setDeltaPredictions(self, delta_prediction):
        self.mDeltaPrediction = delta_prediction

    def getDeltaPredictions(self):
        return self.mDeltaPrediction

    def getLeaf(self, features):
        if not self.mChildren:
            return self
        else:
            featureType = self.tree.mFeatureInfo[self.mSplit.feature].type
            c = None

            if features[self.mSplit.feature] == -1:
                c = self.mChildren[0]
            
            elif featureType == 'nominal':
                c = self.mChildren[int(features[self.mSplit.feature])]
            
            elif featureType == 'ordinal':
                if features[self.mSplit.feature] <= self.mSplit.index:
                    c = self.mChildren[0]
                else:
                    c = self.mChildren[1]
            
            else:
                raise SystemError("Unhandled attribute type")          
            return c.getLeaf(features)

    def traverseToLeaf(self, features):
        if not self.mChildren:
            self.tree.nodePath.append(self)
            return self
        else:
            featureType = self.tree.mFeatureInfo[self.mSplit.feature].type
            c = None

            if features[self.mSplit.feature] == -1:
                c = self.mChildren[0]
            
            elif featureType == 'nominal':
                c = self.mChildren[int(features[self.mSplit.feature])]
            
            elif featureType == 'ordinal':
                if features[self.mSplit.feature] <= self.mSplit.index:
                    c = self.mChildren[0]
                else:
                    c = self.mChildren[1]
            
            else:
                raise SystemError("Unhandled attribute type")
            if c.mChildren:
                self.tree.nodePath.append(c)         
            return c.traverseToLeaf(features)

    def update(self, features, gradHess):
        self.mInstances += 1
        for i in range(len(features)):
            if features[i] == -1:
                continue    
            self.mSplitStats[i][features[i]].addObservation(gradHess)   
        self.mUpdateStats.addObservation(gradHess)
    
    def predict(self):
        return self.mPrediction
    
    def findBestSplit(self):
        best = Split()

        # We can try to update the prediction using the new gradient information
        best.deltaPredictions = [self.computeDeltaPrediction(self.mUpdateStats.getMean())]
        best.lossMean = self.mUpdateStats.getDeltaLossMean(best.deltaPredictions[0])
        best.lossVariance = self.mUpdateStats.getDeltaLossVariance(best.deltaPredictions[0])

        best.feature = -1
        best.index = -1

        for i in range(len(self.mSplitStats)):
            candidate = Split()
            candidate.feature = i

            if (self.tree.mFeatureInfo[i].type == 'nominal'):
                if self.mHasSplit[i]:
                    continue

                candidate.deltaPredictions = list(np.zeros(len(self.mSplitStats[i])))
                lossMean = 0.0
                lossVar = 0.0
                observations = 0

                for j in range(len(self.mSplitStats[i])):
                    p = self.computeDeltaPrediction(self.mSplitStats[i][j].getMean())
                    m = self.mSplitStats[i][j].getDeltaLossMean(p)
                    s = self.mSplitStats[i][j].getDeltaLossVariance(p)
                    n = self.mSplitStats[i][j].getObservationCount()
                    candidate.deltaPredictions[j] = p

                    lossMean = GradHessStats().combineMean(lossMean, observations, m, n)
                    lossVar = GradHessStats().combineVariance(lossMean, lossVar, observations, m, s, n)
                    observations += n
                
                candidate.lossMean = lossMean + len(self.mSplitStats[i]) * self.tree.mOptions.gamma / self.mInstances
                candidate.lossVariance = lossVar
            
            elif (self.tree.mFeatureInfo[i].type == 'ordinal'):
                forwardCumulativeSum = [GradHessStats() for _ in range(self.tree.mFeatureInfo[i].categories - 1)]
                backwardCumulativeSum = [GradHessStats() for _ in range(self.tree.mFeatureInfo[i].categories - 1)]

                # Compute the split stats for each possible split point
                for j in range(self.tree.mFeatureInfo[i].categories - 1):
                    forwardCumulativeSum[j] += self.mSplitStats[i][j]

                    if j > 0:
                        forwardCumulativeSum[j] += forwardCumulativeSum[j - 1]                       
                    
                for j in range(self.tree.mFeatureInfo[i].categories - 2, -1, -1):
                    backwardCumulativeSum[j] += self.mSplitStats[i][j + 1]

                    if j + 1 < len(backwardCumulativeSum):
                        backwardCumulativeSum[j] += backwardCumulativeSum[j + 1]
                        
                candidate.lossMean = float('inf')
                candidate.deltaPredictions = list(np.zeros(2))

                for j in range(len(forwardCumulativeSum)):
                    deltaPredLeft = self.computeDeltaPrediction(forwardCumulativeSum[j].getMean())
                    lossMeanLeft = forwardCumulativeSum[j].getDeltaLossMean(deltaPredLeft)
                    lossVarLeft = forwardCumulativeSum[j].getDeltaLossVariance(deltaPredLeft)
                    numLeft = forwardCumulativeSum[j].getObservationCount()

                    deltaPredRight = self.computeDeltaPrediction(backwardCumulativeSum[j].getMean())
                    lossMeanRight = backwardCumulativeSum[j].getDeltaLossMean(deltaPredRight)
                    lossVarRight = backwardCumulativeSum[j].getDeltaLossVariance(deltaPredRight)
                    numRight = backwardCumulativeSum[j].getObservationCount()

                    lossMean = GradHessStats().combineMean(lossMeanLeft, numLeft, lossMeanRight, numRight)
                    lossVar = GradHessStats().combineVariance(lossMeanLeft, lossVarLeft, numLeft, lossMeanRight, lossVarRight, numRight)

                    if lossMean < candidate.lossMean:
                        candidate.lossMean = lossMean + 2.0 * self.tree.mOptions.gamma / self.mInstances
                        candidate.lossVariance = lossVar
                        candidate.index = j
                        candidate.deltaPredictions[0] = deltaPredLeft
                        candidate.deltaPredictions[1] = deltaPredRight
                        
            else :
                print("Unhandled attribute type")

            if candidate.lossMean < best.lossMean:
                best = candidate
            
        return best
    
    def applySplit(self, split):
        # Should we just update the prediction being made?
        if split.feature == -1:
            self.mPrediction += split.deltaPredictions[0]
            self.tree.mNumNodeUpdates += 1
            self.reset()
            return
        
        self.mSplit = split
        self.tree.mNumSplits += 1
        self.mHasSplit[split.feature] = True

        if self.tree.mFeatureInfo[split.feature].type == 'nominal':
            self.mChildren = []
            for i in range(self.tree.mFeatureInfo[split.feature].categories):
                self.mChildren.append(Node(self.mPrediction + split.deltaPredictions[i], self.mDepth + 1, self.mHasSplit, self.tree))
            
        elif self.tree.mFeatureInfo[split.feature].type == 'ordinal':
            self.mChildren = []
            self.mChildren.append(Node(self.mPrediction + split.deltaPredictions[0], self.mDepth + 1, self.mHasSplit, self.tree))
            self.mChildren.append(Node(self.mPrediction + split.deltaPredictions[1], self.mDepth + 1, self.mHasSplit, self.tree))

        else:
            print("Unhandled attribute type")

        # Free up memory used by the split stats
        self.mSplitStats = []
    
    def computeDeltaPrediction(self, gradHess):
        return -gradHess.gradient / (gradHess.hessian + self.tree.mOptions.mLambda + 2.225E-308)

class Split:
        def __init__(self):
            # lossMean and lossVariance are actually statistics of the approximation to the *change* in loss.
            self.lossMean = 0
            self.lossVariance = 0
            self.deltaPredictions = []
            self.feature = -1
            self.index = -1