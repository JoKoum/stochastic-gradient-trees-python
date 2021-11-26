from utils.GradHess import GradHess
import numpy as np

class SquaredError:
    @staticmethod
    def computeDerivatives(groundTruth, raw):       
        return [
            GradHess(raw[i] - groundTruth[i], 1.0 )
            for i in range(len(raw))]
