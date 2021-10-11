from utils.GradHess import GradHess

class SquaredError:
    @staticmethod
    def computeDerivatives(groundTruth, raw):
        result = [GradHess() for _ in range(len(raw))]

        for i in range(len(result)):
           result[i] = GradHess(raw[i] - groundTruth[i], 1.0 )
        
        return result
