class StreamingGradientTreeOptions:
    def __init__(self):
        self.delta = 1E-7
        self.gracePeriod = 200
        self.initialPrediction = 0.0
        self.mLambda = 0.1
        self.gamma = 1.0