from copy import deepcopy
class GradHess:
    def __init__(self, grad=0.0, hess=0.0, *, gradHess=None):
        
        if gradHess:
            self.gradient = gradHess.gradient
            self.hessian = gradHess.hessian
        else:
            self.gradient = grad
            self.hessian = hess
    
    def __iadd__(self, other):
        self.gradient += other.gradient
        self.hessian += other.hessian
        return self

    def __isub__(self, other):
        self.gradient -= other.gradient
        self.hessian -= other.hessian
        return self
    
    def __add__(self, other):
        sumObj = deepcopy(self)
        sumObj += other
        return sumObj
    
    def __sub__(self, other):
        subObj = deepcopy(self)
        subObj -= other
        return subObj