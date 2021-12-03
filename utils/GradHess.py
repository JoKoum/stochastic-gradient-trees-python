class GradHess:
    def __init__(self, grad=0, hess=0, gradHess=None):
        
        if gradHess:
            self.gradient = gradHess.gradient
            self.hessian = gradHess.hessian
        else:
            self.gradient = grad
            self.hessian = hess
    
    def __add__(self, other):
        self.gradient += other.gradient
        self.hessian += other.hessian
        return self

    def __sub__(self, other):
        self.gradient -= other.gradient
        self.hessian -= other.hessian
        return self