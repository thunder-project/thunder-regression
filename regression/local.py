from numpy import dot, ones, hstack, square, newaxis
from numpy.linalg import inv

class PseudoInvRegression(object):

    def __init__(self, fit_intercept):
        self.fit_intercept = fit_intercept

    def prepare(self, X):
        '''compute the pseudoinvserse of the design matrix'''
        if self.fit_intercept:
            X = hstack([ones([X.shape[0], 1]), X])
        return dot(inv(dot(X.T, X)), X.T)

    def fit(self, pinv, y):
        betas = dot(pinv, y[:, newaxis]).flatten()
        if self.fit_intercept:
            self.coef_ = betas[1:]
            self.intercept_ = betas[0]
        else:
            self.coef_ = betas
            self.intercept_ = 0.0
        return self

    def score(self, X, y):
        yhat = self.predict(X)
        ybar = y.mean()
        SST = square(y - ybar).sum()
        SSR = square(y - yhat).sum()
        if SST == 0:
            return 0.0
        else:
            return 1 - SSR/SST

    def predict(self, X):
        if self.fit_intercept:
            X = hstack([ones([X.shape[0], 1]), X])
            betas = hstack([[self.intercept_], self.coef_])[:, newaxis]
        else:
            betas = self.coef_
        return dot(X, betas).flatten()
