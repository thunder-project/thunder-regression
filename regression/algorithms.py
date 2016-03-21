from copy import deepcopy

from .model import MassRegressionModel
from .utils import toseries


class MassRegressionAlgorithm:
    """
    Base class for mass univariate algorithms
    """

    def __init__(self):
        raise NotImplementedError


    def fit(self, X, y):
        """
        Fit a mass univariate regression model

        Using a single design matrix, the same regression model is fit to any
        number of response variables.

        Parameters
        ----------

        X : ndarray, 2d
            The design matrix, a two-dimensional ndarray. Each row is a unique
            measurement and each column is different regressor / explanatory
            variable.

        y : array-like (thunder Series, BoltArray, or ndarray)
            Collection of response variables. Can be a thunder Series object,
            where each record is a different set of response variables or a
            local (ndarray) or distributed (BoltArray) with the first dimension
            indexing response varibles and the second dimension indexing data
            points.
        """

        y = toseries(y)
        alg = self.alg

        return MassRegressionModel(y.map(lambda v: deepcopy(alg).fit(X, v)))

    def fit_with_score(self, X, y):
        """
        Fit a mass univariate regression model and return the scores as well

        Parameters
        ----------

        X : ndarray, 2d
            The design matrix, a two-dimensional ndarray. Each row is a unique
            measurement and each column is different regressor / explanatory
            variable.

        y : array-like (thunder Series, BoltArray, or ndarray)
            Collection of response variables. Can be a thunder Series object,
            where each record is a different set of response variables or a
            local (ndarray) or distributed (BoltArray) with the first dimension
            indexing response varibles and the second dimension indexing data
            points.
        """

        y = toseries(y)
        alg = self.alg

        def getboth(v):
            fitted = alg.fit(X, v)
            score = fitted.score(X, v)
            return [score, fitted]

        both = y.map(getboth)
        return both.map(lambda v: v[0]), both.map(lambda v: v[1])

class LinearRegression(MassRegressionAlgorithm):
    """
    Mass univariate linear regression algorithm.

    Fits scikit-learn's LinearRegression model to each series.

    Parameters
    ----------

    fit_intercept : bool, optional, default=True
        Whether or not to fit an intercept term. If true, the first coefficient
        on the fitted model will be the intercept.

    normalize : bool, optional, default=False
        Whether or not to normalize each regressor prior to fitting. If true,
        coefficients represent change in response variable for a change in the
        regressor in units of standard deviations from the mean.
    """

    def __init__(self, fit_intercept=True, normalize=False):
        from sklearn.linear_model import LinearRegression as LR
        self.alg = LR(fit_intercept=fit_intercept, normalize=normalize);

class CustomRegression(MassRegressionAlgorithm):
    """
    Custom mass univariate regression algorithm.

    Parameters
    ----------

    algorithm : object
        An object encapsulating a regression algorithm.

        Can be a scikit-learn algorithm or a custom algorithm that adheres to a
        scikit-learn style API. Must implement a fit(X, y) method that takes a
        deisgn matrix and response vector (X and y, both NumPy ndarrays) and
        returns a fitted model. The fitted model should expose data members
        coef_ and intercept_ that contain the fitted coefficients and intercept
        (if any). The fitted model should also implement a score(X, y) method
        giving the r-squared of fit for the given data and a predict(X) method
        that predicts responses given a design matrix.
    """

    def __init__(self, algorithm):
        self.alg = algorithm
