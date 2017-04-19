from .utils import toseries
from numpy import r_, array
from thunder.series import fromrdd

class MassRegressionModel:
    """
    A fitted mass univariate regression model.

    Contains a collection regression models, each fitted with the same design
    matrix, but to a different response varaible.
    """

    def __init__(self, models):
        self.models = models

    @property
    def coef_(self):
        return self.models.map(lambda v: v[0].coef_)

    @property
    def intercept_(self):
        return self.models.map(lambda v: v[0].intercept_)

    @property
    def betas(self):
        def getbetas(model):
            return r_[model.intercept_, model.coef_]
        return self.models.map(lambda v: getbetas(v[0]))

    @property
    def betas_and_scores(self):
        def getvalues(model):
            return r_[model.intercept_, model.coef_, model.score_]
        return self.models.map(lambda v: getvalues(v[0]))

    def predict(self, X):
        return self.models.map(lambda v: v[0].predict(X))

    def score(self, X, y):

        y = toseries(y)

        if y.mode == "spark":
            if not self.models.mode == "spark":
                raise ValueError("model is spark mode, input y must also be spark mode")
            joined = self.models.tordd().join(y.tordd())
            result = joined.mapValues(lambda v: array([v[0][0].score(X, v[1])]))
            series = fromrdd(result, shape=self.models.shape)
            series.values._ordered = False
            return series

        if y.mode == "local":
            if not self.models.mode == "local":
                raise ValueError("mode is local mode, input y must also be local mode")
            return self.models.map(lambda kv: kv[1][0].score(X, y.values[kv[0]]), with_keys=True)

    def predict_and_score(self, X, y):

        y = toseries(y)

        def get_both(model, X, y):
            return r_[model.predict(X), model.score(X, y)]

        if y.mode == "spark":
            if not self.models.mode == "spark":
                raise ValueError("model is spark mode, input y must also be spark mode")
            joined = self.models.tordd().join(y.tordd())
            both = fromrdd(joined.mapValues(lambda v: get_both(v[0][0], X, v[1])))
            both.values._ordered = False

        if y.mode == "local":
            if not self.models.mode == "local":
                raise ValueError("mode is local mode, input y must also be local mode")
            both = self.models.map(lambda kv: get_both(kv[1][0], X, y.values[kv[0]]), with_keys=True)

        return both 
