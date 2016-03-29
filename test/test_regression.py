import pytest

from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
from numpy import allclose, r_, asarray
from numpy.random import randn

from thunder.series import fromarray
from regression import LinearRegression, CustomRegression

pytestmark = pytest.mark.usefixtures("eng")

def fit_models(model, X, y, **kwargs):
	fit = [model(**kwargs).fit(X, v) for v in y.toarray()]
	return list(map(lambda m: r_[m.intercept_, m.coef_], fit))


def score_models(model, X, y, **kwargs):
	return [model(**kwargs).fit(X, v).score(X, v) for v in y.toarray()]


def predict_models(model, X, y, **kwargs):
	return [model(**kwargs).fit(X, v).predict(X) for v in y.toarray()]


def test_linear(eng):
	X = randn(10, 2)
	y = fromarray(randn(10, 4).T, engine=eng)

	truth = asarray(fit_models(LR, X, y))
	betas = LinearRegression().fit(X, y).betas.toarray()
	assert allclose(truth, betas)

	truth = asarray(fit_models(LR, X, y, fit_intercept=True))
	betas = LinearRegression(fit_intercept=True).fit(X, y).betas.toarray()
	assert allclose(truth, betas)


def test_custom(eng):
	X = randn(10, 2)
	y = fromarray(randn(10, 4).T, engine=eng)

	truth = asarray(fit_models(Ridge, X, y))
	betas = CustomRegression(Ridge()).fit(X, y).betas.toarray()
	assert allclose(truth, betas)

	kwargs = {"fit_intercept": False, "normalize": True}
	truth = asarray(fit_models(Ridge, X, y, **kwargs))
	betas = CustomRegression(Ridge(**kwargs)).fit(X, y).betas.toarray()
	assert allclose(truth, betas)

def test_fit_and_score(eng):
	X = randn(10, 2)
	y = fromarray(randn(10, 4).T, engine=eng)

	betas1 = LinearRegression().fit(X, y).betas.toarray()
	rsq1 = LinearRegression().fit(X, y).score(X, y).toarray()

	model, rsq2 = LinearRegression().fit_and_score(X, y)
	betas2, rsq2 = model.betas.toarray(), rsq2.toarray()

	assert allclose(betas1, betas2)
	assert allclose(rsq1, rsq2)

def test_score(eng):
	X = randn(10, 2)
	y = fromarray(randn(10, 4).T, engine=eng)

	truth = asarray(score_models(LR, X, y))
	scores = LinearRegression().fit(X, y).score(X, y).toarray()
	assert allclose(truth, scores)


def test_predict(eng):
	X = randn(10, 2)
	y = fromarray(randn(10, 4).T, engine=eng)

	truth = asarray(predict_models(LR, X, y))
	predictions = LinearRegression().fit(X, y).predict(X).toarray()
	assert allclose(truth, predictions)

def test_predict_and_score(eng):
	X = randn(10, 2)
	y = fromarray(randn(10, 4).T, engine=eng)

	model = LinearRegression().fit(X, y)

	yhat1 = model.predict(X).toarray()
	rsq1 = model.score(X, y).toarray()

	yhat2, rsq2 = model.predict_and_score(X, y)
	yhat2, rsq2 = yhat2.toarray(), rsq2.toarray()

	assert allclose(yhat1, yhat2)
	assert allclose(rsq1, rsq2)
