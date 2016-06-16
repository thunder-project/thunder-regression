import pytest

from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
from numpy import allclose, r_, asarray, newaxis, hstack
from numpy.random import randn

from thunder.series import fromarray
from regression import LinearRegression, FastLinearRegression, CustomRegression

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

	truth = asarray(fit_models(LR, X, y, fit_intercept=False))
	betas = LinearRegression(fit_intercept=False).fit(X, y).betas.toarray()
	assert allclose(truth, betas)


def test_fast_linear(eng):
	X = randn(10, 2)
	y = fromarray(randn(10, 4).T, engine=eng)

	truth = asarray(fit_models(LR, X, y))
	betas = FastLinearRegression().fit(X, y).betas.toarray()
	assert allclose(truth, betas)

	truth = asarray(fit_models(LR, X, y, fit_intercept=False))
	betas = FastLinearRegression(fit_intercept=False).fit(X, y).betas.toarray()
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


def test_score(eng):
	X = randn(10, 2)
	y = fromarray(randn(10, 4).T, engine=eng)

	truth = asarray(score_models(LR, X, y))
	scores = LinearRegression().fit(X, y).score(X, y).toarray()
	assert allclose(truth, scores)


def test_betas_and_scores(eng):
	X = randn(10, 2)
	y = fromarray(randn(10, 4).T, engine=eng)

	true_betas = asarray(fit_models(LR, X, y))
	true_scores = asarray(score_models(LR, X, y))
	truth = hstack([true_betas, true_scores[:, newaxis]])

	result = LinearRegression().fit(X, y).betas_and_scores.toarray()

	assert allclose(truth, result)


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

	yhat = model.predict(X).toarray()
	rsq = model.score(X, y).toarray()
	truth = hstack([yhat, rsq[:, newaxis]])

	result = model.predict_and_score(X, y).toarray()

	assert allclose(truth, result)
