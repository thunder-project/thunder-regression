# thunder-regression

> algorithms for mass univariate regression 

Mass univariate regression is the process of independently regressing multiple response variables against a single set of explantory features. It is common in any domain in which a lage number of response variables are measured, and fitting large collections of such models can benefit significantly from parallelization. 

This package provides a simple API for fitting these kinds of models. It provides a collection of `algorithms` for performing different types of mass regression, all following the [`scikit-learn`](https://github.com/scikit-learn/scikit-learn) style. It also supports providing custom algorithms directly from [`scikit-learn`](https://github.com/scikit-learn/scikit-learn). The `algorithms` are `fit` to data, returning a fitted `model` that contains regression coefficients and allows for `prediction` and `scoring` on new data. Compatible with Python 2.7+ and 3.4+. Works well alongside [`thunder`](https://github.com/thunder-project/thunder) and supprts parallelization via [`spark`](https://github.com/apache/spark), but can also be used as a standalone module on local [`numpy`](https://github.com/numpy/numpy) arrays.

## installation

```bash
pip install thunder-regression
```

## example

In this example we'll create data and fit a collection of models

```python
# generate data

from sklearn.datasets import make_regression
X, Y = make_regression(n_samples=100, n_features=3, n_informative=3, n_targets=10, noise=1.0)

# create and fit the model

from regression import LinearRegression
algorithm = LinearRegression(fit_intercept=False)
model = algorithm.fit(X, Y.T)
```

After fitting, `model.betas` is an array with the 3 coefficients for each of 10 response variables.

## usage

Import and construct an algorithm

```python
from regression import LinearRegression
algorithm = LinearRegression(fit_intercept=False)
```

Fit the algorithm to data in the form of a `samples x features` design matrix `X` and a `targets x samples` response matrix `Y`.

```python
model = algorithm.fit(X, Y)
```

The results of the fit are accessible on the fitted model, and the model can be used to score new data

```python
betas = model.betas
rsq = model.score(X, Y)
```

For all methods, `X` should be a local [`numpy`](https://github.com/numpy/numpy) array, and `Y` can be either a local [`numpy`](https://github.com/numpy/numpy) array, a [`bolt`](http://github.com/bolt-project/bolt) array, or a [`thunder`](http://github.com/thunder-project/thunder) `Series` object.

## api

### algorithm

All algorithms have the following methods:

#### `algorithm.fit(X, Y)`
Fit the algorithm to data
- `X` design matrix, dimensions `samples x features`
- `Y` collection of responses, dimensions `targets x samples`
- returns a fitted `MassRegressionModel`

#### `algorithm.fit_and_score(X, Y)`
Fit the algorithm to data and also compute scores for goodness of fits
- `X` design matrix, dimensions `samples x features`
- `Y` collection of responses, dimensions `targets x samples`
- returns array of scores and a fitted `MassRegressionModel`

### model

The result of fitting an `algorithm` is a model with the following properties and methods:

#### `model.betas`
Array of regression coefficients, dimensions `targets x features`. If an intercept was fit, it will be the 
the first feature.

#### `model.models`
Array of fitted models, dimensions `1 x targets`.

#### `model.coef_`
Array of coefficients, not including a possible intercept term, for consistency with [`scikit-learn`](https://github.com/scikit-learn/scikit-learn).

#### `model.intercept_`
Array of intercepts, for consistency with [`scikit-learn`](https://github.com/scikit-learn/scikit-learn). If no intercepts were fit, all will have values `0.0`.

#### `model.predict(X)`
Predicts the response to new inputs.
- `X` design matrix, dimensions `new samples x features`
- returns an array of responses, dimensions `targets x new samples`

#### `model.score(X, Y)`
Computes the goodness of fit (r-squared, unless otherwise stated) of the model for given data
- `X` design matrix, dimensions `samples x features`
- `Y` collection of responses, dimensions `targets x samples`
- returns an array of scores

#### `model.predict_and_score(X, Y)`
Simultaneously computes the results of `predict(X)` and `score(X, Y)`
- `X` design matrix, dimensions `samples x features`
- `Y` collection of responses, dimensions `targets x samples`
- returns an array of predictions and an array of scores

## list of algorithms

Here are all the algorithms currently available.

#### `LinearRegression(fit_intercept=False, normalize=False)`
Linear regression through ordinary least squares as implemented in scikit-learn's `LinearRegression` algorithm.
- `fit_intercept` whether or not to fit intercept terms
- `normalize` whether or not to normalize the data before fitting the models

#### `CustomRegression(algorithm)`
Use a custom regression algorithm in a mass regression analysis. The provided `algorithm` should operate on single response variables, and must conform to the `scikit-learn` API as follows
- Must implement a `.fit(X, Y)` method that takes a design matrix (`samples x features`) and a response vector and returns an object representing the fitted model.
- The returned fitted model must must have attributes `.coef_` and `.intercept_` that hold the results of the the fit (`.coef_` having dimensions `1 x features` and `.intercept_` being a scalar).
- The returned fitted model must also have methods `.predict(X)` and `.score(X, y)` (`X` having dimensions `new samples x features` and `y` having dimensions `1 x new samples`). The former should return a vector of predictions (dimensions `1 x new samples`) and the former should return a scalar score (likely r-squared).

This allows you to define an algorithm in [`scikit-learn`](https://github.com/scikit-learn/scikit-learn) and then wrap it for mass fitting, for example

```python
from regression import CusomRegression
from sklearn.linear_model import LassoCV
algorithm = CusomRegression(LassoCV(normalize=True, fit_intercept=False))
model = algorithm.fit(X, Y)
```
## tests

Run tests with 

```bash
py.test
```

Tests run locally with [`numpy`](https://github.com/numpy/numpy) by default, but the same tests can be run against a local [`spark`](https://github.com/apache/spark) installation using

```bash
py.test --engine=spark
```
