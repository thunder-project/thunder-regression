# thunder-regression

Algorithms and data structures for mass univariate regression. Mass regression is defined as independently regressing multiple response targets against a single set of explantory features. This module includes a collection of `algorithms` for performing different types of mass regression, all following the `scikit-learn` style, and also supports providing custom algorithms directly from `scikit-learn`. The `algorithms` are `fit` to data, returning a fitted `model` that contains regression coefficients and allows for `predicting` and `scoring` on new data. Compatible with Python 2.7+ and 3.4+. Works well alongside `thunder` and supprts parallelization via Spark, but can also be used as a standalone module on local arrays.

## installation

```bash
pip install thunder-regression
```

## example

Make some example regression data

```python
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=3, n_informative=3, n_targets=10, noise=1.0)
```

Then create and fit a model

```python
from regression import LinearRegression
algorithm = LinearRegression(fit_intercept=False)

model = algorithm.fit(X, y.T)
```

Now `model.betas` is an array with the 3 coefficients for each of 10 responses.

## usage

First pick an algorithm. Most algorithms take parameters, for example

```python
from regression import LinearRegression
algorithm = LinearRegression(fit_intercept=False)
```

Fit the algorithm to data

```python
model = algorithm.fit(X, y.T)
```

Where `X` is an array of `samples x features` and `y` is an array of `targets x samples`. The results of the
fit are no accessible on the fitted model

```python
betas = model.betas
rsq = model.score(X, y.T)
```

## algorithm

All algorithms have the following methods

#### `algorithm.fit(X, y)`
Fit the algorithm to data
- `X` design matrix, dimensions `samples x features`
- `y` collection of responses, dimensions `targets x samples`
- returns a fitted `MassRegressionModel`

#### `algorithm.fit_and_score(X, y)`
Fit the algorithm to data and also compute scores for the goodness of fits (r-squared unless otherwise stated)
- `X` design matrix, dimensions `samples x features`
- `y` collection of responses, dimensions `targets x samples`
- returns array of scores and a fitted `MassRegressionModel`

## model

The result of fitting an `algorithm` is a model with the following properties and methods.

#### `model.betas`
Array of regression coefficients, dimensions `targets x features`. If an intercept was fit, it will be the 
the first feature.

#### `model.models`
Array of fitted models, dimensions `1 x targets`.

#### `model.coef_`
Array of coefficients, not including a possible intercept term, for consistency with scikit-learn interface.

#### `model.intercept_`
Array of intercepts, for consistency with scikit-learn. If no intercepts fit, all will have values `0.0`.

#### `model.predict(X)`
Predicts the response to new inputs.
- `X` design matrix, dimensions `new samples x features`
- returns an array or responses, dimensions `targets x new samples`

#### `model.score(X, y)`
Computes the goodness of fit (r-squared, unless otherwise stated) of the model for given data
- `X` design matrix, dimensions `samples x features`
- `y` collection of responses, dimensions `targets x samples`
- returns an array of scores

#### `model.predict_and_score(X, y)`
Simultaneously computes the results of `predict(X)` and `score(X, y)`
- `X` design matrix, dimensions `samples x features`
- `y` collection of responses, dimensions `targets x samples`
- returns an array of predictions and an array of scores

## list of algorithms

Here are all the algorithms currently available.

#### `LinearRegression(fit_intercept=False, normalize=False)`
Linear regression through ordinary least squares as implemented in scikit-learn's `LinearRegression` algorithm.
- `fit_intercept` whether or not to fit intercept terms
- `normalize` whether or not to normalize the data before fitting the models

#### `CustomRegression(algorithm)`
Use a custom single-series regression algorithm in a mass regression analysis.
- `algorithm` a single-series regression algorithm. Must conform to the `scikit-learn` API in the following ways:
    1. Must implement a `.fit(X, y)` method that takes a design matrix (`samples x features`) and a response
       vector and returns an object representing the fitted model.
    2. The returned fitted model must must have attributes `.coef_` and `.intercept_` that hold the results of the
       the fit (`.coef_` having dimensions `1 x features` and `.intercept_` being a scalar).
    3. The returned fitted model must also have methods `.predict(X)` and `.score(X, y)` (`X` having dimensions
       `new samples x features` and `y` having dimensions `1 x new samples`). The former should return a vector of
       predictions (dimensions `1 x new samples`) and the former should return a scalar score (likely r-squared).

This allows you to define an algorithm in `scikit-learn` and then wrap it directly, for example

```python
from regression import CusomRegression
from sklearn.linear_model import LassoCV
algorithm = CusomRegression(LassoCV(normalize=True, fit_intercept=False))
model = algorithm.fit(X, y)
```
