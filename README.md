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
model = algorithm.fit(X, y)
```

Where `X` is an array of `samples x features` and `y` is an array of `targets x samples`.

## model

[ FILL IN ]

## algorithms

##### `LinearRegression(fit_intercept=False, normalize=False).fit(X, y)`

[ FILL IN ]

##### `CustomRegression(Algorithm).fit(X, y)`

You can define an algorithm in `scikit-learn` and then wrap it directly, for example

```python
from regression import CusomRegression
from sklearn.linear_model import LassoCV
algorithm = CusomRegression(LassoCV(normalize=True, fit_intercept=False))
model = algorithm.fit(X, y)
```