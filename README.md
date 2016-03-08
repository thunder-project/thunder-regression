# thunder-regression
Algorithms and data structures for mass regression -- independently regressing a single set of explantory variables against multiple sets of response variables. Uses a `scikit-learn` style, with a `MassRegression` algorithm that can be `fit` to data to produce a `MassRegressionModel`  that contains the results of all the fits and can be used to `predict` responses to new explantory data. Compatible with Python 2.7+ and 3.4+. Works well alongside `thunder` and supprts parallelization, but can be used as a standalone module on local arrays.

## installation

```bash
pip install thunder-regression
```

## example

Make some example regression data

```python
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=3, n_informative=3, n_targets=10, noise=1.0)
# X.shape returns (100, 3) -- 100 samples for 3 explanatory variables
# y.shape returns (100, 10) -- 100 samples for 10 different responses
```

Define the algorithm for fitting a single model

```python
from sklearn.linear_model import LinearRegression
alg_single = LinearRegression(fit_intercept=False)
```

Set this as the algorithm for use in a mass regression analysis

```python
from regression import MassRegression
alg = MassRegression(alg_single)
```

Fit to the data and collect the coefficients for each fit

```python
model = MassRegression.fit(X, y)
betas = model.coefs_
# betas.shape returns (10, 3) -- 3 coefficients for 10 responses
```
