# thunder-regression
Algorithms and data structures for mass univariate regression -- independently regressing a single set of explantory variables against multiple response variables. Uses a `scikit-learn` sytle. Includes many built-in `algorithms` for different types of linear regression as well as the ability to provide custom algorithms. The `algorithms` are `fit` to data, returning a fitted `model` that contains regression coefficients and allows for `predicting` and `scoring` on new data. Compatible with Python 2.7+ and 3.4+. Works well alongside `thunder` and supprts parallelization, but can be used as a standalone module on local arrays.

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

Choose an algorithm defining the regression model

```python
from regression import LinearRegression
algorithm = LinearRegression(fit_intercept=False)
```

Or define your own algorithm

```python
from regression import CusomRegression
from sklearn.linear_model import LassoCV
algorithm = CusomRegression(LassoCV(normalize=True))
```

Fit to the data and collect the coefficients for each fit

```python
model = algorithm.fit(X, y)
# model.betas.shape returns (10, 3) -- 3 coefficients for 10 responses
```
