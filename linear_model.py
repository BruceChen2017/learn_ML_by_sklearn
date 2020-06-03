from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

boston = load_boston()
# description of dataset
# print(boston.DESCR)
labels = np.append(["intercept"], boston.feature_names)
print(labels)

P = 5
X = boston.data[:,:P]
print("(n_samples, p_features):", X.shape)
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=42)


print("================linear regression==========================")
# linear model
# internally
# by default, center X and y first
# fit both centered X and y by `np.linalg.lstsq`
# this will not change estimated coef with original X,y but intercept
# intercept would still be calculated by usual way: y_mean - beta * x_mean
def _linreg(X, y):
    print("linear regression: least square method")
    X_center = X - X.mean(axis=0)
    y_center = y - y.mean()
    coef_ = np.linalg.lstsq(X_center, y_center, rcond=None)[0]
    intercept_ = y.mean() - coef_.dot(X.mean(axis=0))
    return np.append(intercept_, coef_)

reg = LinearRegression()
reg.fit(X_train, y_train) 
coef1 = np.append(reg.intercept_, reg.coef_)

coef2 = _linreg(X_train, y_train)
print(coef1 - coef2)
assert np.allclose(coef1, coef2, atol=1e-10, rtol=0)

# estimated coef including intercept
print("linear regression: directly solved without centering")
design_matrix = np.hstack([np.ones(len(y_train)).reshape(-1,1), X_train])
coef3 = np.linalg.solve(design_matrix.T.dot(design_matrix), design_matrix.T.dot(y_train))
print(coef1 - coef3)
assert np.allclose(coef1, coef3, atol=1e-10, rtol=0)
"""
# print estimated coef
COL_LEN = 10
col_name = ""
coef_vals = ""
for lab in labels[:len(coef)]:
     col_name += ("{:>" + str(COL_LEN) + "}|").format(lab)
for co in coef:
     coef_vals += ("{:>" + str(COL_LEN) + ".5f}|").format(co)
print("estimated coef:")
print(col_name)
print(coef_vals)
"""

print("====================ridge regression====================")
# ridge regression
# internally 
def _ridge(X, y):
    """
        return (intercept, coef)
    """
    alpha = 1
    X_center = X - X.mean(axis=0)
    y_center = y - y.mean()
    # y_center = y_center.reshape(-1, 1)
    n, p = X_center.shape
    if n < p:
        print("ridge regression: n < p")
        K = X_center.dot(X_center.T)
        K.flat[::n+1] += alpha
        _a = np.linalg.solve(K, y)
        coef_ = np.dot(X_center.T, _a)
        intercept_ = y.mean() - coef_.dot(X.mean(axis=0))
    else:
        print("ridge regression: n >= p")
        A = X_center.T.dot(X_center)
        A.flat[::p + 1] += alpha
        Xy = X_center.T.dot(y_center)
        coef_ = np.linalg.solve(A, Xy)
        intercept_ = y.mean() - coef_.dot(X.mean(axis=0))
    
    return np.append(intercept_, coef_)

# n >= p
n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)

clf = Ridge()
clf.fit(X, y)
coef11 = [clf.intercept_, *clf.coef_]
# print(coef11)
coef12 = _ridge(X, y)
print(coef11 - coef12)
assert np.allclose(coef11, coef12, atol=1e-10, rtol=0)

# n < p
n_samples, n_features = 5, 10
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)

clf = Ridge()
clf.fit(X, y)
coef21 = [clf.intercept_, *clf.coef_]
# print(coef21)
coef22 = _ridge(X, y)
print(coef11 - coef12)
assert np.allclose(coef11, coef12, atol=1e-10, rtol=0)