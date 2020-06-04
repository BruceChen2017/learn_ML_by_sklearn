from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import warnings

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

clf = LinearRegression()
clf.fit(X_train, y_train) 
coef1 = np.append(clf.intercept_, clf.coef_)

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
coef11 = np.append(clf.intercept_, clf.coef_)
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
coef21 = np.append(clf.intercept_, clf.coef_)
# print(coef21)
coef22 = _ridge(X, y)
print(coef11 - coef12)
assert np.allclose(coef11, coef12, atol=1e-10, rtol=0)


# LASSO and ElaticNet
def _sign(x):
    if x > 0: return 1.0
    if x == 0: return 0
    if x < 0: return -1.0

def _elatic_net_cd(X, y, alpha, beta, tol, max_iter):
    """
        return coef_
        references: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/_cd_fast.pyx#L99
    """
    if alpha == 0 and beta == 0:
        warnings.warn("Coordinate descent with no regularization may lead to unexpected results and is discouraged.")

    n, p = X.shape
    # initial coef
    coef_ = np.zeros(p)
    resid = y - X.dot(coef_)
    z = np.square(X).sum(axis=0)

    for n_iter in range(max_iter):
        theta_m = np.abs(coef_).max()
        d_theta_m = 0.0
        for j in range(p):
            if z[j] == 0.0:
                continue
            theta_j_old = coef_[j] # previous value
            if theta_j_old != 0:
                resid += theta_j_old * X[:, j]
            rho_j = (X[:, j] * resid).sum()
            # update theta_j
            theta_j = coef_[j] = _sign(rho_j) * max([abs(rho_j) - alpha, 0]) / (z[j] + beta)
            # update resid
            if theta_j != 0:
                resid -= theta_j * X[:, j]
            # update the maximum absolute coefficient update
            d_theta_j = abs(theta_j - theta_j_old)
            d_theta_m = max(d_theta_m, d_theta_j)
        # Here, we stop when max(abs(update)) is less than max(abs(coef_before_update)) * tol
        if d_theta_m < theta_m * tol:
            break
    else:
        warnings.warn("Objective did not converge. You might want to increase the number of iterations.")
    
    return coef_

def _elatic_net(X, y, alpha, l1_ratio=0.5, tol=1e-4, max_iter=1000):
    X = X.copy()
    y = np.array(y) # copy
    n, p = X.shape
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    X = X - X_mean
    y = y - y_mean
    l1_reg = alpha * l1_ratio * n
    l2_reg = alpha * (1 - l1_ratio) * n
    coef_ = _elatic_net_cd(X, y, l1_reg, l2_reg, tol, max_iter)
    intercept_ = y_mean - X_mean.dot(coef_)
    return np.append(intercept_, coef_)

def _lasso(X, y, alpha, tol=1e-4, max_iter=1000):
    return _elatic_net(X, y, alpha, l1_ratio=1, tol=tol, max_iter=max_iter)

# generate data
np.random.seed(100)
X = np.random.randn(100, 5)*2 + 1
y = 0.5 + 0.1 * X[:, 1] + 0.2 * X[:, 4] + np.random.randn(X.shape[0]) * 2

print("==================LASSO=====================")
from sklearn.linear_model import Lasso
clf = Lasso(alpha=0.2)
clf.fit(X, y)
coef11 = np.append(clf.intercept_, clf.coef_)
coef12 = _lasso(X, y, alpha=0.2)
print(coef11)
print(coef12)

print("==================ElaticNet=====================")
from sklearn.linear_model import ElasticNet
clf = ElasticNet(alpha=0.3)
clf.fit(X, y)
coef11 = np.append(clf.intercept_, clf.coef_)
coef12 = _elatic_net(X, y, alpha=0.3, l1_ratio=0.5)
print(coef11)
print(coef12)