# Linear Model

## Linear regression
From below, $X$ is just raw data matrix not including *intercept* column
### Model setup
Assuming data $(x_i, y_i)_{i=1}^n$ follows linear relationship, i.e.
$$
    \mathbf{y} = \tilde{X}\tilde{\beta} + \boldsymbol{\varepsilon}
$$
Here $\tilde{X} = [\mathbf{1}, X]$, $\tilde{\beta} = [1, \beta]$, i.e. including intercept.   
$$
    \min_{\tilde{\beta}} (\mathbf{y} - \tilde{X}\tilde{\beta})^{T} (\mathbf{y} - \tilde{X}\tilde{\beta})
$$
### Solution 
- $\tilde{\beta} = (\tilde{X}^{T} \tilde{X})^{-1} \tilde{X}^T \mathbf{y}$  
### Sklearn implementation
- center $X$, $\mathbf{y}$ first
- `np.linalg.lstsq(X_center, y_center)` $\rightarrow$ `coef_`
- `y.mean - X.mean(axis=0).dot(coef_)` $\rightarrow$ `intercept_`  

*Remark*: centering $X$ or $\mathbf{y}$(not necessarily both) will not change the solution $\beta$, for detail, check [this](https://stats.stackexchange.com/a/108862/919). Alternatively we can make a general proof directly after centering $X$ and $\mathbf{y}$.  

Rewrite Loss function: 
$$
    \min_{c, \beta} L(c, \beta) = \sum_{i=1}^n (y_i - c - \beta^T x_i)^2   
$$
First order condition:
$$
    \frac{\partial L}{\partial c} = -2\sum_{i=1}^n (y_i - c -\beta^T x_i) = 0 \Rightarrow c = \bar{y} - \beta \bar{x} \quad (0) \\
    \Downarrow
$$

$$
\begin{aligned}
    \frac{\partial L}{\partial \beta} &= -2\sum_{i=1}^n (y_i - c -\beta^T x_i) x_i & \quad (1) \\
    & = -2\sum_{i=1}^n (y_i - \bar{y} -\beta^T (x_i - \bar{x})) x_i  & \quad (2) \\
    & = -2\sum_{i=1}^n (y_i - \bar{y} -\beta^T (x_i - \bar{x})) (x_i - \bar{x}) = 0& \quad (3)
\end{aligned}
$$

Plug $(0)$ into $(1)$, we get $(2)$. Since $\sum_{i=1}^n (y_i - \bar{y} -\beta^T (x_i - \bar{x})) \bar{x} = 0$, we get $(3)$. Since solution to $(3)$ is equivalent to $\min\limits_{\beta} \sum_{i=1}^n (y_i - \bar{y} - \beta^T (x_i - \bar{x}))^2$, we are done!



## Ridge regression  
### Model   
$$
    \min_{\beta} (\mathbf{y} - \mathbf{c} - X\beta)^T (\mathbf{y} - \mathbf{c} - X\beta) + \alpha ||\beta||_{2}^{2}
$$
$\mathbf{c}$ is intercept term. Usually, we do not penalize *constant* term.
### Solution
- $c = \bar{y} - \beta \bar{x}$  
Below, assuming $X$ and $\mathbf{y}$ have been centered.
- for $n \ge p$, $\beta = (X^T X + \alpha I_p)^{-1} X^{T} \mathbf{y}$
- for $n < p$, $\beta = X^{T} (X X^T + \alpha I_n)^{-1} \mathbf{y}$  since $(X^{T} X + \alpha I_p) X^{T} = X^{T}(X X^{T} + \alpha I_n)$  

*Remark*: centering $X$ or $\mathbf{y}$(not necessarily both) will not change the solution $\beta$, proof is similar as above.

### Sklearn implementation
- center $X$, $\mathbf{y}$ first
- fit `X_center` and `y_center`, solved by above solution (1) $n \ge p$ (2) $n < p$ $\rightarrow$ `coef_`
- `y.mean - X.mean(axis=0).dot(coef_)` $\rightarrow$ `intercept_`  
