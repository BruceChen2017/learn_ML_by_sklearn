# Linear Model

## Linear regression
From below, $X \in \mathbf{R}^{n \times p}$ is just raw data matrix not including *intercept* column
### Model
Assuming data $(x_i, y_i)_{i=1}^n$ follows linear relationship, i.e.
$$
    \mathbf{y} = \tilde{X}\tilde{\theta} + \boldsymbol{\varepsilon}
$$
Here $\tilde{X} = [\mathbf{1}, X]$, $\tilde{\theta} = [1, \theta]$, i.e. including intercept.   
$$
    \min_{\tilde{\theta}} (\mathbf{y} - \tilde{X}\tilde{\theta})^{T} (\mathbf{y} - \tilde{X}\tilde{\theta})
$$
### Solution 
- $\tilde{\theta} = (\tilde{X}^{T} \tilde{X})^{-1} \tilde{X}^T \mathbf{y}$  
### Sklearn implementation
- center $X$, $\mathbf{y}$ first
- `np.linalg.lstsq(X_center, y_center)` $\rightarrow$ `coef_`
- `y.mean - X.mean(axis=0).dot(coef_)` $\rightarrow$ `intercept_`  

*Remark*: centering $X$ or $\mathbf{y}$(not necessarily both) will not change the solution $\theta$, for detail, check [this](https://stats.stackexchange.com/a/108862/919). Alternatively we can make a general proof directly after centering $X$ and $\mathbf{y}$.  

>Rewrite Loss function: 
>$$
>    \min_{c, \theta} L(c, \theta) = \sum_{i=1}^n (y_i - c - \theta^T x_i)^2   
>$$
>First order condition:
>$$
>    \frac{\partial L}{\partial c} = -2\sum_{i=1}^n (y_i - c -\theta^T x_i) = 0 \Rightarrow c = \bar{y} - \theta \bar{x} \quad (0) \\
>    \Downarrow
>$$
>
>$$
>\begin{aligned}
>    \frac{\partial L}{\partial \theta} &= -2\sum_{i=1}^n (y_i - c -\theta^T x_i) x_i & \quad (1) \\
>    & = -2\sum_{i=1}^n (y_i - \bar{y} -\theta^T (x_i - \bar{x})) x_i  & \quad (2) \\
>    & = -2\sum_{i=1}^n (y_i - \bar{y} -\theta^T (x_i - \bar{x})) (x_i - \bar{x}) = 0& \quad (3)
>\end{aligned}
>$$
>
>Plug $(0)$ into $(1)$, we get $(2)$. Since $\sum_{i=1}^n (y_i - \bar{y} -\theta^T (x_i - \bar{x})) \bar{x} = 0$, we get $(3)$. Since solution to $(3)$ is equivalent to $\min\limits_{\theta} \sum_{i=1}^n (y_i - \bar{y} - \theta^T (x_i - \bar{x}))^2$, we are done!  
Actually, this argument works with additional convex penalty only for $\theta$



## Ridge regression  
### Model   
$$
    \min_{\theta} (\mathbf{y} - \mathbf{c} - X\theta)^T (\mathbf{y} - \mathbf{c} - X\theta) + \alpha ||\theta||_{2}^{2}
$$
$\mathbf{c}$ is intercept term. Usually, we do not penalize *constant* term.
### Solution
- $c = \bar{y} - \theta \bar{x}$  
Below, assuming $X$ and $\mathbf{y}$ have been centered.
- for $n \ge p$, $\theta = (X^T X + \alpha I_p)^{-1} X^{T} \mathbf{y}$
- for $n < p$, $\theta = X^{T} (X X^T + \alpha I_n)^{-1} \mathbf{y}$  since $(X^{T} X + \alpha I_p) X^{T} = X^{T}(X X^{T} + \alpha I_n)$  

*Remark*: centering $X$ or $\mathbf{y}$(not necessarily both) will not change the solution $\theta$, proof is similar as above.

### Sklearn implementation
- center $X$, $\mathbf{y}$ first
- fit `X_center` and `y_center`, solved by above solution (1) $n \ge p$ (2) $n < p$ $\rightarrow$ `coef_`
- `y.mean - X.mean(axis=0).dot(coef_)` $\rightarrow$ `intercept_`  

## LASSO and ElasticNet
### Model  
#### LASSO
$$
    \min_{c, \theta} \frac{1}{2} (\mathbf{y} - c - X\theta)^{T}(\mathbf{y} - c - X\theta) + \alpha ||\theta||_{1}
$$
As before, $\theta = \argmin\limits_{\theta} \frac{1}{2} (\mathbf{y}_c - X_c\theta)^{T}(\mathbf{y}_c- X_c\theta) + \alpha ||\theta||_{1}$, $\mathbf{y}_c$ and $X_c$ are centered form.  
LASSO can select variables, which means some $\theta_j$ would be zero, but there are some limitations. When $p > n$, lasso can only select at most $n$ variables. ElasticNet can address this problem.
#### ElasticNet
$$
    \min_{c, \theta} \frac{1}{2n} (\mathbf{y} - c - X\theta)^{T}(\mathbf{y} - c - X\theta) + \alpha \rho ||\theta||_{1} + \frac{\alpha (1-\rho)}{2} ||\theta||_{2}^2 
$$    

### Solution
#### LASSO
- For $\theta$, use coordinate descent to update $\theta$ for $T$ iteration, for detail,check [this](https://xavierbourretsicotte.github.io/lasso_derivation.html).  
  $$
    \left\{\begin{array}{ll}
    \theta_{j}=\frac{\rho_{j}+\alpha}{z_{j}} & \text { for } \rho_{j}<-\alpha \\
    \theta_{j}=0 & \text { for }-\alpha \leq \rho_{j} \leq \alpha \\
    \theta_{j}=\frac{\rho_{j}-\alpha}{z_{j}} & \text { for } \rho_{j}>\alpha
    \end{array}\right.
  $$  
  Here $\rho_j = \sum_{i=1}^n x_{ij}(y_i - \sum_{k \ne j}^p x_{ik} \theta_k)$, $z_j = \sum_{i=1}^n x_{ij}^2$, and $(x_i, y_i)$ has been centered.
- For $c$, $c = \bar{y} - \bar{x} \theta$ after $T$ iteration of $\theta$

#### ElasticNet
- For $\theta$, same method as lasso.
  $$
    \left\{\begin{array}{ll}
    \theta_{j}=\frac{\rho_{j}+\alpha}{z_{j} + \beta} & \text { for } \rho_{j}<-\alpha \\
    \theta_{j}=0 & \text { for }-\alpha \leq \rho_{j} \leq \alpha \\
    \theta_{j}=\frac{\rho_{j}-\alpha}{z_{j} + \beta} & \text { for } \rho_{j}>\alpha
    \end{array}\right.
  $$
  Here, $\alpha = \alpha * \rho * n$, $\beta = \alpha * (1 - \rho) * n \quad (\#)$
- For $c$, $c = \bar{y} - \bar{x} \theta$ after $T$ iteration of $\theta$

### Sklearn implementation  
In sklearn, lasso fits as a special case of elasticnet with `l1_ratio=1`, and $\alpha$ and $\beta$ are redefined as $(\#)$ above. For stopping rule, sklearn use [duality gap](https://arxiv.org/abs/1505.03410)