# coding: utf-8
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from typing import Callable


class _Kernel:
    """Kernel function
    """       
    def __call__(self, kernel: Callable[[np.ndarray, np.ndarray], np.ndarray], X: np.ndarray, Y: np.ndarray, bias, **kwargs) -> np.ndarray:       
        if kernel == "linear":
            phi = linear_kernel(X, Y)
        elif kernel == "polynomial":
            phi = polynomial_kernel(X=X, Y=Y,
                                    degree=kwargs.get("degree", None),
                                    gamma=kwargs.get("coef1", None),
                                    coef0=kwargs.get("coef0", None))
        elif kernel == "rbf":
            phi = rbf_kernel(X=X, Y=Y, gamma=kwargs.get("coef1", None))
        else:
            phi = kernel(X, Y)
 
        if len(phi.shape) != 2:
            raise ValueError("Kernel function did not return 2D matrix.")
        
        if phi.shape[0] != X.shape[0]:
            raise ValueError("Kernel function did not return matrix with"
                             "rows equal to number of data points.")
        
        if bias:
            phi = np.append(phi, np.ones((phi.shape[0], 1)), axis=1)
        return phi


class _BaseRVM(BaseEstimator):
    """Base class for implementing Relevant Vector Machines.

    Tipping, Michael E. "Sparse Bayesian learning and the relevance vector machine."
    Journal of machine learning research 1.Jun (2001): 211-244.
    """
    def __init__(self,
                 kernel: str | Callable[[np.ndarray, np.ndarray], np.ndarray] = "rbf",
                 degree: int = 3,
                 coef1: int | None = None,
                 coef0: float = 0.0,
                 n_iter: int = 3000,
                 tol: float = 1e-3,
                 alpha: float = 1e-6,
                 threshold_alpha: float = 1e9,
                 beta: float = 1.e-6,
                 beta_fixed: bool = False,
                 bias: bool = True,
                 verbose=False) -> None:
        
        if not (kernel in ["linear", "polynomial", "rbf"] or callable(kernel)):
            raise ValueError("Kernel is invalid.")
        else:
            self.kernel = kernel
        self.degree = degree
        self.coef1 = coef1
        self.coef0 = coef0
        self.n_iter = n_iter
        self.tol = tol
        self.alpha = alpha
        self.beta = beta
        self.beta_fixed = beta_fixed
        self.treshold_alpha = threshold_alpha
        self.verbose = verbose
        
        def get_params(self):
            """Return parameters as a dictionary."""
            params = {
                'kernel': self.kernel,
                'degree': self.degree,
                'coef1': self.coef1,
                'coef0': self.coef0,
                'n_iter': self.n_iter,
                'tol': self.tol,
                'alpha': self.alpha,
                'threshold_alpha': self.threshold_alpha,
                'beta': self.beta,
                'beta_fixed': self.beta_fixed,
                'bias_used': self.bias,
                'verbose': self.verbose
            }
            return params
        
        def set_params(self, **parameters):
            """Set parameters using kwargs."""
            for parameter, value in parameters.items():
                setattr(self, parameter, value)
            return self