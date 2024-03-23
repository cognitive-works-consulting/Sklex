# coding: utf-8
import numpy as np
from abc import abstractmethod
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.utils.validation import check_X_y
from typing import Any, Callable, Self


class _Kernel:
    """A wrapper for kernel functions to check outputs are correct.
    """
    def __init__(self, function: Callable, bias: bool) -> None:
        self.function = function
        self.bias = bias
        
       
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:                 
        phi = self.function(X, Y)
        if len(phi.shape) != 2:
            raise ValueError("Kernel function did not return 2D matrix.")
        
        if phi.shape[0] != X.shape[0]:
            raise ValueError("Kernel function did not return matrix with"
                             "rows equal to number of data points.")
        if self.bias:
            phi: np.ndarray[Any, np.dtype[Any]] = np.append(phi, np.ones((phi.shape[0], 1)), axis=1)
        return phi


class _BaseRVM(BaseEstimator):
    """Base class for implementing Relevant Vector Machines.

    Tipping, Michael E. "Sparse Bayesian learning and the relevance vector machine."
    Journal of machine learning research 1.Jun (2001): 211-244.
    """
    def __init__(self,
                 kernel: str | Callable = "rbf",
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
        
           
        self.degree = degree
        self.coef1 = coef1
        self.coef0 = coef0
        self.n_iter = n_iter
        self.tol = tol
        self.alpha = alpha
        self.bias = bias
        self.beta = beta
        self.beta_fixed = beta_fixed
        self.threshold_alpha = threshold_alpha
        self.verbose = verbose

        if not (kernel in ["linear", "polynomial", "rbf"] or callable(kernel)):
            raise ValueError("Kernel is invalid.")
        else:
            if kernel == "linear":
                def _linear(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
                    return linear_kernel(X, Y)
                
                
                self.kernel = _Kernel(_linear, self.bias)
                
            elif kernel == "polynomial":
                def _polynomial(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
                    return polynomial_kernel(X=X, Y=Y, degree=degree, gamma=coef1, coef0=coef0)
        
        
                self.kernel = _Kernel(_polynomial, self.bias)

            elif kernel == "rbf":
                def _rbf(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
                    return rbf_kernel(X=X, Y=Y, gamma=coef1)
                
                
                self.kernel = _Kernel(_rbf, self.bias)
            else:
                self.kernel = _Kernel(kernel, self.bias) # type: ignore

            
    def _prune(self) -> None:
        """Remove basis functions based on alpha values."""
        keep_alpha = self.alpha_ < self.threshold_alpha

        if not np.any(keep_alpha):
            keep_alpha[0] = True
            if self.bias_used:
                keep_alpha[-1] = True

            if self.bias_used:
                if not keep_alpha[-1]:
                    self.bias_used = False
                self.relevance_ = self.relevance_[keep_alpha[:-1]]
            else:
                self.relevance_ = self.relevance_[keep_alpha]

            self.alpha_ = self.alpha_[keep_alpha]
            self.alpha_old = self.alpha_old[keep_alpha]
            self.gamma = self.gamma[keep_alpha]
            self.phi = self.phi[:, keep_alpha]
            self.sigma_ = self.sigma_[np.ix_(keep_alpha, keep_alpha)]
            self.m_ = self.m_[keep_alpha]
    
    @abstractmethod
    def _posterior(self):
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit the model to the training data."""
        X, y = check_X_y(X, y)

        n_samples, n_features = X.shape

        self.phi = self.kernel(X, X)

        n_basis_functions = self.phi.shape[1]

        self.relevance_ = X
        self.y = y

        self.alpha_ = self.alpha * np.ones(n_basis_functions)
        self.beta_ = self.beta

        self.m_ = np.zeros(n_basis_functions)

        self.alpha_old = self.alpha_
        
        # TODO: Rewrite this part with something more efficient.
        for i in range(self.n_iter):
            self._posterior()

            self.gamma = 1 - self.alpha_ * np.diag(self.sigma_)
            self.alpha_ = self.gamma / (self.m_ ** 2)

            if not self.beta_fixed:
                self.beta_ = (n_samples - np.sum(self.gamma))/(
                    np.sum((y - np.dot(self.phi, self.m_)) ** 2))

            self._prune()

            if self.verbose:
                print("Iteration: {}".format(i))
                print("Alpha: {}".format(self.alpha_))
                print("Beta: {}".format(self.beta_))
                print("Gamma: {}".format(self.gamma))
                print("m: {}".format(self.m_))
                print("Relevance Vectors: {}".format(self.relevance_.shape[0]))
                print()

            delta = np.amax(np.absolute(self.alpha_ - self.alpha_old))

            if delta < self.tol and i > 1:
                break

            self.alpha_old = self.alpha_

        if self.bias:
            self.bias = self.m_[-1]
        else:
            self.bias = None

        return self
    
    def get_params(self, deep=True) -> dict[str, Any]:
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
        
    def set_params(self, **parameters) -> Self:
        """Set parameters using kwargs."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
