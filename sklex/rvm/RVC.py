# coding: utf-8
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import ClassifierMixin
from sklearn.multiclass import OneVsOneClassifier
from sklex.rvm._BaseRVM import _BaseRVM
from typing import Any


class RVC(_BaseRVM, ClassifierMixin):
    """Relevance Vector Machine for classification.
    """
    def __init__(self, n_iter_posterior=50, **kwargs) -> None:
        """Copy params to object properties, no validation."""
        self.n_iter_posterior = n_iter_posterior
        self.multi_ = OneVsOneClassifier(self)
        super(RVC, self).__init__(**kwargs)

    def get_params(self, deep=True) -> dict[str, Any]:
        """Return parameters as a dictionary."""
        params = super(RVC, self).get_params(deep=deep)
        params['n_iter_posterior'] = self.n_iter_posterior
        return params

    def _classify(self, m, phi) -> Any:
        return expit(np.dot(phi, m))

    def _log_posterior(self, m, alpha, phi, t) -> tuple[np.ndarray, np.ndarray]:

        y = self._classify(m, phi)

        log_p = -1 * (np.sum(np.log(y[t == 1]), 0) +
                      np.sum(np.log(1-y[t == 0]), 0))
        log_p = log_p + 0.5*np.dot(m.T, np.dot(np.diag(alpha), m))

        jacobian = np.dot(np.diag(alpha), m) - np.dot(phi.T, (t-y))

        return log_p, jacobian

    def _hessian(self, m, alpha, phi, t) -> np.ndarray:
        y = self._classify(m, phi)
        B = np.diag(y*(1-y))
        return np.diag(alpha) + np.dot(phi.T, np.dot(B, phi))

    def _posterior(self) -> None:
        result = minimize(
            fun=self._log_posterior,
            hess=self._hessian,
            x0=self.m_,
            args=(self.alpha_, self.phi, self.t),
            method='Newton-CG',
            jac=True,
            options={
                'maxiter': self.n_iter_posterior
            }
        )

        self.m_ = result.x
        self.sigma_ = np.linalg.inv(
            self._hessian(self.m_, self.alpha_, self.phi, self.t)
        )

    def fit(self, X, y):
        """Check target values and fit model."""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("Need 2 or more classes.")
        elif n_classes == 2:
            self.t = np.zeros(y.shape)
            self.t[y == self.classes_[1]] = 1
            return super(RVC, self).fit(X, self.t)
        else:
            self.multi_.fit(X, y)
            return self

    def predict_proba(self, X):
        """Return an array of class probabilities."""
        phi = self.kernel(X, self.relevance_)
        y = self._classify(self.m_, phi)
        return np.column_stack((1-y, y))

    def predict(self, X):
        """Return an array of classes for each input."""
        if len(self.classes_) == 2:
            y = self.predict_proba(X)
            res = np.empty(y.shape[0], dtype=self.classes_.dtype)
            res[y[:, 1] <= 0.5] = self.classes_[0]
            res[y[:, 1] >= 0.5] = self.classes_[1]
            return res
        else:
            return self.multi_.predict(X)