# coding: utf-8
import numpy as np
from sklearn.base import RegressorMixin
from sklex.rvm._BaseRVM import _BaseRVM


class RVR(RegressorMixin, _BaseRVM):
    """Relevance Vector Machine for Regression.
    """
    def _posterior(self) -> None:
        """Compute the posterior distriubtion over weights."""
        i_s = np.diag(self.alpha) + self.beta * np.dot(self.phi.T, self.phi)
        self.sigma_ = np.linalg.inv(i_s)
        self.m_ = self.beta * np.dot(self.sigma_, np.dot(self.phi.T, self.y))
        
    def predict(self, X, eval_MSE=False):
        """Evaluate the RVR model at x."""
        phi = self.kernel(X, self.relevance_)

        y = np.dot(phi, self.m_)

        if eval_MSE:
            MSE = (1/self.beta) + np.dot(phi, np.dot(self.sigma_, phi.T))
            return y, MSE[:, 0]
        else:
            return y
