# coding: utf-8
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklex.rvm._BaseRVM import _BaseRVM

class RVR(ClassifierMixin, _BaseRVM):
    """Relevance Vector Machine for Regression.
    """
    def fit(self, X, y):
        """Fit the RVR to the training data."""
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

            self.gamma = 1 - self.alpha_*np.diag(self.sigma_)
            self.alpha_ = self.gamma/(self.m_ ** 2)

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
       
    def _posterior(self):
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