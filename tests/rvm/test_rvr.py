import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsOneClassifier
from sklex.rvm import RVR



class BaseRVMTestCase(TestCase):

    """Automatic tests for the BaseRVM class."""

    def setUp(self):
        """Set up some common test instances."""
        self.param_test_clf = BaseRVM(
            kernel='linear',
            degree=1,
            coef1=2,
            coef0=3,
            n_iter=200,
            tol=1e-1,
            alpha=1e-2,
            threshold_alpha=1e3,
            beta=1e-3,
            beta_fixed=True,
            bias_used=False,
            verbose=True
        )

    def test__init__(self):
        """Check parameters are initialized correctly."""
        self.assertEqual(self.param_test_clf.kernel, 'linear')
        self.assertEqual(self.param_test_clf.degree, 1)
        self.assertEqual(self.param_test_clf.coef1, 2)
        self.assertEqual(self.param_test_clf.coef0, 3)
        self.assertEqual(self.param_test_clf.n_iter, 200)
        self.assertEqual(self.param_test_clf.tol, 1e-1)
        self.assertEqual(self.param_test_clf.alpha, 1e-2)
        self.assertEqual(self.param_test_clf.threshold_alpha, 1e3)
        self.assertEqual(self.param_test_clf.beta, 1e-3)
        self.assertTrue(self.param_test_clf.beta_fixed)
        self.assertFalse(self.param_test_clf.bias_used)
        self.assertTrue(self.param_test_clf.verbose)

    def test_get_params(self):
        """Check get_params returns a dictionary of the params."""
        params = self.param_test_clf.get_params()

        self.assertEqual(params, {
            'kernel': 'linear',
            'degree': 1,
            'coef1': 2,
            'coef0': 3,
            'n_iter': 200,
            'tol': 1e-1,
            'alpha': 1e-2,
            'threshold_alpha': 1e3,
            'beta': 1e-3,
            'beta_fixed': True,
            'bias_used': False,
            'verbose': True
        })

    def test_set_params(self):
        """Check that set_params sets params from dictionary."""
        params = {
            'kernel': 'poly',
            'degree': 4,
            'coef1': 5,
            'coef0': 6,
            'n_iter': 100,
            'tol': 1e-4,
            'alpha': 1e-5,
            'threshold_alpha': 1e6,
            'beta': 1e-6,
            'beta_fixed': False,
            'bias_used': True,
            'verbose': False
        }

        self.param_test_clf.set_params(**params)

        self.assertEqual(self.param_test_clf.kernel, 'poly')
        self.assertEqual(self.param_test_clf.degree, 4)
        self.assertEqual(self.param_test_clf.coef1, 5)
        self.assertEqual(self.param_test_clf.coef0, 6)
        self.assertEqual(self.param_test_clf.n_iter, 100)
        self.assertEqual(self.param_test_clf.tol, 1e-4)
        self.assertEqual(self.param_test_clf.alpha, 1e-5)
        self.assertEqual(self.param_test_clf.threshold_alpha, 1e6)
        self.assertEqual(self.param_test_clf.beta, 1e-6)
        self.assertFalse(self.param_test_clf.beta_fixed)
        self.assertTrue(self.param_test_clf.bias_used)
        self.assertFalse(self.param_test_clf.verbose)

    def test_apply_kernel_linear(self):
        """Check linear kernel is applied correctly."""
        clf = BaseRVM(kernel='linear', bias_used=False)

        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])

        phi = clf._apply_kernel(x, y)
        target = np.array([[17, 23], [39, 53]])

        np.testing.assert_array_equal(phi, target)

    def test_apply_kernel_rbf(self):
        """Check RBF kernel is applied correctly."""
        clf = BaseRVM(kernel='rbf', bias_used=False, coef1=0.5)

        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])

        phi = clf._apply_kernel(x, y)
        target = np.array([
            [1.12535175e-07, 2.31952283e-16],
            [1.83156389e-02, 1.12535175e-07]
        ])

        np.testing.assert_allclose(phi, target)

    def test_apply_kernel_poly(self):
        """Check polynomial kernel is applied correctly."""
        clf = BaseRVM(kernel='poly', bias_used=False, degree=2, coef1=1,
                      coef0=0.5)

        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])

        phi = clf._apply_kernel(x, y)
        target = np.array([
            [306.25, 552.25],
            [1560.25, 2862.25]
        ])

        np.testing.assert_allclose(phi, target)

    def test_apply_kernel_custom(self):
        """Check custom kernels are applied correctly."""
        def custom(x, y):
            return 2*x.dot(y.T)

        clf = BaseRVM(kernel=custom, bias_used=False)

        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])

        phi = clf._apply_kernel(x, y)
        target = np.array([[34, 46], [78, 106]])

        np.testing.assert_array_equal(phi, target)

    def test_apply_kernel_bias(self):
        """Check the kernel function correctly applies a bias."""
        clf = BaseRVM(kernel='linear', bias_used=True)

        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])

        phi = clf._apply_kernel(x, y)
        target = np.array([
            [17, 23, 1],
            [39, 53, 1]
        ])

        np.testing.assert_array_equal(phi, target)

    def test_apply_kernel_invalid(self):
        """Check that an invalid kernel choice raises an exception."""
        clf = BaseRVM(kernel='wrong')

        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])

        try:
            clf._apply_kernel(x, y)
        except ValueError as error:
            self.assertEqual(str(error), "Kernel selection is invalid.")
        else:
            self.fail()

    def test_apply_kernel_1D(self):
        """Check that _apply_kernel catches a non-2D phi."""
        def custom(x, y):
            return np.array([1, 2])

        clf = BaseRVM(kernel=custom, bias_used=False)

        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])

        try:
            clf._apply_kernel(x, y)
        except ValueError as error:
            self.assertEqual(
                str(error),
                "Custom kernel function did not return 2D matrix"
            )
        else:
            self.fail()

    def test_apply_kernel_row_mismatch(self):
        """Check that _apply_kernel catches a mismatch between input/output."""
        def custom(x, y):
            return np.array([[1, 2]])

        clf = BaseRVM(kernel=custom, bias_used=False)

        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])

        try:
            clf._apply_kernel(x, y)
        except ValueError as error:
            self.assertEqual(
                str(error),
                """Custom kernel function did not return matrix with rows"""
                """ equal to number of data points."""
            )
        else:
            self.fail()


class RVRTestCase(TestCase):

    """Tests for the RVR class."""

    def test_posterior(self):
        """Check the posterior over weights function returns mean and covar."""
        clf = RVR()

        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])

        clf.phi = clf._apply_kernel(x, y)

        clf.alpha_ = np.ones(3)
        clf.m_ = np.ones(3)
        clf.beta_ = 1
        clf.y = np.array([1, 1])

        clf._posterior()

        m_target = np.array([6.103885e-03, 3.750334e-08, 6.666294e-01])
        sigma_target = np.array([
            [9.997764e-01, -1.373791e-09, -6.103885e-03],
            [-1.373791e-09, 1.000000e+00, -3.750334e-08],
            [-6.103885e-03, -3.750334e-08, 3.333706e-01]
        ])

        np.testing.assert_allclose(clf.m_, m_target)
        np.testing.assert_allclose(clf.sigma_, sigma_target)

    def test_predict(self):
        """Check the predict function works with pre-set values."""
        clf = RVR(kernel='linear', bias_used=False)

        clf.relevance_ = np.array([[1, 1]])
        clf.m_ = np.array([1])

        y = clf.predict(np.array([1, 1]))
        self.assertEqual(y, 2)

    def test_fit(self):
        """Check the fit function works correctly."""
        clf = RVR(kernel='linear', threshold_alpha=1e3, verbose=True)

        X = np.array([
            [1],
            [2],
            [3],
        ])
        y = np.array([1, 2, 3])
        np.random.seed(1)
        y = y + 0.1 * np.random.randn(y.shape[0])

        clf.fit(X, y)

        m_target = np.array([0.065906, 0.131813, 0.197719, 0.159155])

        np.testing.assert_array_equal(clf.relevance_, X)
        np.testing.assert_allclose(clf.m_, m_target, rtol=1e-3)

    def test_regression_linear(self):
        """Check regression works with a linear function."""
        clf = RVR(kernel='linear', alpha=1e11)

        x = np.arange(1, 100)
        y = x + 5

        X = x[:, np.newaxis]

        clf.fit(X, y)

        score = clf.score(X, y)

        m_target = np.array([1, 5])

        self.assertGreater(score, 0.99)
        np.testing.assert_allclose(clf.m_, m_target)

        prediction, mse = clf.predict(np.array([[50]]), eval_MSE=True)
        self.assertAlmostEqual(prediction[0], 55, places=3)
        self.assertAlmostEqual(mse[0], 6.18e-6, places=3)

    def test_regression_linear_noise(self):
        """Check regression works with a linear function with added noise."""
        clf = RVR(kernel='linear', alpha=1e11)

        x = np.arange(1, 101)
        y = x + 5

        np.random.seed(1)
        y = y + 0.1 * np.random.randn(y.shape[0])

        X = x[:, np.newaxis]

        clf.fit(X, y)
        score = clf.score(X, y)

        m_target = np.array([1, 5])
        rel_target = np.array([[1]])

        self.assertGreater(score, 0.99)
        np.testing.assert_allclose(clf.m_, m_target, rtol=1e-2)
        np.testing.assert_allclose(clf.relevance_, rel_target)
        self.assertAlmostEqual(clf.beta_, 126.583, places=3)

        prediction, mse = clf.predict(np.array([[50]]), eval_MSE=True)
        self.assertAlmostEqual(prediction[0], 55.006, places=3)
        self.assertAlmostEqual(mse[0], 0.00798, places=5)

    def test_regression_sinc(self):
        """Check regression works with y=sinc(x)."""
        clf = RVR()
        x = np.linspace(0, 10, 101)
        y = np.sinc(x)

        np.random.seed(1)
        y = y + 0.1 * np.random.randn(y.shape[0])

        X = x[:, np.newaxis]

        clf.fit(X, y)
        score = clf.score(X, y)

        m_target = [
            1.117655e+00, -6.334513e-01, 5.868671e-01, -4.370936e-01,
            2.320311e-01, -4.638864e-05, -7.505325e-02, 6.133291e-02
        ]

        self.assertGreater(score, 0.85)
        np.testing.assert_allclose(clf.m_, m_target, rtol=1e-3)
        self.assertEqual(clf.relevance_.shape, (8, 1))

        prediction, mse = clf.predict(np.array([[0.5]]), eval_MSE=True)
        self.assertAlmostEqual(prediction[0], 0.611, places=3)
        self.assertAlmostEqual(mse[0], 0.00930, places=5)
