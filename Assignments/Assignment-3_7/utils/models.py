"""
models.py
Utility for models
"""

from abc import ABC, abstractmethod

import gmr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from .helpers import plot_curves


class BaseModelABC(ABC):
    def __init__(self, bias=False):
        self.bias = bias

    def _add_bias(self, X):
        """
        Add bias term to the input data

        Parameters
        ----------
        X : array of shape (n_points, d)

        Returns
        -------
        X : array of shape (n_points, d+1)
        """
        if self.bias:
            return np.hstack([X, np.ones((X.shape[0], 1))])
        return X

    def ode_differential(self, x, t):
        """
        function used for rk4 simulation
        """
        return self.predict(x[None])[0]

    def rk4_sim(self, t0, t_end, x0, f, dt=1e-3):
        """
        simlution done with rk4

        Returns
        -------
        x : shape (n_steps,self.d)
        t : shape - (n_steps,)
        """
        # Calculate slopes
        x, t = x0, t0
        x_list, t_list = [x0], [t0]
        while t <= t_end:
            k1 = dt * f(x, t)
            k2 = dt * f(x + k1 / 2.0, t + dt / 2.0)
            k3 = dt * f(x + k2 / 2.0, t + dt / 2.0)
            k4 = dt * f(x + k3, t + dt)

            # Calculate new x and y
            x = x + 1.0 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t = t + dt
            x_list.append(x)
            t_list.append(t)

        return np.array(x_list), np.array(t_list)

    def imitate(self, x0, t_end=5):
        """
        imitation with new starting point

        Parameters
        ----------
        x0  - starting point - shape (self.d,)

        Returns
        -------
        x_rk4 - simulated data  - shape (n_steps,self.d)
        t_rk4 - time - shape (n_steps,)
        """
        f = self.ode_differential
        return self.rk4_sim(0, t_end, x0, f)

    @abstractmethod
    def fit(self, X, Y):
        """
        Fit the model to the data X and Y
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict using the fitted model.
        """
        pass


class LeastSquares(BaseModelABC):
    def __init__(self, lam=1e-2, bias=False):
        super().__init__(bias)
        self.lam = lam

    def fit(self, X, Y):
        """
        store the weight matrix into self.w

        self.w should be array of shape (3,2) if bias is enabled and
        (2,2) if bias is False


        params:
            X: data of shape (n_points,2)
            Y: X_dot of shape(n_points,2)
        """

        X = self._add_bias(X)
        d = X.shape[1]
        self.w = np.linalg.inv(X.T @ X + self.lam * np.eye(d)) @ (X.T @ Y)

    def predict(self, X):
        """
        returns prediction from the model X_dot

        params:
            X: array of shape (n_points,2)
        returns:
            predicted X_dot: array of shape (n_points,2)
        """

        X = self._add_bias(X)
        return X @ self.w


class LWR(BaseModelABC):
    def __init__(self, mvns: list[gmr.MVN], bias=True):
        """
        gaussians for weights
        params:
          mvns: list of gaussians
          bias: x@weight+bias
        """
        super().__init__(bias)
        self.mvns = mvns
        self.ws = None

    def fit(self, X, Y):
        """
        gets the weight matrix into self.ws

        self.ws is a list of length len(self.mvns), with each element as weight matrix (3,2) if bias is True, (2,2) if bias is False

        params:
            X: data of shape (n_points,2)
            Y: X_dot of shape(n_points,2)
        """

        X = self._add_bias(X)
        self.ws = []

        for mvn in self.mvns:
            weights = mvn.to_probability_density(X)
            W = np.diag(weights)
            XtWX = X.T @ W @ X
            XtWY = X.T @ W @ Y
            W_opt = np.linalg.solve(XtWX, XtWY)
            self.ws.append(W_opt)

    def predict(self, X):
        """
        returns prediction from the model =>X_dot

        merge the multple predictions from different weight matrices.

        params:
            X: array of shape (n_points,2)
        returns:
            predicted X_dot: array of shape (n_points,2)
        """

        X = self._add_bias(X)
        predictions = np.zeros((X.shape[0], self.ws[0].shape[1]))
        total_weight = np.zeros(X.shape[0])

        for i in range(len(self.mvns)):
            mvn = self.mvns[i]
            W_opt = self.ws[i]
            weights = mvn.to_probability_density(X)
            predictions += (X @ W_opt) * weights[:, np.newaxis]
            total_weight += weights

        return predictions / np.maximum(total_weight[:, np.newaxis], 1e-8)


class RBFN(BaseModelABC):
    def __init__(self, mvns, bias=True):
        """
        gaussians for weights
        """
        self.mvns = mvns
        self.bias = bias

    def _compute_activations(self, X):
        """
        Compute activations for each Gaussian basis function.
        """
        activations = np.array(
            [mvn.to_norm_factor_and_exponents(X)[1] for mvn in self.mvns]
        ).T
        return np.exp(activations)

    def fit(self, X, Y):
        """
        store the weight matrix into self.w

        self.w should be array of shape (len(self.mvns)+1,2) if bias is enabled and
        (len(self.mvns),2) if bias is False


        params:
            X: data of shape (n_points,2)
            Y: X_dot of shape(n_points,2)
        """

        Phi = self._compute_activations(X)
        if self.bias:
            Phi = np.hstack([Phi, np.ones((Phi.shape[0], 1))])
        self.w = np.linalg.pinv(Phi) @ Y

    def predict(self, X):
        """
        returns prediction from the model X_dot

        params:
            X: array of shape (n_points,2)
        returns:
            predicted X_dot: array of shape (n_points,2)
        """

        Phi = self._compute_activations(X)
        if self.bias:
            Phi = np.hstack([Phi, np.ones((Phi.shape[0], 1))])
        return Phi @ self.w


class GMR(BaseModelABC):
    def __init__(self, n_mixture):
        """
        params:
            n_mixture: number of gaussians
        """
        self.gmm = gmr.GMM(n_mixture)

    def fit(self, X, Y, data):
        """
        the GMR package has EM algorithm that can be used to train the GMM

        params:
            X: data of shape (n_points,2)
            Y: X_dot of shape(n_points,2)
        """
        Z = np.hstack([X, Y])
        self.gmm = self.gmm.from_samples(Z, n_iter=100, init_params="kmeans++")

    def visualise_fit(self, data):
        """
        Visualize the fitted GMM components and plot the training data.
        """
        splot = plt.subplot(111)
        mvns = [
            gmr.MVN(
                mean=self.gmm.means[i][:2], covariance=self.gmm.covariances[i][:2, :2]
            )
            for i in range(len(self.gmm.priors))
        ]
        plot_curves(data)
        splot = plt.subplot(111)
        for mvn in mvns:
            gmr.plot_error_ellipse(splot, mvn, factors=[1])
        print("posterioir:", self.gmm.priors)
        print("shapes of means:", self.gmm.means.shape)
        print("shapes of covariances:", self.gmm.covariances.shape)

    def predict(self, X):
        """
        You can use the GMR package, or also can write your own code

        returns prediction from the model X_dot

        params:
            X: array of shape (n_points,2)
        returns:
            predicted X_dot: array of shape (n_points,2)
        """
        # Note: You cannot use gmm.predict or gmm.condition, use the formula for conditioning of gaussian mixture)

        n_points = X.shape[0]
        n_components = len(self.gmm.priors)
        d_x, d_y = 2, 2

        weights = np.zeros((n_points, n_components))
        weighted_cond_means = np.zeros((n_points, d_y))

        for k in range(n_components):
            mu = self.gmm.means[k]
            mu_x, mu_y = mu[:d_x], mu[d_x:]

            Sigma = self.gmm.covariances[k]
            Sigma_xx = Sigma[:d_x, :d_x]
            Sigma_yx = Sigma[d_x:, :d_x]

            inv_Sigma_xx = np.linalg.inv(Sigma_xx)
            det_Sigma_xx = np.linalg.det(Sigma_xx)
            norm_const = 1.0 / (2 * np.pi * np.sqrt(det_Sigma_xx))

            diff = X - mu_x
            # Efficient quadratic form computation using Einstein summation convention
            exponent = -0.5 * np.einsum("ij,ij->i", diff @ inv_Sigma_xx, diff)
            pdf = norm_const * np.exp(exponent)

            weights[:, k] = self.gmm.priors[k] * pdf

            # E[Y|x, k] = mu_y + Sigma_yx * inv(Sigma_xx) * (x - mu_x)
            A_k = Sigma_yx @ inv_Sigma_xx
            cond_mean = mu_y + diff @ A_k.T

            weighted_cond_means += weights[:, k : k + 1] * cond_mean

        weight_sum = np.sum(weights, axis=1, keepdims=True)
        weight_sum[weight_sum == 0] = 1e-10
        return weighted_cond_means / weight_sum


class GPR(BaseModelABC):
    def __init__(self, kernel, **kwargs):
        """
        initializing the gpr model with a kernel
        """
        self.gpr = GaussianProcessRegressor(kernel, n_targets=2, **kwargs)

    def sample(self, X):

        return self.gpr.sample_y(X, n_samples=1)[:, :, 0]

    def fit(self, X, Y):
        """
        store the coefficients in self.coeff

        params:
            X: data of shape (n_points,2)
            Y: X_dot of shape(n_points,2)
        """

        self.gpr.fit(X, Y)
        self.X_train = X
        K = self.gpr.kernel_(X) + np.eye(X.shape[0]) * self.gpr.alpha
        self.coeff = np.linalg.solve(K, Y)

    def predict(self, X_star):
        """
        returns prediction from the model X_dot

        params:
            X: array of shape (n_points,2)
        returns:
            predicted X_dot: array of shape (n_points,2)
        """

        K_star = self.gpr.kernel_(X_star, self.X_train)
        return K_star.dot(self.coeff)
