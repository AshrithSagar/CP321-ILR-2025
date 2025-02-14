"""
models.py
Utility for models
"""

import gmr
import matplotlib.pyplot as plt
import numpy as np

from .helpers import plot_curves


class BaseModel:
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


class LeastSquares(BaseModel):
    def __init__(self, lam=1e-2, bias=False):
        super().__init__(bias)
        self.lam = lam

    def fit(self, X, Y):
        X = self._add_bias(X)
        d = X.shape[1]
        self.w = np.linalg.inv(X.T @ X + self.lam * np.eye(d)) @ (X.T @ Y)

    def predict(self, X):
        X = self._add_bias(X)
        return X @ self.w


class LWR(BaseModel):
    def __init__(self, mvns, bias=True):
        super().__init__(bias)
        self.mvns = mvns
        self.ws = None

    def fit(self, X, Y):
        X = self._add_bias(X)
        self.ws = []
        for mvn in self.mvns:
            nfe = mvn.to_norm_factor_and_exponents(X)[0][:, None]
            W = np.linalg.inv(X.T @ (nfe * X) + 1e-6 * np.eye(X.shape[1])) @ (
                X.T @ (nfe * Y)
            )
            self.ws.append(W)

    def predict(self, X):
        X = self._add_bias(X)
        predictions = np.zeros((X.shape[0], self.ws[0].shape[1]))
        for i, mvn in enumerate(self.mvns):
            predictions += X @ self.ws[i] * mvn.pdf(X)
        return predictions


class RBFN:
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


class GMR:
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

        gmm = self.gmm
        means = gmm.means
        covariances = gmm.covariances
        priors = gmm.priors
        n_mixture = len(priors)
        n_points = X.shape[0]
        d = X.shape[1]
        X_dot = np.zeros((n_points, d))
        for i in range(n_points):
            x = X[i]
            numerator = 0
            denominator = 0
            for j in range(n_mixture):
                mean = means[j]
                covariance = covariances[j]
                prior = priors[j]
                numerator += (
                    prior * np.linalg.inv(covariance[d:, d:]) @ (x - mean[:d])
                    + mean[d:]
                )
                denominator += prior
            X_dot[i] = numerator / denominator
        return X_dot
