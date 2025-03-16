"""
models.py
Utility for models
"""

import itertools
from abc import ABC, abstractmethod
from functools import partial

import gmr
import matplotlib.pyplot as plt
import numpy as np
import torch
from gmr import GMM, MVN, plot_error_ellipses
from numpy.typing import NDArray
from scipy import optimize
from sklearn.gaussian_process import GaussianProcessRegressor

from .helpers import plot_curves, plot_curves3
from .utils import plot_curves3_ax


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


class SEDS(BaseModelABC):
    def __init__(self, attractor, n_mixture):

        assert attractor.ndim == 1
        self.attractor = attractor
        self.n_mixture = n_mixture
        self.d = len(attractor)

    def initialize_params(self, y, yd):
        """
        initializing params(for furthur constrained optimization),
        from the algorithm mentioned in the paper

        parameters
        -----------
        y: state space ,shape - (number of points,self.d)
        yd: derivative of state space - (number of points,self.d)

        Returns
        --------
        priors : shape-(self.n_mixture,)
        means : shape-(self.n_mixture,2*self.d)
        covars : shape-(self.n_mixture,2*self.d,2*self.d)

        """

        priors = np.ones(self.n_mixture) / self.n_mixture

        means = np.zeros((self.n_mixture, 2 * self.d))
        for k in range(self.n_mixture):
            means[k, : self.d] = y[np.random.randint(0, y.shape[0])]
            means[k, self.d :] = yd[np.random.randint(0, yd.shape[0])]

        covars = np.zeros((self.n_mixture, 2 * self.d, 2 * self.d))
        for k in range(self.n_mixture):
            covars[k] = np.eye(2 * self.d)

        return priors, means, covars

    def cov2A(self, cov):
        """
        helper function for converting the full covariance matrices to Ak matrices

        Parameters
        ----------
        cov : covariance matrices , shape -(self.n_mixture,2*self.d,2*self.d)

        Returns
        -------
        As : A matrices , shape - (self.n_mixture,self.d,self.d)
        """
        As = []
        for k in range(self.n_mixture):
            As.append(
                cov[k][self.d :, : self.d] @ np.linalg.inv(cov[k][: self.d, : self.d])
            )
        return np.stack(As)

    def _to_optim_vector(self, priors, means, covars):
        """
        converting priors,means,covars to a single vector for optimization purpose, with respective transformations

        Parameters
        ----------
        priors : shape-(self.n_mixture,)
        means : shape-(self.n_mixture,2*self.d)
        covars : shape-(self.n_mixture,2*self.d,2*self.d)

        Returns
        -------
        x :  vector shape - (self.n_mixture+self.n_mixture*self.d+self.n_mixture*2*self.d*2*self.d,)

        """
        chol_decomps = np.array([np.linalg.cholesky(covar) for covar in covars])
        return np.hstack(
            (np.log(priors), means[:, : self.d].flatten(), chol_decomps.flatten())
        )

    def _from_optim_vector(self, x):
        """
        given the one dimension vector , gets back the priors,means,covariance

        Parameters
        ----------
        x : vector shape - (self.n_mixture+self.n_mixture*self.d+self.n_mixture*2*self.d*2*self.d,)

        Returns
        --------
        priors : shape-(self.n_mixture,)
        means : shape-(self.n_mixture,2*self.d)
        covars : shape-(self.n_mixture,2*self.d,2*self.d)

        """

        # priors
        priors = np.exp(x[: self.n_mixture])
        priors /= priors.sum()

        # covars
        Ls = x[-self.n_mixture * 2 * self.d * 2 * self.d :].reshape(
            self.n_mixture, 2 * self.d, 2 * self.d
        )
        covars = Ls @ np.transpose(Ls, (0, 2, 1))

        # means
        As = self.cov2A(covars)
        mu1 = x[self.n_mixture : -self.n_mixture * 2 * self.d * 2 * self.d].reshape(
            -1, self.d
        )
        mu2 = As @ ((mu1 - self.attractor)[..., None])
        mu2 = mu2[:, :, 0]
        means = np.concatenate([mu1, mu2], axis=-1)

        return priors, means, covars

    def _torch_from_optim_vector(self, x0):
        """
        same as '_from_optim_vector' method but with input as pytorch tensor instead of numpy
        so that, computational graph form for the usage of autograd package
        """

        # prior
        priors = torch.exp(x0[: self.n_mixture])
        priors = priors / priors.sum()

        # covars
        Ls = x0[-self.n_mixture * 2 * self.d * 2 * self.d :].reshape(
            self.n_mixture, 2 * self.d, 2 * self.d
        )
        covars = Ls @ torch.transpose(Ls, 1, 2)

        # means
        As = []
        for k in range(self.n_mixture):
            As.append(
                covars[k][self.d :, : self.d]
                @ torch.linalg.inv(covars[k][: self.d, : self.d])
            )
        As = torch.stack(As)
        mu1 = x0[self.n_mixture : -self.n_mixture * 2 * self.d * 2 * self.d].reshape(
            -1, self.d
        )
        mu2 = As @ ((mu1 - torch.from_numpy(self.attractor))[..., None])
        mu2 = mu2[:, :, 0]
        means = torch.concatenate([mu1, mu2], axis=-1)

        return priors, means, covars

    def objective(self, x, y, yd):
        """
        negative loglikelihood as objective function, with logsumexp trick for computational stability


        Paramaters
        ------------
        x : parameter for computational graph,
            shape - (self.n_mixture+self.n_mixture*self.d+self.n_mixture*2*self.d*2*self.d,)
        y: state space ,shape - (number of points,self.d)
        yd: derivative of state space - (number of points,self.d)

        Returns
        -------
        f : objective value at x
        g : gradient of f at x
        """
        x = torch.from_numpy(x)
        Y = np.concatenate([y, yd], axis=-1)
        Y = torch.from_numpy(Y)

        # making x as parameters
        x.requires_grad_()
        priors, means, covars = self._torch_from_optim_vector(x)

        # objective calculation
        xm = (Y[:, None, :] - means[None])[..., None]
        q = -((xm.transpose(-1, -2) @ torch.linalg.inv(covars)) @ xm)[:, :, 0, 0] / 2
        q = q - (torch.log(torch.linalg.det(covars))[None] / 2)
        q = q + torch.log(priors)
        f = -torch.mean(torch.logsumexp(q, dim=1))

        # calculates gradient wrt x
        f.backward()

        return f.detach().numpy(), x.grad.detach().numpy()

    def constraint_func(self, x, requires_grad=False):
        """
        constraints are A+A.T<0 (negative definite)
        returns 'constraint value'  or 'derivatives of constraints' depending on the requires_grad parameter

        Parameters
        ----------
        x : parameter for computational graph,
            shape - (self.n_mixture+self.n_mixture*self.d+self.n_mixture*2*self.d*2*self.d,)
        requires_grad : if False - only constraint_value, if True - only gradient of the constraint

        Returns
        --------
        Cs : constraints values -shape - (self.n_mixture*self.d,)
                or
        gCs : gradient of constraints - shape - (self.n_mixture*self.d,x.shape)

        """

        # making x , as leaf of the conmputational graph
        x = torch.from_numpy(x)
        if requires_grad:
            x.requires_grad_()

        # covars
        Ls = x[-self.n_mixture * 2 * self.d * 2 * self.d :].reshape(
            self.n_mixture, 2 * self.d, 2 * self.d
        )
        covars = Ls @ torch.transpose(Ls, 1, 2)

        # A matrices
        As = []
        for k in range(self.n_mixture):
            As.append(
                covars[k][self.d :, : self.d]
                @ torch.linalg.inv(covars[k][: self.d, : self.d])
            )
        As = torch.stack(As)

        # B matrices
        Bs = As + torch.transpose(As, 1, 2)

        # constraints
        Cs = []
        for i in range(self.d):
            Cs.append(torch.linalg.det(Bs[:, : i + 1, : i + 1]) * ((-1) ** (i + 1)))
        Cs = torch.concat(Cs)

        # if only constraint values are required
        if not requires_grad:
            return Cs.detach().numpy()

        # if only gradient values are required
        gCs = []
        for i in range(len(Cs)):
            gCs.append(
                torch.autograd.grad(
                    Cs[i], x, torch.ones(Cs[i].shape), retain_graph=True
                )[0]
            )
        gCs = torch.stack(gCs)

        return gCs.detach().numpy()

    def predict(self, X):
        """
        returns prediction from the model X_dot

        params:
            X: array of shape (n_points,2)
        returns:
            predicted X_dot: array of shape (n_points,2)
        """

        return self.gmm.predict(range(self.d), X)

    def ode_differential(self, x, t, gmm):
        """
        function used for rk4 simulation
        """
        return gmm.predict(range(self.d), x[None])[0]

    def imitate(self, x0, t_end=1):
        """
        imitation with new starting point to reach the attractor

        Parameters
        ----------
        x0  - starting point - shape (self.d,)

        Returns
        -------
        x_rk4 - simulated data with trained GMM - shape (n_steps,self.d)
        t_rk4 - time - shape (n_steps,)
        """
        f = partial(self.ode_differential, gmm=self.gmm)
        return self.rk4_sim(0, t_end, x0, f)

    def fit(self, x, xd):
        """
        initialize the params,"SLSQP" for optimization, as given in the paper

        Parametes
        ---------
        x: state space ,shape - (number of points,self.d)
        xd: derivative of state space - (number of points,self.d)

        Return:
        -------
        No Returns (Stores the GMM if optimization is successfull)

        """

        # initialization of parameters
        priors, means, covars = self.initialize_params(x, xd)

        # converting to single vector
        x0 = self._to_optim_vector(priors, means, covars)

        # inequlity constraints and their derivatives
        ineq_const = {
            "type": "ineq",
            "fun": lambda a: self.constraint_func(a, requires_grad=False),
            "jac": lambda a: self.constraint_func(a, requires_grad=True),
        }

        # optimize
        res = optimize.minimize(
            partial(self.objective, y=x, yd=xd),
            x0,
            method="SLSQP",
            jac=True,
            constraints=[ineq_const],
            options={"maxiter": 1000},
        )

        # verbose
        print("initial_object_value:", self.objective(x0, x, xd)[0])
        print("final objective_value:", res.fun)

        priors, means, covars = self._from_optim_vector(res.x)
        self.gmm = GMM(self.n_mixture, priors=priors, means=means, covariances=covars)


class DMP(BaseModelABC):
    def __init__(self, alpha: float, n_features: int):
        """
        set the hyperparameters
        """
        # set the hyperparameters from the inputs
        # self.alpha,self.beta,self.alpha_z,self.n_features
        # information about the values of the parameters is given in the theory part

        self.alpha = alpha
        self.beta = self.alpha / 4.0
        self.alpha_z = self.alpha / 3.0
        self.n_features = n_features

    def derivative(self, x: NDArray) -> NDArray:
        """
        difference method for calculating derivative

        params:
            x: array of shape (number of trajectories,number of timesteps,2)

        returns
            xd: array of shape (number of trajectories,number of timesteps,2)
        """
        T, ndims = x.shape
        return np.vstack((np.zeros((1, ndims)), np.diff(x, axis=0) / self._dt))

    def get_features(self, z: NDArray, cz: NDArray, hz: NDArray) -> NDArray:
        """
        Returns the PSI matrix as given in the theory part

        input:
            z : array of shape (n_steps,)
            cz: centers of gaussian in phase domain ,array of shape(n_features,)
            hz: scaling factor for each of the gaussian ,array of shape (n_features,)
        returns:
            features: array of shape (n_steps,n_features)
        """
        ################################
        psi = np.exp(-hz * (z[:, None] - cz[None, :]) ** 2)
        return psi / np.sum(psi, axis=1, keepdims=True)
        ################################

    def fit(self, x: NDArray):
        """
        learn the weight vector from LeastSquares,store the weight vectors in self.w (array of shape

        input
            x: trajectory data (n_steps,2)
        """

        # set x0 (starting point) , g(goal point) in self.x0,self.g respectively.
        # assumption is that g(goal point) is the end of the trajectory , and x0 is the start of the trajectory
        # self.x (array of shape (n_steps,2) , self.x0 (array of shape (2,)) , self.g (array of shape (2,))
        ################################
        self.x = x
        self.x0 = x[0]
        self.g = x[-1]
        ################################

        self.T_train = 1.0
        self._dt = self.T_train / (x.shape[0] - 1)

        # get the speed and acceleration using self.derivative method and store it in self.xd,self.xdd
        # self.xd (array of shape (n_steps,2)),self.xdd (array of shape (n_steps,2))
        ################################
        self.xd = self.derivative(x)
        self.xdd = self.derivative(self.xd)
        ################################

        # calulate f_target as mentioned in the theory part and store in "f_target" variable
        # f_target (array of shape (n_steps,2))
        ################################
        f_target = (self.T_train**2) * self.xdd - self.alpha * (
            self.beta * (self.g - x) - self.T_train * self.xd
        )
        ################################

        # set the centers,scaling for basis functions in self.cz,self.hz variables (Note that equal spacing in time domain ,not in phase domain)
        # information about self.hz  (scaling parameter) is given in the theory part
        # self.cz (array of shape (n_features,)),self.hz (array of shape(n_features,))
        ################################
        self.cz = np.linspace(0, self.T_train, self.n_features)
        self.hz = self.n_features / (self.cz + 1e-8)
        ################################

        t = np.linspace(0, self.T_train, x.shape[0])
        z = np.exp(-self.alpha_z * t / self.T_train)
        # store the feature matrix in "feature" variable, use the get_features method.
        # features (array of shape (n_steps,n_features))
        ################################
        features = self.get_features(z, self.cz, self.hz)
        ################################

        # get the weight vectors using leastsquares(can also use np.linalg.pinv) and store in self.w
        # self.w (array of shape (n_features,2))
        ################################
        F: NDArray = f_target / (self.g - self.x0)
        # self.w = np.linalg.pinv(features) @ F
        self.w = np.linalg.lstsq(features * z[:, None], F, rcond=None)[0]
        ################################

    def f_external(self, z):
        """
        once we have the weight vector,get the control function.

        input:
            z: float (phase variable)
        output
            f_ext: array of shape (2,) (for both x1,x2)
        """
        cz = self.cz
        hz = self.hz
        # here cz is the centers of the gaussians, hz are the scaling factors
        # forcing function definition given in the theory part
        ################################
        psi = np.exp(-hz * (z - cz) ** 2)
        psi /= np.sum(psi)
        return (self.g - self.x0) * np.dot(psi, self.w) * z
        ################################

    def ode_differential(self, x, t, f_ext):
        """spring-mass dynamcis
        used for rk4 simulation later ,f_ext is function of z

        input :
            x : array of shape (5,) where 5 dimension where x is x1,x2,x1_dot,x2_dot,z
            t : float >=0.0 (dummy parameter , you will not be using this)
            f_ext : function that takes z as input and returns 2 dimensional array (control forces for x1,x2)
        output: array of shape (5,) where 5 dimension where x is x1_dot,x2_dot,x1_dot_dot,x2_dot_dot,z_dot
        """
        alpha, beta, alpha_z, tau = self.alpha, self.beta, self.alpha_z, self.tau
        # Use the above hyperparameters for dynamical system
        ################################
        x1, x2, x1_dot, x2_dot, z = x
        f = f_ext(z)
        x_ddot = (
            alpha
            * (beta * (self.g - np.array([x1, x2])) - tau * np.array([x1_dot, x2_dot]))
            + f
        )
        z_dot = -alpha_z * z / tau
        return np.array(
            [x1_dot, x2_dot, x_ddot[0] / self.tau**2, x_ddot[1] / self.tau**2, z_dot]
        )
        ################################

    def imitate(self, x0=None, g=None, tau=1.0):
        """
        after learning , we can change the starting position and the ending position for imitation,
        temporal variation is done by tau
        """
        if x0 is not None:
            self.x0 = x0
        if g is not None:
            self.g = g
        self.tau = tau

        # function to use for controller
        f_ext = self.f_external

        # dynamics function
        f_diff = partial(self.ode_differential, f_ext=f_ext)

        # inital point with zero velocity,zeros acceleration
        x_initial = np.array([self.x0[0], self.x0[1], 0.0, 0.0, 1.0])

        # rk4 simulation, till convergence
        x_rk4, t_rk4 = self.rk4_sim(0, x_initial, f_diff)

        # plotting
        plt.plot(x_rk4[:, 0], x_rk4[:, 1], label="dmp")
        plt.plot(self.x[:, 0], self.x[:, 1], label="original")
        plt.legend()
        plt.show()

    def rk4_sim(self, t0, x0, f, dt=1e-3, max_iter=1e5):
        """
        simlution done with rk4

        Returns
        -------
        x : shape (n_steps,5)
        t : shape - (n_steps,)
        """

        # Calculate slopes
        x, t = x0, t0
        x_list, t_list = [x0], [t0]
        i = 0
        while np.linalg.norm(x[:2] - self.g) > 5e-1:
            k1 = dt * f(x, t)
            k2 = dt * f(x + k1 / 2.0, t + dt / 2.0)
            k3 = dt * f(x + k2 / 2.0, t + dt / 2.0)
            k4 = dt * f(x + k3, t + dt)

            # Calculate new x and y
            x = x + 1.0 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t = t + dt
            x_list.append(x)
            t_list.append(t)
            i += 1
            if i >= max_iter:
                print("MAX ITER REACHED : taking too long to converge")
                print(f"simulated for {t} seconds")
                return
        print(f"Took {t} seconds to reach the goal")

        return np.array(x_list), np.array(t_list)


class ProMP(BaseModelABC):
    def __init__(self, n_dims=2, nweights_per_dim=20):

        self.n_dims = n_dims
        self.nweights_per_dim = nweights_per_dim

    def get_features(self, T, overlap=0.7):
        """
        gaussian feature vector for time ,and its derivative

        params:
            T:1d array , query time for features
            overlap: float(0.0-1.0) , heuristic for overlap between gaussians
        returns:
            rbfs.T : array(T x dim),feature vectors
            rbfs_derivative.T : array(T x dim), derivative of feature vectors
        """
        assert T.ndim == 1
        assert np.max(T) <= 1.0 and np.min(T) >= 0.0
        h = -1.0 / (8.0 * self.nweights_per_dim**2 * np.log(overlap))
        centers = np.linspace(0, 1, self.nweights_per_dim)
        rbfs = np.exp(-((T[None, ...] - centers[..., None]) ** 2) / (2.0 * h))
        rbfs_sum_per_step = rbfs.sum(axis=0)
        rbfs_deriv = (centers[..., None] - T[None, ...]) / h
        rbfs_deriv *= rbfs
        rbfs_deriv_sum_per_step = rbfs_deriv.sum(axis=0)
        rbfs_deriv = (
            rbfs_deriv * rbfs_sum_per_step - rbfs * rbfs_deriv_sum_per_step
        ) / (rbfs_sum_per_step**2)
        rbfs /= rbfs_sum_per_step
        return rbfs.T, rbfs_deriv.T

    def fit(self, x, xd):
        """
        fits w for each trajectory and estimates mean,covariance of w
        use the get_features method for feature calulation

        params:
            x:array of shape - (number of trajectories,n_steps,self.n_dims)
        """
        # store the mean of w in self.mean_w -  array of shape(self.n_dim*self.nweights_per_dim,) (Note that this is a 1 dimensional array)
        # store the covaraince of w in self.cov_w - array of shape(self.n_dim*self.nweights_per_dim,self.n_dim*self.nweights_per_dim) (2d array)

        num_trajectories, n_steps, _ = x.shape
        T = np.linspace(0, 1, n_steps)
        Phi, _ = self.get_features(T)  # Get basis features
        Phi = self._nd_block_diagonal(Phi, self.n_dims)  # Expand to block-diagonal
        # Solve for weights: w = (Phi^T Phi)^-1 Phi^T x
        w_all = np.linalg.pinv(Phi.T @ Phi) @ Phi.T @ x.reshape(num_trajectories, -1).T
        self.mean_w = np.mean(w_all, axis=1)
        self.cov_w = np.cov(w_all, rowvar=True)

    def sample_trajectories(self, n_sample, mean_w=None, cov_w=None):
        """
        samples trajectories given mean_w and cov_w , if not given uses the default (self.mean_w,self.cov_w)
        w ~ Normal(mean_w,cov_w)
        x_sample = mean of p( x / w )

        params:
            n_sample : int , number of trajectories to sample
            mean_w : array of shape - (self.nweights_per_dim*self.n_dims,)
            cov_w : array of shape - (self.nweights_per_dim*self.n_dims,self.nweights_per_dim*self.n_dims)
        returns:
            x_sample:array of shape - (n_sample,1000,self.n_dims)

        """
        # use this T for getting features as we trained assuming this T
        T = np.linspace(0, 1, 1000)

        mean_w = self.mean_w if mean_w is None else mean_w
        cov_w = self.cov_w if cov_w is None else cov_w
        sampled_w = np.random.multivariate_normal(mean_w, cov_w, size=n_sample)
        Phi, _ = self.get_features(T)
        Phi = self._nd_block_diagonal(Phi, self.n_dims)
        x_sample = (Phi @ sampled_w.T).T.reshape(n_sample, 1000, self.n_dims)
        return x_sample

    def conditioning_on_xt(self, xt, t, cov=0.0):
        """
        changes the mean,covariance of the w based by conditioning on  x(t)

        params:
            xt : array of shape (self.n_dims,)
            t : 0.0 or 1.0 => 0.0 corresponds to starting point, 1.0 corresponds to the ending point

        returns:
            new_mean : array of shape (self.nweights_per_dim*self.n_dims,)
            new_cov : array of shape (self.nweights_per_dim*self.n_dims,self.nweights_per_dim*self.n_dims)
        """

        assert xt.ndim == 1
        assert t <= 1.0 and t >= 0

        y = np.zeros(2 * len(xt))
        y[::2] = xt

        phi = np.vstack(self.get_features(np.array([t])))
        phi = self._nd_block_diagonal(phi, self.n_dims)
        phi = phi.T
        common_term = (
            self.cov_w
            @ phi
            @ np.linalg.pinv(cov * np.eye(len(phi.T)) + phi.T @ self.cov_w @ phi)
        )
        new_mean_w = self.mean_w + common_term @ (y - phi.T @ self.mean_w)
        new_cov_w = self.cov_w - common_term @ phi.T @ self.cov_w
        return new_mean_w, new_cov_w

    def _nd_block_diagonal(self, partial_1d, n_dims):
        """Replicates matrix n_dims times to form a block-diagonal matrix.

        We also accept matrices of rectangular shape. In this case the result is
        not officially called a block-diagonal matrix anymore.

        Parameters
        ----------
        partial_1d : array, shape (n_block_rows, n_block_cols)
            Matrix that should be replicated.

        n_dims : int
            Number of times that the matrix has to be replicated.

        Returns
        -------
        full_nd : array, shape (n_block_rows * n_dims, n_block_cols * n_dims)
            Block-diagonal matrix with n_dims replications of the initial matrix.
        """
        assert partial_1d.ndim == 2
        n_block_rows, n_block_cols = partial_1d.shape

        full_nd = np.zeros((n_block_rows * n_dims, n_block_cols * n_dims))
        for j in range(n_dims):
            full_nd[
                n_block_rows * j : n_block_rows * (j + 1),
                n_block_cols * j : n_block_cols * (j + 1),
            ] = partial_1d
        return full_nd


# not moving frames
class TPGMM(BaseModelABC):
    def __init__(self, As, Bs, n_mixture):
        """
        params:
            As : array of shape(n_frames,n_trajectories,3,3)
            Bs : array of shape(n_frames,n_trajectories,3)
            n_mixture: number of gaussians in gaussian mixture
        """
        self.As = As
        self.Bs = Bs
        self.n_frames = len(As)
        self.n_feature = As.shape[-1]
        self.n_mixture = n_mixture
        self._reg_factor = 1e-15
        self._reg_cov = np.eye(self.n_feature) * self._reg_factor

    def fit(self, data, max_iter=200, threshold=1e-5):
        """
        step-1 : Transform the data in different frames
        step-2 : initialize the gmm parameters

        Does EM on transformed data and stores the parameters of the model
        input is expected to be aligned with time

        params:
            X: array of shape (n_trajectories,n_steps,3)
        """

        # store the trajectory as viewed from each frame in variable("X")
        # X - shape (n_frames,n_trajectories,n_steps,3)
        # hint:X[0]-(n_trajectories,n_steps,3) is the data viewed w.r.t frame 1, X[1] is the the data viewd w.r.t frame 2
        ################################
        n_traj, n_steps, n_features = data.shape
        X = np.zeros((self.n_frames, n_traj, n_steps, self.n_feature))
        for i in range(self.n_frames):
            X[i] = np.einsum("tij,taj->tai", self.As[i], data) + self.Bs[i][:, None, :]

        # initialize means,covars using  self.time_bases_init() method , store them in self.means,self.covars respectively
        # self.priors as uniform (equal probability)
        ################################
        self.means, self.covars = self.time_based_init(X)
        self.priors = np.full(self.n_mixture, 1 / self.n_mixture)

        # reshape X to (n_frames,n_points,3)
        X = X.reshape(self.n_frames, -1, self.n_feature)

        probabilities = self.gauss_probs(X)
        log_likelihood = self._log_likelihood(probabilities)

        epoch_idx = 0
        while True:
            # Expectation step (use self.expectation_step method)
            ################################
            h = self.expectation_step(probabilities)

            # Maximization step (use self.maximization_step method)
            ################################
            h = h.mean(axis=0)
            self.maximization_step(X, h)

            # update probabilities and log likelihood
            probabilities = self.gauss_probs(X)
            updated_log_likelihood = self._log_likelihood(probabilities)

            # Logging
            difference = updated_log_likelihood - log_likelihood
            if np.isnan(difference):
                raise ValueError("improvement is nan")

            print(
                f"Epoch:{epoch_idx} Log likelihood: {updated_log_likelihood} improvement {difference}"
            )
            epoch_idx += 1

            # break if threshold is reached or max_iter reached
            if difference < threshold or epoch_idx >= max_iter:
                if epoch_idx >= max_iter:
                    print("max_iter reached")
                else:
                    print("threshold satisfied")
                break

            log_likelihood = updated_log_likelihood

    def expectation_step(self, probabilities):
        """
        input:
            probabilities :  (num_frames, n_components, num_points)
        returns:
            h : contribution of each gauusian to a point - array of shape (n_mixture,n_points)
        """
        return self._update_h(probabilities)

    def maximization_step(self, X, h):
        """
        updates the priors, means, covariances
        input:
            h : contribution of each gauusian to a point - array of shape (n_mixture,n_points)
            X (ndarray): shape: (num_frames, num_points, num_features)
        """
        self._update_priors(h)
        self._update_means(X, h)
        self._update_covars(X, h)

    def time_based_init(self, X):
        """
        initializes params by slicing the data in self.n_mixture parts in time

        input:
            X : shape of (n_frames,n_traj,n_steps,n_features)
        returns:
            means: shape -  (n_frames,n_mixture,3)
            covars: shape - (n_frames,n_mixture,3,3)
        """
        split_size = X.shape[2] // self.n_mixture
        X = np.array(
            [
                X[:, :, i * split_size : (i + 1) * split_size, :].reshape(
                    X.shape[0], -1, X.shape[-1]
                )
                for i in range(self.n_mixture)
            ]
        )
        means = np.array([np.mean(x, axis=1) for x in X])
        means = means.transpose((1, 0, 2))

        covars = []
        for x in X:
            covars.append(np.array([np.cov(x_, rowvar=False) for x_ in x]))
        covars = np.array(covars)
        covars = covars.transpose((1, 0, 2, 3))

        return means, covars

    def gauss_probs(self, X):
        """calculate the gaussian probability for a given data set.

        Variable explanation:
        D ... number of features

        Args:
            X (ndarray): data with shape: (num_frames, num_points, num_features)

        Returns:
            ndarray: probability shape (num_frames, n_components, num_points)
        """
        num_frames, num_points, num_features = X.shape
        probs = np.empty((num_frames, self.n_mixture, num_points))

        for frame_idx, component_idx in itertools.product(
            range(num_frames), range(self.n_mixture)
        ):
            frame_data = X[frame_idx]
            cluster_mean = self.means[frame_idx, component_idx]
            cluster_cov = self.covars[frame_idx, component_idx]
            probs[frame_idx, component_idx] = MVN(
                cluster_mean, cluster_cov
            ).to_probability_density(frame_data)
        return probs

    def _update_h(self, probabilities):
        """update h as per equations given in the paper

        Args:
            data (ndarray): shape: (num_frames, num_points, num_features)
            probabilities (ndarray): shape (num_frames, n_components, num_points)
        Returns:
            ndarray: h-parameter. shape: (n_components, num_points(7000))
        """
        ################################
        weighted_probs = probabilities * self.priors[None, :, None]
        h = weighted_probs / np.sum(weighted_probs, axis=1, keepdims=True)
        return h

    def _update_priors(self, h):
        """update priors in self.priors , doesn't return anything

        Args:
            h (ndarray): shape: (n_components, n_points(7000))
        """
        ################################
        self.priors = np.mean(h, axis=1)

    def _update_means(self, X, h):
        """updates the mean parameter (self.mean), doesn't return anything
        Args:
            X (ndarray): shape: (num_frames, num_points, num_features)
            h (ndarray): shape: (n_components, num_points)
        """
        ################################
        h = h[None, :, :]
        weighted_sum = np.einsum("fmp,fpd->fmd", h, X)
        h_sum = np.sum(h, axis=2, keepdims=True)
        self.means = weighted_sum / h_sum

    def _update_covars(self, X, h):
        """updates the covariance parameters
        Args:
            X (ndarray): shape: (num_frames, num_points, num_features)
            h (ndarray): shape: (n_components, num_points)
        """
        num_frames = X.shape[0]
        cov = np.empty_like(self.covars)
        for frame_idx, component_idx in itertools.product(
            range(num_frames), range(self.n_mixture)
        ):
            frame_data = X[frame_idx]
            component_mean = self.means[frame_idx, component_idx]
            component_h = h[component_idx]

            centered = frame_data - component_mean
            # shape: (num_points, num_features, num_features)
            mat_aggregation = np.einsum("ij,ik->ijk", centered, centered)
            # swap dimensions to: (num_features, num_features, num_points)
            mat_aggregation = mat_aggregation.transpose(1, 2, 0)
            # weighted sum and division by h. shape: (num_features, num_features)
            cov[frame_idx, component_idx] = (
                mat_aggregation @ component_h
            ) / component_h.sum()

        # shape: (num_frames, num_components,num_features, num_features)
        self.covars = cov + self._reg_cov

    def _log_likelihood(self, probabilities):
        """calculates the log likelihood of given probabilities

        Args:
            probabilities (ndarray): shape: (num_frames, n_components, num_points)

        Returns:
            float: log likelihood
        """
        probabilities = np.prod(probabilities, axis=0)
        # reshape to: (num_points, n_components)
        probabilities = probabilities.T
        weighted_sum = probabilities @ self.priors  # shape (num_points)
        return np.sum(np.log(weighted_sum)).item()

    def plot_gaussians_wrt_frames(self, Data, As, Bs):
        """
        Plots Projected Gaussians,Product of Gaussians on to Main Frame, and the mean trajectory for the given Parameters

        Params:
            Data: array of shape (n_traj,n_steps,n_features)
            As:array of shape (n_frames,n_features,n_features)
            Bs:array of shape (n_frames,n_features)
        """

        # projected means and covariances
        projected_means = (As[:, None] @ (self.means[..., None])) + (
            Bs[:, None, ..., None]
        )
        projected_means = projected_means[..., 0]
        projected_covars = (
            As[:, None] @ self.covars @ np.transpose(As[:, None], (0, 1, 3, 2))
        )

        # Product of Gaussians
        inv_projected_covars = np.linalg.inv(projected_covars)
        final_covars = np.linalg.inv(np.sum(inv_projected_covars, axis=0))
        final_means = (
            final_covars
            @ np.sum(inv_projected_covars @ (projected_means[..., None]), 0)
        )[..., 0]

        # plotting projected Gaussians
        splot = plt.subplot(111)
        plot_curves3(Data[:, :, 1:], alpha=0.2)
        for i in range(self.n_frames):
            gmm = GMM(
                len(self.priors),
                self.priors,
                projected_means[i][:, 1:],
                projected_covars[i][:, 1:, 1:],
            )
            plot_error_ellipses(splot, gmm, factors=[1])

        plt.title(f"With respect to Main Reference,Projected Gaussians")
        plt.show()

        # plotting product of Gaussians
        splot = plt.subplot(111)
        gmm = GMM(
            len(self.priors), self.priors, final_means[:, 1:], final_covars[:, 1:, 1:]
        )
        plot_curves3(Data[:, :, 1:], alpha=0.1)
        plot_error_ellipses(splot, gmm, factors=[1])

        # mean trajectory with the product of Gaussians
        gmm = GMM(len(self.priors), self.priors, final_means, final_covars)
        time = np.linspace(0, 2, 1000)
        new_traj = gmm.predict([0], time[..., None])
        plot_curves3(new_traj[None])

        plt.title(f"With respect to Main Reference,Product of Gaussians")
        plt.show()

    def plot_gaussians_wrt_frames_ax(self, Data, As, Bs, ax: list[plt.Axes]):
        """
        Plots Projected Gaussians,Product of Gaussians on to Main Frame, and the mean trajectory for the given Parameters

        Params:
            Data: array of shape (n_traj,n_steps,n_features)
            As:array of shape (n_frames,n_features,n_features)
            Bs:array of shape (n_frames,n_features)
        """

        # projected means and covariances
        projected_means = (As[:, None] @ (self.means[..., None])) + (
            Bs[:, None, ..., None]
        )
        projected_means = projected_means[..., 0]
        projected_covars = (
            As[:, None] @ self.covars @ np.transpose(As[:, None], (0, 1, 3, 2))
        )

        # Product of Gaussians
        inv_projected_covars = np.linalg.inv(projected_covars)
        final_covars = np.linalg.inv(np.sum(inv_projected_covars, axis=0))
        final_means = (
            final_covars
            @ np.sum(inv_projected_covars @ (projected_means[..., None]), 0)
        )[..., 0]

        # plotting projected Gaussians
        plot_curves3_ax(ax[0], Data[:, :, 1:], alpha=0.2)
        for i in range(self.n_frames):
            gmm = GMM(
                len(self.priors),
                self.priors,
                projected_means[i][:, 1:],
                projected_covars[i][:, 1:, 1:],
            )
            plot_error_ellipses(ax[0], gmm, factors=[1])

        ax[0].set_title(f"Projected Gaussians")

        # plotting product of Gaussians
        gmm = GMM(
            len(self.priors), self.priors, final_means[:, 1:], final_covars[:, 1:, 1:]
        )
        plot_curves3_ax(ax[1], Data[:, :, 1:], alpha=0.1)
        plot_error_ellipses(ax[1], gmm, factors=[1])

        # mean trajectory with the product of Gaussians
        gmm = GMM(len(self.priors), self.priors, final_means, final_covars)
        time = np.linspace(0, 2, 1000)
        new_traj = gmm.predict([0], time[..., None])
        plot_curves3_ax(ax[1], new_traj[None])

        ax[1].set_title(f"Product of Gaussians")
