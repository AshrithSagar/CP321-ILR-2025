"""
models.py
Utility for models
"""

from abc import ABC, abstractmethod
from functools import partial

import gmr
import matplotlib.pyplot as plt
import numpy as np
import torch
from gmr import GMM
from numpy.typing import NDArray
from scipy import optimize
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

    def predict(self, X):
        """For backwards compatibility with BaseModelABC"""
        raise NotImplementedError

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
