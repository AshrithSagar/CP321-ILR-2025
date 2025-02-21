#SEDs algorithm for Likelihood minimization
from gmr import GMM
import torch
from scipy import optimize
import numpy as np
from functools import partial

class SEDS():
    
    def __init__(self,attractor,n_mixture):
        
        assert attractor.ndim==1
        self.attractor = attractor
        self.n_mixture = n_mixture
        self.d = len(attractor)

    def initialize_params(self,y,yd):
        '''
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
            
        '''
        pass


    def cov2A(self,cov):
        '''
        helper function for converting the full covariance matrices to Ak matrices

        Parameters
        ----------
        cov : covariance matrices , shape -(self.n_mixture,2*self.d,2*self.d)

        Returns
        -------
        As : A matrices , shape - (self.n_mixture,self.d,self.d)
        '''
        As = []
        for k in range(self.n_mixture):
            As.append(cov[k][self.d:,:self.d]@np.linalg.inv(cov[k][:self.d,:self.d]))
        return np.stack(As)

    def _to_optim_vector(self,priors, means, covars):
        '''
        converting priors,means,covars to a single vector for optimization purpose, with respective transformations

        Parameters
        ----------
        priors : shape-(self.n_mixture,)
        means : shape-(self.n_mixture,2*self.d)
        covars : shape-(self.n_mixture,2*self.d,2*self.d)

        Returns
        -------
        x :  vector shape - (self.n_mixture+self.n_mixture*self.d+self.n_mixture*2*self.d*2*self.d,)
        
        '''
        chol_decomps = np.array([np.linalg.cholesky(covar) for covar in covars])
        return np.hstack((np.log(priors), means[:,:self.d].flatten(),chol_decomps.flatten()))
        
    def _from_optim_vector(self,x):
        '''
        given the one dimension vector , gets back the priors,means,covariance

        Parameters
        ----------
        x : vector shape - (self.n_mixture+self.n_mixture*self.d+self.n_mixture*2*self.d*2*self.d,)

        Returns
        --------
        priors : shape-(self.n_mixture,)
        means : shape-(self.n_mixture,2*self.d)
        covars : shape-(self.n_mixture,2*self.d,2*self.d)
        
        '''

        #priors
        priors = np.exp(x[:self.n_mixture])
        priors /= priors.sum()

        #covars
        Ls = x[-self.n_mixture*2*self.d*2*self.d:].reshape(self.n_mixture,2*self.d,2*self.d)
        covars = Ls@np.transpose(Ls,(0,2,1))

        #means
        As = self.cov2A(covars)
        mu1 = x[self.n_mixture:-self.n_mixture*2*self.d*2*self.d].reshape(-1,self.d)
        mu2 = As@((mu1-self.attractor)[...,None])
        mu2 = mu2[:,:,0]
        means = np.concatenate([mu1,mu2],axis=-1)
        
        return priors,means,covars
        
    def _torch_from_optim_vector(self,x0):
        '''
        same as '_from_optim_vector' method but with input as pytorch tensor instead of numpy
        so that, computational graph form for the usage of autograd package
        '''

        #prior
        priors = torch.exp(x0[:self.n_mixture])
        priors = priors/priors.sum()

        #covars
        Ls = x0[-self.n_mixture*2*self.d*2*self.d:].reshape(self.n_mixture,2*self.d,2*self.d)
        covars = Ls@torch.transpose(Ls,1,2)

        #means
        As = []
        for k in range(self.n_mixture):
            As.append(covars[k][self.d:,:self.d]@torch.linalg.inv(covars[k][:self.d,:self.d]))
        As = torch.stack(As)
        mu1 = x0[self.n_mixture:-self.n_mixture*2*self.d*2*self.d].reshape(-1,self.d)
        mu2 = As@((mu1-torch.from_numpy(self.attractor))[...,None])
        mu2 = mu2[:,:,0]
        means = torch.concatenate([mu1,mu2],axis=-1)
        
        return priors,means,covars

    def objective(self,x,y,yd):
        '''
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
        '''
        x = torch.from_numpy(x)
        Y = np.concatenate([y,yd],axis=-1)
        Y = torch.from_numpy(Y)

        #making x as parameters
        x.requires_grad_()
        priors,means,covars = self._torch_from_optim_vector(x)

        #objective calculation
        xm = (Y[:,None,:]-means[None])[...,None]
        q = -((xm.transpose(-1,-2)@torch.linalg.inv(covars))@xm)[:,:,0,0]/2
        q = q-(torch.log(torch.linalg.det(covars))[None]/2)
        q = q+torch.log(priors)
        f = -torch.mean(torch.logsumexp(q,dim=1))

        #calculates gradient wrt x
        f.backward()
        
        return f.detach().numpy(),x.grad.detach().numpy()

    def constraint_func(self,x,requires_grad=False):
        '''
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
        
        '''

        #making x , as leaf of the conmputational graph
        x = torch.from_numpy(x)
        if requires_grad:x.requires_grad_()

        #covars
        Ls = x[-self.n_mixture*2*self.d*2*self.d:].reshape(self.n_mixture,2*self.d,2*self.d)
        covars = Ls@torch.transpose(Ls,1,2)

        #A matrices
        As = []
        for k in range(self.n_mixture):
            As.append(covars[k][self.d:,:self.d]@torch.linalg.inv(covars[k][:self.d,:self.d]))
        As = torch.stack(As)

        #B matrices
        Bs = As + torch.transpose(As,1,2)

        #constraints
        Cs = []
        for i in range(self.d):
            Cs.append(torch.linalg.det(Bs[:,:i+1,:i+1])*((-1)**(i+1)))
        Cs = torch.concat(Cs)

        # if only constraint values are required
        if not requires_grad:
            return Cs.detach().numpy()

        #if only gradient values are required
        gCs = []
        for i in range(len(Cs)):
            gCs.append(torch.autograd.grad(Cs[i],x,torch.ones(Cs[i].shape),retain_graph=True)[0])
        gCs = torch.stack(gCs)
        
        return gCs.detach().numpy()

    def predict(self,X):
        '''
        returns prediction from the model X_dot
    
        params:
            X: array of shape (n_points,2)
        returns:
            predicted X_dot: array of shape (n_points,2)
        '''
        
        return self.gmm.predict(range(self.d),X)

    def ode_differential(self,x,t,gmm):
        '''
        function used for rk4 simulation
        '''
        return gmm.predict(range(self.d),x[None])[0]
        
        
    def rk4_sim(self,t0,t_end,x0,f,dt=1e-3):
        '''
        simlution done with rk4

        Returns
        -------
        x : shape (n_steps,self.d)
        t : shape - (n_steps,)
        '''
        # Calculate slopes
        x,t = x0,t0
        x_list,t_list = [x0],[t0]
        while t<=t_end:
            k1 = dt*f(x,t)
            k2 = dt*f(x+k1/2.,t+dt/2. )
            k3 = dt*f(x+k2/2.,t+dt/2. )
            k4 = dt*f(x+k3 , t+dt )
            
            # Calculate new x and y
            x = x + 1./6*(k1+2*k2+2*k3+k4)
            t = t + dt
            x_list.append(x)
            t_list.append(t)
        
        return np.array(x_list),np.array(t_list)

    def imitate(self,x0,t_end=1):
        '''
        imitation with new starting point to reach the attractor

        Parameters
        ----------
        x0  - starting point - shape (self.d,)

        Returns
        -------
        x_rk4 - simulated data with trained GMM - shape (n_steps,self.d)
        t_rk4 - time - shape (n_steps,)
        '''
        f = partial(self.ode_differential,gmm=self.gmm)
        return self.rk4_sim(0,t_end,x0,f)

    def fit(self,x,xd):
        '''
        initialize the params,"SLSQP" for optimization, as given in the paper

        Parametes
        ---------
        x: state space ,shape - (number of points,self.d)
        xd: derivative of state space - (number of points,self.d)

        Return:
        -------
        No Returns (Stores the GMM if optimization is successfull)
        
        '''

        #initialization of parameters
        priors,means,covars = self.initialize_params(x,xd)

        #converting to single vector
        x0 = self._to_optim_vector(priors,means,covars)

        #inequlity constraints and their derivatives
        ineq_const = {'type':'ineq',
                         'fun':lambda a:self.constraint_func(a,requires_grad=False),
                         'jac':lambda a:self.constraint_func(a,requires_grad=True)}

        #optimize
        res = optimize.minimize(partial(self.objective,y=x,yd=xd),x0,method="SLSQP",jac=True,constraints=[ineq_const],options={'maxiter':1000})
        
        #verbose
        print("initial_object_value:",self.objective(x0,x,xd)[0])
        print("final objective_value:",res.fun)

        priors,means,covars = self._from_optim_vector(res.x)
        self.gmm = GMM(self.n_mixture,priors=priors,means=means,covariances=covars)
        