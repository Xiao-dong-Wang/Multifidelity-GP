import autograd.numpy as np
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b, minimize
import traceback
import sys
from autograd import value_and_grad
from util import chol_inv

# A conventional gaussian process class for bayesian optimization
class GP:
    # Initialize GP class
    # train_x shape: (dim, num_train);   train_y shape: (num_train, ) 
    def __init__(self, dataset, bfgs_iter=100, debug=True, k=0):
        self.k = k
        self.train_x = dataset['train_x']
        self.train_y = dataset['train_y']
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.dim = self.train_x.shape[0]
        self.num_train = self.train_x.shape[1]
        self.normalize()
        self.idx1 = [self.dim-1]
        self.idx2 = np.arange(self.dim-1)
        self.jitter = 1e-8

    def normalize(self):
        self.train_y = self.train_y.reshape(-1)
        self.mean = self.train_y.mean()
        self.std = self.train_y.std() + 0.000001
        self.train_y = (self.train_y - self.mean)/self.std

    # Initialize hyper_parameters
    def get_default_theta(self):
        if self.k: # kernel2 MF
            # sn2 + (output_scale + lengthscale) + (output_scale + lengthscales) * 2
            theta = np.random.randn(3 + 2*self.dim)
            theta[2] = np.maximum(-100, np.log(0.5*(self.train_x[self.dim-1].max() - self.train_x[self.dim-1].min())))
            for i in range(self.dim-1):
                tmp = np.maximum(-100, np.log(0.5*(self.train_x[i].max() - self.train_x[i].min())))
                theta[4+i] = tmp
                theta[4+self.dim+i] = tmp
        else: # kernel1 RBF
            # sn2 + output_scale + lengthscales
            theta = np.random.randn(2 + self.dim)
            for i in range(self.dim):
                theta[2+i] = np.maximum(-100, np.log(0.5*(self.train_x[i].max() - self.train_x[i].min())))
        theta[0] = np.log(np.std(self.train_y) + 0.000001) # sn2
        return theta
    
    # RBF kernel
    def kernel1(self, x, xp, hyp):
        output_scale = np.exp(hyp[0])
        lengthscales = np.exp(hyp[1:]) + 0.000001
        diffs = np.expand_dims((x.T/lengthscales).T, 2) - np.expand_dims((xp.T/lengthscales).T, 1)
        return output_scale * np.exp(-0.5*np.sum(diffs**2, axis=0))
    
    # MF kernel
    def kernel2(self, x, xp, hyp):
        hyp_f = hyp[:2]
        hyp_rho = hyp[2:2+self.dim]
        hyp_delta = hyp[2+self.dim:]
        return self.kernel1(x[self.idx1], xp[self.idx1], hyp_f) * self.kernel1(x[self.idx2], xp[self.idx2], hyp_rho) + self.kernel1(x[self.idx2], xp[self.idx2], hyp_delta)

    def kernel(self, x, xp, hyp):
        if self.k: 
            return self.kernel2(x, xp, hyp)
        else:
            return self.kernel1(x, xp, hyp)

    def neg_log_likelihood(self, theta):
        sn2 = np.exp(theta[0])
        hyp = theta[1:]
         
        K = self.kernel(self.train_x, self.train_x, hyp) + sn2 * np.eye(self.num_train)
        L = np.linalg.cholesky(K)

        logDetK = np.sum(np.log(np.diag(L)))
        alpha = chol_inv(L, self.train_y.T)
        nlz = 0.5*(np.dot(self.train_y, alpha) + self.num_train*np.log(2*np.pi)) + logDetK
        if(np.isnan(nlz)):
            nlz = np.inf

        self.nlz = nlz
        return nlz

    # Minimize the negative log-likelihood
    def train(self):
        theta0 = self.get_default_theta()
        self.loss = np.inf
        self.theta = np.copy(theta0)

        nlz = self.neg_log_likelihood(theta0)

        def loss(theta):
            nlz = self.neg_log_likelihood(theta)
            return nlz

        def callback(theta):
            if self.nlz < self.loss:
                self.loss = self.nlz
                self.theta = np.copy(theta)

        gloss = value_and_grad(loss)
        try:
            fmin_l_bfgs_b(gloss, theta0, maxiter=self.bfgs_iter, m=100, iprint=self.debug, callback=callback)
        except np.linalg.LinAlgError:
            print('GP. Increase noise term and re-optimization.')
            theta0 = np.copy(self.theta)
            theta0[0] += np.log(10)
            try:
                fmin_l_bfgs_b(gloss, theta0, maxiter=self.bfgs_iter, m=10, iprint=self.debug, callback=callback)
            except:
                print('GP. Exception caught, L-BFGS early stopping...')
                if self.debug:
                    print(traceback.format_exc())
        except:
            print('GP. Exception caught, L-BFGS early stopping...')
            if self.debug:
                print(traceback.format_exc())
        
        sn2 = np.exp(self.theta[0])
        hyp = self.theta[1:]
        K = self.kernel(self.train_x, self.train_x, hyp) + sn2 * np.eye(self.num_train) + self.jitter * np.eye(self.num_train)
        self.L = np.linalg.cholesky(K)
        self.alpha = chol_inv(self.L, self.train_y.T)
        if self.k:
            self.for_diag = np.exp(self.theta[1]) * np.exp(self.theta[3]) + np.exp(self.theta[3+self.dim])
        else:
            self.for_diag = np.exp(self.theta[1])
        print('GP. Finished training process.')

    def predict(self, test_x, is_diag=1):
        sn2 = np.exp(self.theta[0])
        hyp = self.theta[1:]
        K_star = self.kernel(test_x, self.train_x, hyp)
        py = np.dot(K_star, self.alpha)
        KvKs = chol_inv(self.L, K_star.T)
        if is_diag:
            ps2 = self.for_diag + sn2 - (K_star * KvKs.T).sum(axis=1)
        else:
            ps2 = sn2 - np.dot(K_star, KvKs) + self.kernel(test_x, test_x, hyp)
        ps2 = np.abs(ps2)
        py = py * self.std + self.mean
        py = py.reshape(-1)
        ps2 = ps2 * (self.std**2)
        return py, ps2

