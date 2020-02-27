r"""
Dimroth-Watson mixture model
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from watson_distribution import DimrothWatson
from scipy.optimize import minimize
from warnings import warn


__all__ = ('DimrothWatsonMixture')
__author__ = ('Duncan Campbell')


class DimrothWatsonMixture(object):
    """
    class for modelling a distribution as a set of axis-aligned Dimroth-Watson distributions.
    """

    def __init__(self, n_components=2, k=None, w=None):
        """
        Parameters
        ----------
        n_components : int, optional
            number of components in mixture model

        k : array_like, optional
            length n_components array of shape parameters
            default is for np.array([0]*n_components)

        w : array_like, optional
            length n_components array of mixture weights.
            weights must sum to 1.0.
            default is for np.array([1.0/n_components]*n_components)

        """
        
        self.params = None
        self.n_components = int(n_components)

        self.d = DimrothWatson

        # initialize parameters of components
        k = np.atleast_1d(k)
        w = np.atleast_1d(w)
        set_params(k, w)

    
    def set_params(self, k=None, w=None):
        """
        Set the paramaters of the component watson distributions

        Parameters
        ----------
        k : array_like
            length n_components array of shape parameters

        w : array_like
            length n_components array of mixture weights

        Returns
        -------
        params : dict
            dictionary of parameters of the form:
            params[int component] = (w, k)
        """

        # default is to set k=0
        # and equal weights
        k0 = 0
        w0 = 1.0/self.n_components
        if self.params is None:
            # set diuctionary values
            for i in range(0, self.n_components):
                self.params[i] = (w0,k0)
        
        # otherwise set each component
        else:
            if len(k) != self.n_components:
                msg = ('k must be an array of lenght n_components.')
                raise ValueError(msg)

            if len(w) != self.n_components:
                msg = ('w must be an array of lenght n_components.')
                raise ValueError(msg)
            
            if np.sum(w)!=1.0:
                msg = ('sum of mixture weifhts must be equal to 1.')
                raise ValueError(msg)
            
            # set diuctionary values
            for i in range(0, self.n_components):
                self.params[i] = (w[i],k[i])

        return self.params

    def membership_ratio(self, x):
        """

        Parameters
        ----------
        x : array_like
            array of vaslues of cos(theta).

        Returns
        -------
        f : numpy.array
            shape(len(x), n_components) array of ratios
            of membership probabilities
        """

        x = np.atleast_1d(x)
        N = len(x)

        # calculate probability of each x for each componenet
        p = np.zeros((N, self.n_components))
        for i in range(0, self.n_components):
            k = self.params[i][1]
            w = self.params[i][0]
            p[:,i] = w * self.d.pdf(x, k=k)

        # calculatye the ratio of probability in one component
        # relative to all components combined
        f = np.zeros((N, self.n_components))
        for i in range(0, self.n_components):
            f[:,i] = p[:,i]/np.sum(p, axis=-1)

        return f

    def fit(x, ptol=0.01, max_iter=50, verbose=False):
        """
        Fit for the parameters of the mixture model.

        Parameters
        ----------
        x : array_like
            array of cos(theta) values
    
        ptol : float
    
        max_iter : int

        Returns
        -------
        params : dict
            dictionary of parameters of the form:
            params[int component] = (w, k)
        """
        
        continue_loop=True
        p0 = 0.0
        num_iter = 0
        while continue_loop==True:
            r = self.membership_ratio(x)
            p1 = minimize(f, (p0), args=(x, r, ), bounds=[(-0.99,0.99)]).x[0]
            num_iter += 1
            dp = (p1-p0)/p1
            if (dp<ptol) | (num_iter>=max_iter):
                continue_loop=False
            if verbose:
                print(num_iter, p1)
            p0=p1

    return self.params

    def _liklihood(x, f):
        """
        Parameters
        ----------
        x : array_like
            array of cos(theta) values

        x : array_like
            array of membership ratios
        
        Returns
        -------
        lnL : numpy.array
            log-liklihood sample `x` was drawn from the mixture distribution
        """

        # process arguments
        x = np.atleast_1d(x)
        f = np.atleast_1d(f)

        if len(x) != np.shape(f)[0]:
            msg = ('`x` and `f` must be the same shape.')
            raise ValueError(msg)

        # size of sample
        N = len(x)[0]

        # calculate the probabilities each point in the sample
        # was drawn from each individual component
        p = np.zeros((N, self.n_components))
        for i in range(0, self.n_components):
            k = self.params[i][1]
            w = self.params[i][0]
            p[:,i] = w*self.d.pdf(x[:,i], k=k)

        # log-liklihood liklihood
        l = np.zeros(self.n_components)
        for i in range(0, self.n_components):
            l[i] = np.sum(f[:,i]*np.log(p[:,i]))

        return -1.0*np.sum(l)


def fit_watson_mixture_model(x, ptol=0.01, max_iter=50, verbose=False):
    """
    fit for the alignment strength of a symmetric dimroth-watson k-componenent mixture model
    
    Parameters
    ----------
    x : array_like
        A N by k array of cos(theta)
    
    ptol : float
    
    max_iter : int
    """

    def f(p, x, r):
        """
        function to minimize in each step
        """
        k = alignment_strenth(p)
        l = watson_mixture_liklihood(x, k=k, f=r)
        return l

    continue_loop=True
    p0 = 0.0
    num_iter = 0
    while continue_loop==True:
        r = watson_mixture_membership(x, p0)
        p1 = minimize(f, (p0), args=(x, r, ), bounds=[(-0.99,0.99)]).x[0]
        num_iter += 1
        dp = (p1-p0)/p1
        if (dp<ptol) | (num_iter>=max_iter):
            continue_loop=False
        if verbose:
            print(num_iter, p1)
        p0=p1

    return p1


def membership(x, p):
    """
    return the membership ratio for a symmetric dimroth-watson k-componenent mixture model
    
    Parameters
    ----------
    x : array_like
        A N by k array of cos(theta)
    
    p : array_like
        probability
    
    Returns
    -------
    lnL : numpy.array
        log-liklihood sample `x` was drawn from the distribution
    """

    d = DimrothWatson()

    # process arguments
    x = np.atleast_1d(x)

    k = alignment_strenth(p)

    # size of sample
    N = np.shape(x)[0]
    
    # number of distributions
    N_components = np.shape(x)[1]

    p = np.zeros((N, N_components))
    for i in range(0, N_components):
        p[:,i] = d.pdf(x[:,i], k=k)

    f = np.zeros((N, N_components))
    for i in range(0, N_components):
        f[:,i] = p[:,i]/np.sum(p, axis=-1)

    return f


def liklihood(x, f, k):
    """
    Return negative log-liklihood of a symmetric dimroth-watson k-componenent mixture model
    
    Parameters
    ----------
    x : array_like
        A N by k array of cos(theta)
    
    f: array_like
        membership
    
    k : float
        shape parameter of the distribution
    
    Returns
    -------
    lnL : numpy.array
        log-liklihood sample `x` was drawn from the distribution
    """

    # initialize distribution
    d = DimrothWatson()

    # process arguments
    x = np.atleast_1d(x)
    f = np.atleast_1d(f)
    if np.shape(x) != np.shape(f):
        msg = ('`x` and `f` must be the same shape.')
        raise ValueError(msg)

    # size of sample
    N = np.shape(x)[0]
    
    # number of distributions
    N_components = np.shape(x)[1]

    # calculate the probabilities each point in the sample
    # was drawn from each individual component
    p = np.zeros((N, N_components))
    for i in range(0, N_components):
        p[:,i] = d.pdf(x[:,i], k=k)

    # log-liklihood liklihood
    l = np.zeros((N_components,))
    for i in range(0, N_components):
        l[i] = np.sum(f[:,i]*np.log(p[:,i]))

    return -1.0*np.sum(l)


