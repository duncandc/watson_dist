r"""
Dimroth-Watson mixture model
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from watson_dist import DimrothWatson
from scipy.optimize import minimize



__all__ = ('DimrothWatsonMixture')
__author__ = ('Duncan Campbell')


class DimrothWatsonMixture(object):
    r"""
    class for modelling a distribution as a mixture of axis-aligned Dimroth-Watson distributions.
    """

    def __init__(self, n_components=2, k=None, w=None):
        r"""
        Parameters
        ----------
        n_components : int, optional
            number of watson distribution components in the model

        k : array_like, optional
            length n_components array of shape parameters
            default is for np.array([0]*n_components)

        w : array_like, optional
            length n_components array of mixture weights.
            weights must sum to 1.0.
            default is for np.array([1.0/n_components]*n_components)

        """

        self.n_components = int(n_components)

        self.d = DimrothWatson()

        # initialize parameters of components
        self.params = []
        k = np.atleast_1d(k)
        w = np.atleast_1d(w)
        self.set_params(k, w)

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
        if self.params == []:
            for i in range(0, self.n_components):
                self.params.append([w0,k0])

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

            # set dictionary values
            for i in range(0, self.n_components):
                self.params[i] = [w[i],k[i]]

        return self.params

    def membership_ratio(self, x):
        """
        component membership probability

        Parameters
        ----------
        x : array_like
            array of vaslues of cos(theta).

        Returns
        -------
        r : numpy.array
            shape(len(x), n_components) array membership probabilities
        """

        x = np.atleast_1d(x)
        N = len(x)

        # liklihood for each x for each component
        p = np.zeros((N, self.n_components))
        for i in range(0, self.n_components):
            w = self.params[i][0]
            k = self.params[i][1]

            p[:,i] = w * self.d.pdf(x, k=k)

        # membership probability
        r = np.zeros((N, self.n_components))
        for i in range(0, self.n_components):
            r[:,i] = p[:,i]/np.sum(p, axis=-1)

        return r

    def fit(self, x, ptol=0.01, max_iter=50, verbose=False):
        """
        Fit mixture model

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

        r = self.membership_ratio(x)  # partial membership
        for pp in range(10):

            # update distribution parameters
            for i in range(self.n_components):
                result = minimize(self._log_liklihood, (self.params[i][1]), args=(x, r[:,i], ), bounds=[(-100,100)])
                self.params[i][1] = result.x[0]

            # update mixing coefficients
            r = self.membership_ratio(x)
            for i in range(self.n_components):
                self.params[i][0] = np.mean(r[:,i])

        return self.params

    def _log_liklihood(self, p, x, w):
        """
        log liklihood of single component

        Parameters
        ----------
        x : array_like

        w : array_like

        Returns
        -------
        lnL : numpy.array
            negative log-liklihood sample `x` was drawn from the mixture distribution
        """

        # process arguments
        x = np.atleast_1d(x)
        w = np.atleast_1d(w)

        L = self.d.pdf(x, p)
        l = np.sum(w*np.log(L))

        return -1.0*l


