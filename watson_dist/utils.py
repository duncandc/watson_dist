r"""
utilities for the watson distribution package
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


__all__ = ('erfiinv')
__author__ = ('Duncan Campbell')


def erfiinv(y, kmax=100):
    r"""
    aproximation of the inverse imaginary error function, :math:`{\rm erfi}^{-1}(y)`,
    for :math:`y` close to zero.
    
    Parameters
    ----------
    y : array_like
        array of floats
    
    Returns
    -------
    x : numpy.array
        aproximate value of inverse imaginary error function
    """

    c = np.zeros(kmax)
    c[0] = 1.0
    c[1] = 1.0
    result = 0.0
    for k in range(0, kmax):
        # Calculate C sub k
        if k > 1:
            c[k] = 0.0
            for m in range(0, k):
                term = (c[m]*c[k - 1 - m])/((m + 1.0)*(2.0*m + 1.0))
                c[k] += term
        result += ((-1.0)**k*c[k]/(2.0*k + 1))*((np.sqrt(np.pi)/2)*y)**(2.0*k + 1)
    return result