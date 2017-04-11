import numpy as np

from numpy import poly1d
from sympy import mpmath as mp
from matplotlib import pyplot as plt
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D #needed for 3d plotting

def singular_surface_plot(f, x_bounds=(-1.,1), y_bounds=(-1.,1.), res=500, threshold=2., lip=.1):
    """ Plots the absolute value of a function as a surface plot """
        
    x_vec = np.linspace(x_bounds[0], x_bounds[1], res)
    y_vec = np.linspace(y_bounds[0], y_bounds[1], res)

    X,Y = np.meshgrid(x_vec,y_vec)

    Z = 1 / ( X + 1j*Y)
    Z = np.abs(Z)

    # Set the values between threshold and
    # threshold + lip to be equal to threshold.
    # This forms a somewhat more concrete
    # edge at the top of the plot.
    Z[(threshold+lip>Z)&(Z>threshold)]  =  threshold
    Z[(threshold-lip<Z)&(Z<-threshold)] = -threshold

    Z[np.absolute(Z) >= threshold + lip] = np.nan

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X,Y,Z, cmap = 'coolwarm')
    plt.show()


def partial_fractions(p, q):
    """ Finds the partial fraction representation of the rational
        function 'p' / 'q' where 'q' is assumed to not have any repeated
        roots. 'p' and 'q' are both assumed to be numpy poly1d objects.
        Returns two arrays. One containing the coefficients for
        each term in the partial fraction expansion, and another containing
        the corresponding roots of the denominators of each term. """
    
    coeffs = []

    roots_q = q.roots
    deriv_q = q.deriv()

    for root in roots_q:
        coeffs.append(p(root)/deriv_q(root))

    return np.array(coeffs), roots_q

def cpv(p, q, tol = 1E-8):
    """ Evaluates the cauchy principal value of the integral over the
        real numbers of 'p' / 'q'. 'p' and 'q' are both assumed to be numpy
        poly1d objects. 'q' is expected to have a degree that is
        at least two higher than the degree of 'p'. Roots of 'q' with
        imaginary part of magnitude less than 'tol' are treated as if they
        had an imaginary part of 0. """

    coeffs, roots_q = partial_fractions(p,q)
    upper_mask = np.argwhere(roots_q.imag > tol)

    A = np.sum(coeffs[upper_mask])

    return 2* np.pi * 1.0j * A

def count_roots(p):
    """ Counts the number of roots of the polynomial object 'p' on the
        interior of the unit ball using an integral. """
    raise NotImplementedError()
