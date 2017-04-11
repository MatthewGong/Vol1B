import math
import numpy as np

from numpy.random import normal
from matplotlib import pyplot as plt
from scipy import linalg as la
from sympy import subfactorial
from astropy.table import Table, Column


##############  Problem 1  ##############
def prob1():
    '''
    Randomly perturb w_coeff by replacing each coefficient a_i with a_i*r_i, where
    r_i is drawn from a normal distribution centered at 1 with varience 1e-10.
    	
    Plot the roots of 100 such experiments in a single graphic, along with the roots
    of the unperturbed polynomial w(x)
    	
    Using the final experiment only, estimate the relative and absolute condition number
    (in any norm you prefer).
    	
    RETURN:
    Should display graph of all 100 perturbations.
    Should print values of relative and absolute condition.
    '''
    w_roots = np.arange(1, 21)
    w_coeffs = np.array([1, -210, 20615, -1256850, 53327946, -1672280820,
				40171771630, -756111184500, 11310276995381,
    				-135585182899530, 1307535010540395,
    				-10142299865511450, 63030812099294896,
    				-311333643161390640, 1206647803780373360,
    				-3599979517947607200, 8037811822645051776,
    				-12870931245150988800, 13803759753640704000,
    				-8752948036761600000, 2432902008176640000])
        
    # plot the unperturbed points
    plt.scatter(w_roots, np.zeros(20))

    # randomly perturb the coeffs 100 times
    for i in xrange(0,100):

        # draw random numbers from a normal distribution
        rand = normal(1,1e-10,21)

        # perturb the coeffecients
        perturbed_coeff = w_coeffs * rand

        # find the perturbed roots
        perturbed_roots = np.roots(np.poly1d(perturbed_coeff))

        # plot the real and imaginary portions of the roots
        real = np.real(perturbed_roots)
        imag = np.imag(perturbed_roots)

        plt.scatter(real,imag,s = .5)

    plt.show()

    # use the infinity norm to find the condition number
        # This will only see the condition of the last one since the last iteration is preserved to this point
    condition_abs = la.norm(perturbed_roots - w_roots, np.inf)/la.norm(rand, np.inf)
    condition_rel = condition_abs * la.norm(w_coeffs, np.inf)/la.norm(w_roots,np.inf)

    print condition_abs
    print condition_rel

##############  Problem 2  ##############	
def eig_condit(M):
    '''
    Approximate the condition number of the eigenvalue problem at M.
    
    INPUT:
    M - A 2-D square NumPy array, representing a square matrix.
    
    RETURN:
    A tuple containing approximations to the absolute and 
    relative condition numbers of the eigenvalue problem at M.
    '''

    # take the first eigan value of some matrix
    eig_vals = la.eig(M)[0]

    # make a perturbation matrix the same shape as M with real and imaginary values
    perturbation = normal(0, 1e-10, M.shape) + normal(0, 1e-10, M.shape) *1j

    # find the first eigen value of the pertrubed matrix
    perturbed_eig_vals = la.eig(M+perturbation)[0]

    condition_abs = la.norm(eig_vals-perturbed_eig_vals) / la.norm(perturbation)
    condition_rel = condition_abs * la.norm(M) / la.norm(eig_vals)

    print "The order of magnitude for a  2x2 symmetric matrix is 1.0 ^ -1"

    return condition_abs, condition_rel

#   1 pt extra credit
def plot_eig_condit(x0=-100, x1=100, y0=-100, y1=100, res=10):
    '''
    Create a grid of points. For each pair (x,y) in the grid, find the 
    relative condition number of the eigenvalue problem, using the matrix 
    [[1 x]
     [y 1]]
    as your input. You can use plt.pcolormesh to plot the condition number
    over the entire grid.
    
    INPUT:
    x0 - min x-value of the grid
    x1 - max x-value
    y0 - min y-value
    y1 - max y-value
    res - number of points along each edge of the grid
    '''
    raise NotImplementedError('plot_eig_condit not implemented')

##############  Problem 3  ##############
def integral(n):
    '''
    RETURN I(n)
    '''
    
    return ((-1)**n * subfactorial(n) + (-1)**(n+1) * math.factorial(n)/np.exp(1))

def prob3():
    '''
    For the values of n in the problem, compute integral(n). Compare
    the values to the actual values, and print your explanation of what
    is happening.
    '''
    
    #actual values of the integral at specified n
    actual_values = [0.367879441171, 0.145532940573, 0.0838770701034, 
                 0.0590175408793, 0.0455448840758, 0.0370862144237, 
                 0.0312796739322, 0.0270462894091, 0.023822728669, 
                 0.0212860390856, 0.0192377544343] 

    # these N's are given
    N = [1] +  range(5,55,5)

    # calculate the integral given some n
    vals = [integral(x) for x in N]

    # create a nice output of the values calculated 
    string = "\n" + "n\t" + " Actual value of I(n)\t" + "Approximated value of I(n) "

    for i in xrange(11):
        string += '\n' + str(N[i]) + '\t' + str(actual_values[i]) + '\t\t' + str(vals[i])

    print string

    print " due to catastrophic cancellation some of the estimates become wrong"