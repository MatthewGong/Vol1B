import sys
import numpy as np

from scipy import linalg as la
from matplotlib import pyplot as plt

def gmres(A, b, x_0, k=100, tol=1e-8):
    '''Calculate approximate solution of Ax=b using GMRES algorithm.
        
    INPUTS:
    A    - Callable function that calculates Ax for any input vector x.
    b    - A NumPy array of length m.
    x0   - An arbitrary initial guess.
    k    - Maximum number of iterations of the GMRES algorithm. Defaults to 100.
    tol  - Stop iterating if the residual is less than 'tol'. Defaults to 1e-8.
    
    RETURN:
    Return (y, res) where 'y' is an approximate solution to Ax=b and 'res'
    is the residual.
    
    Examples:
    >>> a = np.array([[1,0,0],[0,2,0],[0,0,3]])
    >>> A = lambda x: a.dot(x)
    >>> b = np.array([1, 4, 6])
    >>> x0 = np.zeros(b.size)
    >>> gmres(A, b, x0)
    (array([ 1.,  2.,  2.]), 1.09808907533e-16)
    '''

    Q = np.empty((len(b),k+1))
    H = np.zeros((k+1, k))

    r_0 = b - A(x_0)
    Q[:,0] = r_0 / np.linalg.norm(r_0)

    beta = la.norm(r_0)

    e_1    = np.zeros(k+1)
    e_1[0] = beta

    for j in xrange(k):
        Q[:, j+1] = A(Q[:, j])

        for i in xrange(j+1):
            H[i, j]   = np.dot(Q[:, i], Q[:, j+1])
            Q[:, j+1] = Q[:,j +1] - H[i, j]*Q[:, i]

        H[j+1, j] = np.linalg.norm(Q[:, j+1])
        Q[:, j+1] = Q[:, j+1] / H[j+1, j]

        y, res, rank, s = np.linalg.lstsq(H[:j+2,:j+1], (e_1[:j+2]))

     
        if res < tol:
            return Q[:, :j+1].dot(y) + x_0, res

    return Q[:, :k].dot(y) + x_0, res


def plot_gmres(A, b, x_0, tol=1e-8):
    '''Use the GMRES algorithm to approximate the solution to Ax=b. 
        Plot the eigenvalues of A and the convergence of the algorithm.
    
    INPUTS:
    A   - A 2-D NumPy array of shape mxm.
    b   - A 1-D NumPy array of length m.
    x0  - An arbitrary initial guess.
    tol - Stop iterating and create the desired plots when the residual is
    less than 'tol'. Defaults to 1e-8.
    
    OUTPUT:
    Follow the GMRES algorithm until the residual is less than tol, for a
    maximum of m iterations. Then create the two following plots (subplots
    of a single figure):
    
    1. Plot the eigenvalues of A in the complex plane.
    
    2. Plot the convergence of the GMRES algorithm by plotting the
    iteration number on the x-axis and the residual on the y-axis.
    Use a log scale on the y-axis.
    '''

    residuals = []

    eig_vals, eig_vecs = la.eig(A)
    k = len(b)

    Q = np.empty((len(b),k+1))
    H = np.zeros((k+1, k))

    r_0 = b - A.dot(x_0)
    Q[:,0] = r_0 / np.linalg.norm(r_0)

    beta = la.norm(r_0)

    e_1    = np.zeros(k+1)
    e_1[0] = beta

    #
    res = tol + 1
    j = 0

    while j < k and res > tol:
        Q[:, j+1] = A.dot(Q[:, j])

        for i in xrange(j+1):
            H[i, j]   = np.dot(Q[:, i], Q[:, j+1])
            Q[:, j+1] = Q[:,j +1] - H[i, j]*Q[:, i]

        H[j+1, j] = np.linalg.norm(Q[:, j+1])
        Q[:, j+1] = Q[:, j+1] / H[j+1, j]

        y, res, rank, s = np.linalg.lstsq(H[:j+2,:j+1], (e_1[:j+2]))
        
        #set up next iteration
        residuals.append(res)
        j += 1

    # make the plots
    plt.subplot(1,2,1) 
    plt.scatter(eig_vals.real, eig_vals.imag)

    plt.subplot(1,2,2)
    plt.yscale('log')

    x_vec = np.linspace(0, len(residuals), len(residuals))
    plt.plot(x_vec, residuals)

    plt.show()

def problem2():
    '''Create the function for problem 2 which calls plot_gmres on An for n = -4,-2,0,2,4.
        Print out an explanation of how the convergence of the GMRES algorithm
        relates to the eigenvalues.
        
    '''
    m = 200

    b = np.ones(m)

    x_0 = np.zeros(m)

    n_vec = [-4,-2,0,2,4]

    I_mat = np.eye(m)
    P_mat = np.random.normal(0, 1./(2*np.sqrt(m)), (m,m))

    for n in n_vec:
        A = n*I_mat + P_mat
        plot_gmres(A, b, x_0)


def gmres_k(A, b, x_0, k=5, tol=1E-8, restarts=50):
    '''Use the GMRES(k) algorithm to approximate the solution to Ax=b.
        
    INPUTS:
    A        - A callable function that calculates Ax for any vector x.
    b        - A NumPy array.
    x0       - An arbitrary initial guess.
    k        - Maximum number of iterations of the GMRES algorithm before
    restarting. Defaults to 5.
    tol      - Stop iterating if the residual is less than 'tol'. Defaults
    to 1E-8.
    restarts - Maximum number of restarts. Defaults to 50.
    
    RETURN:
    Return (y, res) where 'y' is an approximate solution to Ax=b and 'res'
    is the residual.
    '''

    Q = np.empty((len(b),k+1))
    H = np.zeros((k+1, k))

    r_0 = b - A(x_0)
    Q[:,0] = r_0 / np.linalg.norm(r_0)

    beta = la.norm(r_0)

    e_1    = np.zeros(k+1)
    e_1[0] = beta

    #
    n = 0

    while n <= restarts:
        for j in xrange(k):
            Q[:, j+1] = A(Q[:, j])

            for i in xrange(j+1):
                H[i, j]   = Q[:, i].dot(Q[:, j+1])
                Q[:, j+1] = Q[:,j +1] - H[i, j]*Q[:, i]

            H[j+1, j] = np.linalg.norm(Q[:, j+1])
            Q[:, j+1] = Q[:, j+1] / H[j+1, j]

            y, res, rank, s = np.linalg.lstsq(H[:j+2,:j+1], (e_1[:j+2]))

            if res < tol:
                return Q[:, :j+1].dot(y) + x_0, res

        # prepare next iteration
        x_0 = y
        n  += 1

    return y, res



def test(): 
    a = np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    #a = np.random.rand(3,3)
    A = lambda x: a.dot(x)
    b = np.array([1., 4., 6.])
    x_0 = np.zeros(b.size)
    print gmres(A, b, x_0)
    #problem2()
    print gmres_k(A, b, x_0)

test()
problem2()