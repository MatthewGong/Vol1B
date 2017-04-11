import numpy as np


from numpy.linalg import inv
from scipy import linalg as la
from matplotlib import pyplot as plt

# Problem 1
def ps_scatter_plot(A, epsilon=.001, num_pts=20):
    '''Plots the 'poorman's pseudospectrum' of a matrix A
    Parameters:
    A : ndarray of size (n,n)
    The matrix whose pseudospectrum is to be plotted.
    epsilon : float
    The norm of the random matrices that are generated.
    Defaults to 10**-3
    num_pts : int
    The number of matrices, E, that will be used in the
    algorithm. Defaults to 20.
    '''

    z, eig_vec = la.eig(A)

    pseudospectra = []
    Eig_mats      = []

    n,m = A.shape

    I_mat = np.eye(n)

    while len(Eig_mats) < num_pts:
    	matrix = np.random.rand(n,n)
    	matrix = (matrix/la.norm(matrix)) * epsilon

    	if la.norm(matrix) == epsilon:
    		Eig_mats.append(matrix)
    		z1, eig_vec = la.eig(A+matrix)
    		pseudospectra.append(z1)

    # plot the psuedospectra
    # plot the original eig_val
    plt.scatter(np.real(z),np.imag(z), color = 'red')
    plt.scatter(np.real(pseudospectra), np.imag(pseudospectra))
    plt.show() 

# Problem 2
def ps_contour_plot(A, m = 150, epsilon_vals=None):
    '''Plots the pseudospectrum of the matrix A as a contour plot.  Also,
    plots the eigenvalues.
    Parameters:
        A : square, 2D ndarray
            The matrix whose pseudospectrum is to be plotted
        m : int
            accuracy
        epsilon_vals : list of floats
            If k is in epsilon_vals, then the epsilon-pseudospectrum
            is plotted for epsilon=10**-k
            If epsilon_vals=None, the defaults of plt.contour() are used
            instead of any specified values.
    '''

    # given by the lab
    T = la.schur(A)[0]
    eigs_A = np.diagonal(T,0)
    xvals,yvals = ps_grid(eigs_A,m)
    sigmin = np.zeros((m,m))
    n,r = A.shape

    for k in range(m-1):
    
        for j in range(m-1):
            T1 = (xvals[k] + 1j*yvals[j])*np.eye(n,n) - T
            T2 = T1.T.conjugate()
    
            sigold = 0
            qold = np.zeros((n,1))
            beta = 0
    
            H = np.zeros((n,n))
            q =np.random.normal(size = (n,1))+1j*np.random.normal(size =(n,1))
            q /= la.norm(q)
            for p in xrange(n-2):
                b1 = np.linalg.solve(T2,q)
                b2 = np.linalg.solve(T1,b1)
    
                v = b2 -beta*qold
                alpha = np.real(np.vdot(q,v))
                v = v - alpha*q
                beta = la.norm(v)
                qold = q
    
                q =v/beta
                H[p+1, p] = beta
                H[p, p+1] = beta
                H[p, p] = alpha
                sig =abs(max(la.eigvals(H[:p+1, :p+1])))
                if abs(sigold/sig - 1) < .001:
                    break
                sigold = sig
            sigmin[j,k] = np.sqrt(sig)

    # show the different sizes of epsilons
    plt.contour(xvals, yvals, np.log10(sigmin), levels = epsilon_vals)

    plt.scatter(np.real(eigs_A), np.imag(eigs_A))
    plt.show()

def problem3(n=120,epsilon=.001,num_pts=20):
	'''
	Parameters:
	n : int
	The size of the matrix to use. Defaults to a 120x120 matrix.
	epsilon : float
	The norm of the random matrices that are generated.
	Defaults to 10**-3
	num_pts : int
	The number of matrices, E, that will be used in the
	algorithm. Defaults to 20.
	'''
	ones_119 = np.ones(119)
	ones_118 = np.ones(118)

	# define the matrix via diagonals
	A_diag_1 = np.diag( -1  * ones_118,2)
	A_diag_2 = np.diag(  1j * ones_119,1)	
	A_diag_3 = np.diag( -1j * ones_119,-1)
	A_diag_4 = np.diag(ones_118 ,-2)

	# make the Matrix and it's hermitian
	A_mat  = A_diag_1 + A_diag_2 + A_diag_3 + A_diag_4
	A_herm = A_diag_1 + A_diag_2 + A_diag_3 + (-1)*A_diag_4

	# NonNormal
	A_diag_5 = np.diag(1j*np.random.rand(n))
	A_nrml = A_diag_1 + A_diag_2 + A_diag_5

	# plot the pseudospectra
	ps_scatter_plot(A_mat)
	ps_scatter_plot(A_herm)
	ps_scatter_plot(A_nrml)

	"""
	B = np.array([[-1,0,0], 
				  [ 0,1,0],
				  [ 0,0,1j]])

	B_nrml = np.array([[-1,-1,-1],
					   [ 0, 1, 1],
					   [ 0, 0, 1j]])

	ps_contour_plot(B)
	ps_contour_plot(B_nrml)
	"""
    

def ps_grid(eig_vals, grid_dim):
    """
        Computes the grid on which to plot the pseudospectrum
        of a matrix. This is a helper function for ps_contour_plot().
        """
    x0, x1 = min(eig_vals.real), max(eig_vals.real)
    y0, y1 = min(eig_vals.imag), max(eig_vals.imag)
    xmid = (x0 + x1) /2.
    xlen1 = x1 - x0 +.01
    ymid = (y0 + y1) / 2.
    ylen1 = y1 - y0 + .01
    xlen = max(xlen1, ylen1/2.)
    ylen = max(xlen1/2., ylen1)
    x0 = xmid - xlen
    x1 = xmid + xlen
    y0 = ymid - ylen
    y1 = ymid + ylen
    x = np.linspace(x0, x1, grid_dim)
    y = np.linspace(y0, y1, grid_dim)
    return x,y

def test(): 
	A = np.zeros((120, 120))
	eyes = np.ones(119)
	ones = np.ones(118)
	A1 = np.diag(-1*ones, 2)
	A2 = np.diag(1j*eyes, 1)
	A3 = np.diag(-1j*eyes, -1)
	A4 = np.diag(ones, -2)
	A = A1+ A2+ A3+A4

	#print Aa
	ps_scatter_plot(A)
	ps_contour_plot(A)

#test()
problem3()