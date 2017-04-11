import numpy as np

from scipy import linalg as la
from scipy import fftpack as fft

from matplotlib import pyplot as plt


def arnoldi(b, Amul, k, tol=1E-8):
    '''Perform `k' steps of the Arnoldi iteration on the linear 
        operator defined by `Amul', starting with the vector 'b'.
    '''

    Q = np.zeros((len(b),k+1), dtype = np.complex128)
    H = np.zeros((k+1,k), dtype = np.complex128)

    Q[:,0] = b/np.sqrt(np.vdot(b,b))

    for j in xrange(k):
        Q[:,j+1] = Amul(Q[:,j])

        # perform interation
        for i in xrange(j+1):
            H[i,j]   = np.vdot(Q[:, i], Q[:,j+1])
            Q[:,j+1] = Q[:,j+1] - H[i,j]*Q[:,i]

        # set subdiagonal
        H[j+1,j] = np.sqrt(np.vdot(Q[:,j+1],Q[:,j+1]))
        
        if abs(H[j+1,j]) < tol:
            return H[:j+1,:j+1], Q[:,:j+1]

        Q[:,j+1] = Q[:,j+1] /  H[j+1,j]

    # if we didn't terminate early, return the whole matrix
    return H[:-1,:], Q


def ritz(Amul, dim, k, iters):
    ''' Find `k' Ritz values of the linear operator defined by `Amul'.
    '''

    if iters > dim or iters < k:
        return ValueError("The number of iterations must be between dim and k")

    # seeded guess
    b = np.random.rand(dim)
    
    H, Q = arnoldi(b, Amul, iters)

    H_eig = la.eig(H, right = False)

    # sort the eigenvalues
    sorted_index = np.argsort(H_eig)
    H_eig[sorted_index[::-1]]

    return H_eig[:k]

def fft_eigs(dim=2**20,k=4):
    '''Return the largest k Ritz values of the Fast Fourier transform
        operating on a space of dimension dim.
    '''
    # Use the fourier transform for Amul
    Amul = fft.fft

    return ritz(Amul, dim, k, k)

def plot_ritz(A, n, iters):
    ''' Plot the relative error of the Ritz values of `A'.
    '''

    eig_val, eig_vec = np.linalg.eig(A)

    A_eig = np.sort(eig_val)
    A_eig = A_eig[::-1] #reverse the order
    A_eig = A_eig[:n] #choose the first n

    relative_error = np.empty((iters, n))

    b = np.random.rand(len(A))

    for k in xrange(1,iters + 1):
        H_k, Q_k = arnoldi(b, A.dot, k)
        val_k, vec_k = np.linalg.eig(H_k)

        # sort and reverse the values
        val_k = np.sort(val_k)
        val_k = val_k[::-1]

        if k < n:
            # prevents trying to use too many values
            for i in xrange(0,k):
               relative_error[k-1, i] = np.sqrt(abs((A_eig[i] - val_k[i])**2)) / la.norm(A_eig[:k])

        else:
            for i in xrange(0,n):
               relative_error[k-1, i] = np.sqrt(abs((A_eig[i] - val_k[i])**2)) / la.norm(A_eig)

    # plot each error as we iterate through time
    x = range(0,iters)
    for error in xrange(0,n): 
        plt.plot(x, relative_error[:,error])

    plt.yscale('log')
    plt.ylim((1e-19,10))
    plt.show()
