"""Volume I Lab 18: Profiling and Optimizing Python Code
<Name>Matt Gong
<Class>
<Date>
"""

import scipy
import time
import numpy as np

from scipy import linalg
from numba import jit
from numba import double

# Problem 1
def compare_timings(f, g, *args):
    """Compares the timings of 'f' and 'g' with arguments '*args'.

    Inputs:
        f (callable): first function to compare.
        g (callable): second function to compare.
        *args (any type): arguments to use when callings functions
            'f' and 'g'
    Returns:
        comparison (string): The comparison of the runtimes of functions
            'f' and 'g' in the following format :
                Timing for <f>: <time>
                Timing for <g>: <time>
            where the values inside <> vary depending on the inputs)
    """

    start_f = time.time()
    f(*args)
    end_f   = time.time() - start_f

    start_g = time.time()
    g(*args)
    end_g   = time.time() - start_g

    return "It took " + str(end_f) + " seconds for " + str(f) + " and it took " + str(end_g) + " seconds for " + str(g) 



# Problem 2
def LU(A):
    """Returns the LU decomposition of a square matrix."""
    n = A.shape[0]
    U = np.array(np.copy(A), dtype=float)
    L = np.eye(n)
    for i in range(1,n):
        for j in range(i):
            L[i,j] = U[i,j]/U[j,j]
            for k in range(j,n):
                U[i,k] -= L[i,j] * U[j,k]
    return L,U

def LU_opt(A):
    """Returns the LU decomposition of a square matrix."""

    n = A.shape[0]
    U = np.array(np.copy(A), dtype = float)
    L = np.eye(n)

    for i in range(1,n):
        for j in range(i):
            L[i,j]   = U[i,j]/U[j,j]
            U[i,j:] -= L[i,j] * U[j,j:] 


def compare_LU(A):
    """Prints a comparison of LU and LU_opt with input of the matrix A."""
    times = compare_timings(LU, LU_opt, A)

    print times
    return times


# Problem 3
def mysum(X):
    """Return the sum of the elements of X.
    Inputs:
        X (array) - a 1-D array
    """
    total = 0

    for i in X:
        total += i

    return total

def compare_sum(X):
    """Prints a comparison of mysum and sum and prints a comparison
    of mysum and np.sum."""

    # test against built in sum function
    print compare_timings(mysum, sum ,X)
    
    print compare_timings(mysum, np.sum ,X)

        


# Problem 4
def fib(n):
    """A generator that yields the first n Fibonacci numbers."""
 
    a,b = 0,1

    for i in xrange(1,n+1):
        a, b = b, a + b

        yield a


# Problem 5
def foo(n):
    """(A part of this Problem is to be able to figure out what this
    function is doing. Therefore, no details are provided in
    the docstring.)
    """
    my_list = []
    for i in range(n):
        num = np.random.randint(-9,9)
        my_list.append(num)
    evens = 0
    for j in range(n):
        if j%2 == 0:
            evens += my_list[j]
    return my_list, evens

# Problem 5
def foo_opt(n):
    """An optimized version of 'foo'"""
    
    # draws a random int from -9 to 9 and adds it to a list
    my_list = [np.random.randint(-9,9) for i in xrange(n)]

    # adds up every other element in the list
    evens = sum(my_list[::2])

    return my_list, evens

def compare_foo(n):
    """Prints a comparison of foo and foo_opt"""

    print compare_timings(foo,foo_opt, n)



# Problem 6
def pymatpow(X, power):
    """ Return X^{power}.

    Inputs:
        X (array) - A square 2-D NumPy array
        power (int) - The power to which we are taking the matrix X.
    Returns:
        prod (array) - X^{power}
    """
    prod = X.copy()
    temparr = np.empty_like(X[0])
    size = X.shape[0]
    for n in xrange(1, power):
        for i in xrange(size):
            for j in xrange(size):
                tot = 0.
                for k in xrange(size):
                    tot += prod[i,k] * X[k,j]
                temparr[j] = tot
            prod[i] = temparr
    return prod

@jit(nopython = True)
def numba_matpow(X, power):
    """ Return X^{power}. Compiled using Numba.

    Inputs:
        X (array) - A square 2-D NumPy array
        power (int) - The power to which we are taking the matrix X.
    Returns:
        prod (array) - X^{power}
    """

    prod = X.copy()
    temparr = np.empty_like(X[0])
    size = X.shape[0]
    for n in xrange(1, power):
        for i in xrange(size):
            for j in xrange(size):
                tot = 0.
                for k in xrange(size):
                    tot += prod[i,k] * X[k,j]
            temparr[j] = tot
        prod[i] = temparr
    return prod



def numpy_matpow(X, power):
    """ Return X^{power}.

    Inputs:
        X (array) - A square 2-D NumPy array
        power (int) - The power to which we are taking the matrix X.
    Returns:
        prod (array) - X^{power}
    """

    prod = X.copy()

    for n in xrange(1,power):
        prod = prod.dot(X)

    return prod

def compare_matpow(X, power):
    """Prints a comparison of pymatpow and numba_matpow and prints a
    comparison of pymatpow and numpy_matpow"""
    numba_matpow(X,power)
    
    print compare_timings(pymatpow,numba_matpow, X, power)
    print compare_timings(pymatpow,numpy_matpow, X, power)

# Problem 7
def pytridiag(a,b,c,d):
    """Solve the tridiagonal system Ax = d where A has diagonals a, b, and c.

    Inputs:
        a, b, c, d (array) - All 1-D NumPy arrays of equal length.
    Returns:
        x (array) - solution to the tridiagonal system.
    """
    n = len(a)

    # Make copies so the original arrays remain unchanged
    aa = np.copy(a)
    bb = np.copy(b)
    cc = np.copy(c)
    dd = np.copy(d)

    # Forward sweep
    for i in xrange(1, n):
        temp = aa[i]/bb[i-1]
        bb[i] = bb[i] - temp*cc[i-1]
        dd[i] = dd[i] - temp*dd[i-1]

    # Back substitution
    x = np.zeros_like(a)
    x[-1] = dd[-1]/bb[-1]
    for i in xrange(n-2, -1, -1):
        x[i] = (dd[i]-cc[i]*x[i+1])/bb[i]

    return x

def init_tridiag(n):
    """Initializes a random nxn tridiagonal matrix A.

    Inputs:
        n (int) : size of array

    Returns:
        a (1-D array) : (-1)-th diagonal of A
        b (1-D array) : main diagonal of A
        c (1-D array) : (1)-th diagonal of A
        A (2-D array) : nxn tridiagonal matrix defined by a,b,c.
    """
    a = np.random.random_integers(-9,9,n).astype("float")
    b = np.random.random_integers(-9,9,n).astype("float")
    c = np.random.random_integers(-9,9,n).astype("float")


    # check that there's no 0's in the matrix
    a[a==0] = 1
    b[b==0] = 1
    c[c==0] = 1

    A = np.zeros((b.size,b.size))
    np.fill_diagonal(A,b)
    np.fill_diagonal(A[1:,:-1],a[1:])
    np.fill_diagonal(A[:-1,1:],c)
    return a,b,c,A

@jit
def numba_tridiag(a,b,c,d):
    """Solve the tridiagonal system Ax = d where A has diagonals a, b, and c.

    Inputs:
        a, b, c, d (array) - All 1-D NumPy arrays of equal length.
    Returns:
        x (array) - solution to the tridiagonal system.
    """
        # Make copies so the original arrays remain unchanged

    n = len(a)

    aa = np.copy(a)
    bb = np.copy(b)
    cc = np.copy(c)
    dd = np.copy(d)

    # Forward sweep
    for i in xrange(1, n):
        temp = aa[i]/bb[i-1]
        bb[i] = bb[i] - temp*cc[i-1]
        dd[i] = dd[i] - temp*dd[i-1]

    # Back substitution
    x = np.zeros_like(a)
    x[-1] = dd[-1]/bb[-1]
    for i in xrange(n-2, -1, -1):
        x[i] = (dd[i]-cc[i]*x[i+1])/bb[i]

    return x

def compare_tridiag():
    """Prints a comparison of numba_tridiag and pytridiag and prints
    a comparison of numba_tridiag and scipy.linalg.solve."""

    n = 1000000
    a, b, c, A = init_tridiag(n)

    # create a random diagonal
    d = np.random.random_integers(-9,9,n).astype('float')

    print compare_timings(numba_tridiag, pytridiag, a, b, c, d)

    # comparing on a 1000 sized system
    n_small = 1000
    a, b, c, A = init_tridiag(n_small)

    d_small = np.random.random_integers(-9,9,n).astype('float')

    
    # comparing with scipy
    start_f = time.time()
    numba_tridiag(a, b, c, d_small)
    end_f   = time.time() - start_f

    start_g = time.time()
    scipy.linalg.solve(A,d_small)
    end_g   = time.time() - start_g

    return "It took " + str(end_f) + " seconds for numba_tridiag and it took " + str(end_g) + " seconds for scipylinalg"


# Problem 8
def laplacian(A):

    n,n = A.shape

    D = np.zeros((n,n))

    for i in range(len(A)):
        for j in range(len(A)):
            if i == j:
                D[i][j] = sum(A[i])

    laplacian = D-A

    return laplacian

@jit
def laplace_fast(A):

    D = np.zeros(A.shape)

    for i in range(len(A)):

        D[i][i] = sum(A[i])

    return D-A
  
def compare_old():
    """Prints a comparison of an old algorithm from a previous lab
    and the optimized version."""
    A = np.random.random((400,400))

    laplace_fast(A)

    print compare_timings(laplacian,laplace_fast, A)

    print "The laplace only cares about the diagonal so we can forgo the second loop. jit also improves the speed. "
