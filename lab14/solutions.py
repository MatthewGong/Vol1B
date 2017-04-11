'''
Lab 14 - Newton's Method.
<Name> Matthew Gong
<Class>
<Date>
'''

import numpy as np
from matplotlib import pyplot as plt

def Newtons_method(f, x0, Df, iters=15, tol=.002):
    '''Use Newton's method to approximate a zero of a function.
    
    INPUTS:
    f     - A function handle. Should represent a function from 
            R to R.
    x0    - Initial guess. Should be a float.
    Df    - A function handle. Should represent the derivative 
            of `f`.
    iters - Maximum number of iterations before the function 
            returns. Defaults to 15.
    tol   - The function returns when the difference between 
            successive approximations is less than `tol`.
    
    RETURN:
    A tuple (x, converged, numiters) with
    x           - the approximation to a zero of `f`
    converged   - a Boolean telling whether Newton's method 
                converged
    numiters    - the number of iterations the method computed
    '''
    
    # force the initial guess to be a float
    x0 = float(x0)

    # for the number of iterations do newtons method
    for i in xrange(iters):
        #print i
        
        xk = x0 - f(x0)/Df(x0)

        # terminate early if we're below a certain tolerance
        if abs(x0-xk) < tol:

            # X, it converged, num_iters
            return xk, True, i+1
    
        else:
            # increment the step and repeat
            x0 = xk

        # position, didn't converge, max iters reached
    return xk, False, iters


def prob2():
    '''
    Print the answers to the questions in problem 2.
    '''
    f  = lambda x:  np.cos(x)
    Df = lambda x: -np.sin(x)

    #print Newtons_method(f, 1, Df, 15, 1e-5)
    #print Newtons_method(f, 2, Df, 15, 1e-5) 
 
    print '1. it takes 4 iterations to converge at x0 = 1 and 6 for x0 = 2'
    
    g  = lambda x: np.sin(x)/x - x
    Dg = lambda x: (-(x**2 + np.sin(x) - x*np.cos(x))/x**2)
    zero = Newtons_method(g, 1 ,Dg, 15, 1e-7)

    print '2. the zeros is at ' + str(zero[0])

    h   = lambda x: x**9
    Dh  = lambda x: 9*x**8
    ans = Newtons_method(h, 1, Dh, 100, 1e-5)
    print ans
    print '3. it takes ' + str(ans[2]) + " iterations to because x**9 is very shallow as you get within -.5 to .5"

    k   = lambda x: np.sign(x)*np.power(np.abs(x), 1./3)
    Dk  = lambda x: (1./3)*np.sign(x)*np.power(np.abs(x), -2./3)

    oscil = Newtons_method(k, 0.01, Dk, 100)

    print '4. It never converges since the function is oscilating'

def Newtons_2(f, x0, iters=15, tol=.002):
    '''
    Optional problem.
    Re-implement Newtons method, but without a derivative.
    Instead, use the centered difference method to estimate the derivative.
    '''
    raise NotImplementedError('Newtons Method 2 not implemented')

def plot_basins(f, Df, roots, xmin, xmax, ymin, ymax, numpoints=100, iters=15, colormap='brg'):
    '''Plot the basins of attraction of f.
    
    INPUTS:
    f       - A function handle. Should represent a function 
            from C to C.
    Df      - A function handle. Should be the derivative of f.
    roots   - An array of the zeros of f.
    xmin, xmax, ymin, ymax - Scalars that define the domain 
            for the plot.
    numpoints - A scalar that determines the resolution of 
            the plot. Defaults to 100.
    iters   - Number of times to iterate Newton's method. 
            Defaults to 15.
    colormap - A colormap to use in the plot. Defaults to 'brg'. 
    
    RETURN:
    Returns nothing, but should display a plot of the basins of attraction.
    '''

    xreal = np.linspace(-1.5, 1.5, 700)
    ximag = np.linspace(-1.5, 1.5, 700)
    Xreal, Ximag = np.meshgrid(xreal, ximag)
    Xold = Xreal+1j*Ximag

    diff = 10

    # implement a slightly different version of Newtons Method
    for i in xrange(iters):
        Xnew = Xold - f(Xold)/Df(Xold)
        diff = np.max(np.absolute(Xnew-Xold))
        Xold = Xnew

    def match(x):
        return np.argmin(abs(roots - [x]*len(roots)))

    Xnewest = np.vectorize(match)(Xnew)

    plt.pcolormesh(Xreal,Ximag, Xnewest, cmap=colormap)
    plt.show()

def test():
    f  = lambda x: x**3 - x
    Df = lambda x: 3*x**2 -1 
    plot_basins(f, Df, np.array([-1,0,1]), -1.5, 1.5, -1.5, 1.5, 1000, 50)

def prob5():
    '''
    Using the function you wrote in the previous problem, plot the basins of
    attraction of the function x^3 - 1 on the interval [-1.5,1.5]X[-1.5,1.5]
    (in the complex plane).
    '''
    f  = lambda x: x**3 - 1
    Df = lambda x: 3*x**2
    
    roots = np.array([1, -1j**(1./3), 1j**(2./3)])

    plot_basins(f, Df, roots, -1.5, 1.5, -1.5, 1.5, 1000, 50)
