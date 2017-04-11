# spec.py
"""Volume 1, Lab 16: Importance Sampling and Monte Carlo Simulations.
<Name> Matthew Gong
<Class>
<Date>
"""

import numpy as np
import scipy.stats as stats

from matplotlib import pyplot as plt

# Problem 1 
def prob1(n):
    """Approximate the probability that a random draw from the standard
    normal distribution will be greater than 3.
    Returns: your estimated probability.
    """

    # create a giant draw from a normal distribution
    random_draws = np.random.normal(loc= 0, scale = 1, size = n)

    # mask the values
    mask = random_draws > 3

    return np.sum(mask)/float(n)

# Problem 2
def prob2():
    """Answer the following question using importance sampling: 
            A tech support hotline receives an average of 2 calls per 
            minute. What is the probability that they will have to wait 
            at least 10 minutes to receive 9 calls?
    Returns:
        IS (array) - an array of estimates using 
            [5000, 10000, 15000, ..., 500000] as number of 
            sample points."""
     
    N = np.arange(5000, 500001, 5000)

    
    
    def importance(N):
        h = lambda x : x > 10
        f = lambda x : stats.gamma(9, scale = 0.5).pdf(x)
        g = lambda x : stats.norm(loc = 10).pdf(x)

        # create a draw of size n
        random_draw = np.random.normal(10, size = N)
        return 1./N *np.sum((h(random_draw) * f(random_draw) / g(random_draw)))

    return np.vectorize(importance)(N)



# Problem 3
def prob3():
    """Plot the errors of Monte Carlo Simulation vs Importance Sampling
    for the prob2()."""

    h = lambda x: x > 10

    N = range(5000,500001, 5000)

    estimates = []

    for n in N:
        random_draw = np.random.gamma(9, scale = 0.5, size = n)

        estimate = 1./n * np.sum(h(random_draw))
        estimates.append(estimate)

    # arrayify it
    estimates = np.array(estimates)

    m   = 1 - stats.gamma(a = 9, scale = 0.5).cdf(10)
    
    y   =  abs(estimates - m)
    y_2 = abs(prob2() - m)

    plt.plot(N,y)
    plt.plot(N,y_2)

    plt.show()


# Problem 4
def prob4():
    """Approximate the probability that a random draw from the
    multivariate standard normal distribution will be less than -1 in 
    the x-direction and greater than 1 in the y-direction.
    Returns: your estimated probability"""


    N = 500000
    random_draws = np.random.multivariate_normal(mean = [-1,1], cov =[[1,0],[0,1]], size = N)

    h = lambda x: x[0] < -1 and x[1] > 1
    f = lambda x: stats.multivariate_normal(mean = [ 0, 0]).pdf(x)
    g = lambda x: stats.multivariate_normal(mean = [-1, 1]).pdf(x)

    probability = [h(random_draws[i]) * f(random_draws[i]) / g(random_draws[i]) for i in range(N)]

    return 1./N * np.sum(probability)

