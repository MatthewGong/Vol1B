import numpy as np
from matplotlib import pyplot as plt

def plot_complex(f, xbounds, ybounds, res=401):

    '''Plot the complex function f.
    INPUTS:
    f - A function handle. Should represent a function from C to C.
    xbounds - A tuple (xmin, xmax) describing the bounds on the real part of the domain.
    ybounds - A tuple (ymin, ymax) describing the bounds on the imaginary part of the domain.
    res - A scalar that determines the resolution of the plot (number of points
        per side). Defaults to 401.
    
    OUTPUTS:
    graph of the function f(z)
    '''
    x_vec = np.linspace(xbounds[0], xbounds[1], res)
    y_vec = np.linspace(ybounds[0], ybounds[1], res)

    X,Y = np.meshgrid(x_vec,y_vec)
    Z = X + 1j*Y

    plt.pcolormesh(X,Y, np.angle(f(Z)), cmap = 'hsv', vmin = -np.pi, vmax = np.pi)
    plt.show()

def problem2():
    '''Create the plots specified in the problem statement.
    >>>>>>>Please title each of your plots!!!!<<<<<<<
    Print out the answers to the questions.
    '''
    f1 = lambda z: z**2
    f2 = lambda z: z**3
    f3 = lambda z: z**4

    f4 = lambda z: z**3 - 1j*z**4 - 3*z**6

    xbounds = (-1,1)
    ybounds = (-1,1)

    plt.title("Z**2")
    plot_complex(f1,xbounds,ybounds)


    plt.title("Z**3")
    plot_complex(f2,xbounds,ybounds)


    plt.title("Z**4")
    plot_complex(f3,xbounds,ybounds)

    print "As the power increases so does the order of the function as shown by the number of times the color cycles through the rainbow around the point"

    plt.title("Z**3 - iz**4 - 3z**6")
    plot_complex(f4,xbounds,ybounds)


    print "The direction of the rainbows being counterclockwise shows that point is a zero instead of a singular point. We see there are three zeros of order one and on zero of order 3 which lines up with what we know about the function having 6 roots"

    
def problem3():
    '''Create the plots specified in the problem statement.
    Print out the answers to the questions.
    '''

    f1 = lambda z: 1/z
    f2 = lambda z: z

    xbounds = (-1,1)
    ybounds = (-1,1)

    plt.title("1/Z")
    plot_complex(f1,xbounds,ybounds)


    plt.title("Z")
    plot_complex(f2,xbounds,ybounds)

    print "the colors move clockwise for 1/Z indicating it's a singular point instaed of a zero"

    f3 = lambda z: z**-2
    f4 = lambda z: z**-3
    f5 = lambda z: z**2 + 1j*z**-1 + z**-3

    plt.title("1/Z^2")
    plot_complex(f3,xbounds,ybounds)
    plt.title("1/Z^3")
    plot_complex(f4,xbounds,ybounds)
    plt.title("Z^2 + i/Z + 1/Z^2")
    plot_complex(f5,xbounds,ybounds)

    print "like with zeros the order of the pole is shown by how many times the colors cycle around it"

def problem4():
    '''For each plot, create the graph using plot_complex and print out the
    number and order of poles and zeros below it.'''

    f1 = lambda z: np.exp(z)
    f2 = lambda z: np.tan(z)
    f3 = lambda z: (16*z**4 + 32*z**3 + 32*z**2 + 16*z +4)/(16*z**4 -16*z**3 + 5*z**2)

    xbounds = (-8,8)
    ybounds = (-8,8)


    plt.title("e^z")
    plot_complex(f1,xbounds,ybounds)
    plt.title("tan(Z)")
    plot_complex(f2,xbounds,ybounds)
    plt.title("16*z**4 + 32*z**3 + 32*z**2 + 16*z +4 / 16*z**4 -16*z**3 + 5*z**2")
    plot_complex(f3,(-1,1),(-1,1))

    print "e^z has no pols or zeros but has infinite order"
    print "tan(Z) alternates between poles and zeros and has an infinite number of them"
    print "The polynomial has two second order zeros and 2 first order poles and one second order pole "

def problem5():
    '''
    For each polynomial, print out each zero and its multiplicity.
    Organize this so the output makes sense.
    '''
    f1 = lambda z: -2*z**7 + 2*z**6 - 4*z**5 + 2*z**4 - 2*z**3 - 4*z**2 + 4*z - 4
    f2 = lambda z: z**7 + 6*z**6 - 131*z**5 - 419*z**4 + 4906*z**3 - 131*z**2 - 420*z + 4900

    xbounds = (-2,2)
    ybounds = (-2,2)



    plt.title("Function")
    plot_complex(f1,xbounds,ybounds)

    print "There are 7 first order zeros, thus the multiplicity is 7."


    xbounds = (-25,25)
    ybounds = (-25,25)
    
    plt.title("Function")
    plot_complex(f2,xbounds,ybounds)
    
    print "There are 2 second order zeros and 3 first order zeros"

def problem6():
    '''Create the plots specified in the problem statement.
    Print out the answers to the questions.
    '''

    f1 = lambda z: np.sin(1/(100*z))

    xbounds = (-1,1)
    ybounds = (-1,1)

    plt.title("Function")
    plot_complex(f1,xbounds,ybounds)

    print "It looks just like F(z) = z"

    xbounds = (-.01,.01)
    ybounds = (-.01,.01)

    plt.title("Function")
    plot_complex(f1,xbounds,ybounds)

    print "Zooming in we can see there are an infinite amount of poles"

    f2 = lambda z: z + 1000*z**2

    xbounds = (-1,1)
    ybounds = (-1,1)

    plt.title("Function")
    plot_complex(f2,xbounds,ybounds)

    print "It looks just like F(z) = z"

    xbounds = (-.005,.0051)
    ybounds = (-.005,.0051)

    plt.title("Function")
    plot_complex(f2,xbounds,ybounds)

    print "Zooming in we can see there are an two distinct zeros at 0 and -1/1000"


def problem7():
    '''Create the plots specified in the problem statement.
    Print out the answers to the questions.
    '''
    f1 = lambda z: np.sqrt(z)

    xbounds = (-1,1)
    ybounds = (-1,1)

    plt.title("Function")
    plot_complex(f1,xbounds,ybounds)

    print "Only two of the colors show up, blue and green."


    f2 = lambda z: -np.sqrt(z)

    plt.title("Function")
    plot_complex(f2,xbounds,ybounds)


    print "The rest of the colors are present."

def extraCredit():
    '''
    Create a really awesome complex plot. You can do whatever you want, as long as 
    it's cool and you came up with it on your own.
    You can also animate one of the plots in the lab (look up matplotlib animation)
    Title your plot or print out an explanation of what it is.
    '''
    f1 = lambda z: np.sqrt(z) + np.tan(z)/(1-np.tan(z)) + z**3

    xbounds = (-4,4)
    ybounds = (-4,4)

    plt.title("Function")
    plot_complex(f1,xbounds,ybounds)

    print "It's pretty and things get weird around -2.4"
