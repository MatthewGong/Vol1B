import numpy as np
import scipy.sparse as spar
import scipy.linalg as la
from scipy.sparse import linalg as sla

def to_matrix(filename,n):
    '''
    Return the nxn adjacency matrix described by datafile.
    INPUTS:
    datafile (.txt file): Name of a .txt file describing a directed graph. 
        Lines describing edges should have the form '<from node>\t<to node>\n'.
        The file may also include comments.
    n (int): The number of nodes in the graph described by datafile
    RETURN:
        Return a SciPy sparse `dok_matrix'.
    '''

    #create zero matrix to add adjacencies to
    adj = np.zeros((n,n))

    with open(filename,'r') as myfile:
        
        for line in myfile:
            nodes = line.strip().split()

            try:
                adj[nodes[0],nodes[1]] = 1

            except:
                pass


    return adj

def calculateK(A,N):
    '''
    Compute the matrix K as described in the lab.
    Input:
        A (array): adjacency matrix of an array
        N (int): the datasize of the array
    Return:
        K (array)
    '''
    D_matrix = np.sum(A, axis=1)

    for i in xrange(len(D_matrix)):

        # modify adjaceny matrix so sinks have 1's instead of zeros
        if D_matrix[i] == 0:
            A[i, : ]    = 1
            D_matrix[i] = N

        #compute K using broadcasting
        K = (A/D_matrix[:,None]).T

    return K

def iter_solve(adj, N=None, d=.85, tol=1E-5):
    '''
    Return the page ranks of the network described by `adj`.
    Iterate through the PageRank algorithm until the error is less than `tol'.
    Inputs:
    adj - A NumPy array representing the adjacency matrix of a directed graph
    N (int) - Restrict the computation to the first `N` nodes of the graph.
            Defaults to N=None; in this case, the entire matrix is used.
    d     - The damping factor, a float between 0 and 1.
            Defaults to .85.
    tol  - Stop iterating when the change in approximations to the solution is
        less than `tol'. Defaults to 1E-5.
    Returns:
    The approximation to the steady state.
    '''

    K = calculateK(adj,N)

    # normalized vector 
    p_0 = np.ones(N)/N

    # equation was given as
    p_1 = d*K.dot(p_0) + ((1-d) / N*np.ones(N))

    # iterate until sufficiently close to tolerance
    while la.norm(p_1 - p_0) > tol:
        p_0 = p_1
        p_1 = d*K.dot(p_0) + ((1-d) / N*np.ones(N))

    ranks = p_1

    return ranks

def eig_solve( adj, N=None, d=.85):
    '''
    Return the page ranks of the network described by `adj`. Use the
    eigenvalue solver in scipy.linalg to calculate the steady state
    of the PageRank algorithm
    Inputs:
    adj - A NumPy array representing the adjacency matrix of a directed graph
    N - Restrict the computation to the first `N` nodes of the graph.
            Defaults to N=None; in this case, the entire matrix is used.
    d     - The damping factor, a float between 0 and 1.
            Defaults to .85.
    Returns:
    The approximation to the steady state.
    '''

    K = calculateK(adj,N)

    ones_mat = np.ones((N,N))

    Eigan_mat = d*K + ((1-d) / N) * ones_mat

    eig_val, eig_vec = la.eig(Eigan_mat)

    max_eigan_val = np.argmax(np.real(eig_val))

    return eig_vec[:, max_eigan_val] / sum(eig_vec[:, max_eigan_val])

    
def problem5(filename='ncaa2013.csv'):
    '''
    Create an adjacency matrix from the input file.
    Using iter_solve with d = 0.7, run the PageRank algorithm on the adjacency 
    matrix to estimate the rankings of the teams.
    Inputs:
    filename - Name of a .txt file containing data for basketball games. 
        Should contain a header row: 'Winning team,Losing team",
        after which each row contains the names of two teams,
        winning and losing, separated by a comma
    Returns:
    sorted_ranks - The array of ranks output by iter_solve, sorted from highest
        to lowest.
    sorted_teams - List of team names, sorted from highest rank to lowest rank.   
    '''

    teams   = set()
    matches = []


    with open(filename, 'r') as myfile:

        # build the list of teams that competed and the matches they played
        for line in myfile:
            nodes = line.strip().split(',')
            if len(nodes) == 2:
                teams.add(nodes[0])
                teams.add(nodes[1])
                matches.append(nodes)
   
    # create the adjacency matrix
    teams_ind = list(teams)
    adj = np.zeros((len(teams_ind),len(teams_ind)))
  
    with open(filename, 'r') as myfile:

        for line in myfile:
            game = line.strip().split(',')
            if len(game) ==2:
                j = teams_ind.index(game[0])
                i = teams_ind.index(game[1])
                adj[i,j] = 1
    
    # rank the teams
    ranks = iter_solve(adj,len(teams_ind))

    # match team to their ranks
    ranks_teams = zip(ranks,teams_ind)

    # sort the order of the teams
    ranks_teams.sort()

    sorted_teams = [s for (i,s) in ranks_teams]
    sorted_ranks = [i for (i,s) in ranks_teams]

    print "From the rankings we got in this porblem, it is surprising that none of the top 4 made to the Final Four. Wichita was the lowest ranked to make it into the Final Four."

    return sorted_ranks[::-1], sorted_teams[::-1]
    
def problem6():
    '''
    Optional problem: Load in and explore any one of the SNAP datasets.
    Run the PageRank algorithm on the adjacency matrix.
    If you can, create sparse versions of your algorithms from the previous
    problems so that they can handle more nodes.
    '''
    pass

def test():
    N = 8
    A = to_matrix('datafile.txt',8)
    calculateK(A,N)
    iter_solve(A,N)
    eig_solve(A,N)
    print problem5()

test()