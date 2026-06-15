import numpy as np

def lagrange_basis(nodes,x,k):
    y=np.zeros(x.size)
    for ix, xi in enumerate(x):
        tmp=[(xi-nodes[j])/(nodes[k]-nodes[j])  for j in range(len(nodes)) if j!=k]
        y[ix]=1.
        for z in range(len(tmp)):
            y[ix]=y[ix]*tmp[z]
    return y

def lagrange_basis_deriv(nodes,x,k):
    y=np.zeros(x.size)
    for ix, xi in enumerate(x):
        for i in range(len(nodes)):
            p=1.
            if i!=k:
                for j in range(len(nodes)):
                    if j!=i and j!=k:
                        p = p*(xi-nodes[j])/(nodes[k]-nodes[j])
                y[ix]+= p/(nodes[k]-nodes[i])
    return y

def equispaced(order):
    '''
    Takes input d and returns the vector of d equispaced points in [-1,1]
    And the integral of the basis functions interpolated in those points
    '''
    nodes= np.linspace(-1,1,order)
    w= np.zeros(order)
    n_quad, w_quad = lglnodes(order+1,1e-15)
    for k in range(order):
        basis_quad=lagrange_basis(nodes,n_quad,k)
        w[k] = 0.
        for z in range(len(n_quad)):
            w[k]=w[k]+basis_quad[z]*w_quad[z]

    return nodes, w


def lglnodes(n,epss=1e-15):
    '''
    Python translation of lglnodes.m
    Computes the Legendre-Gauss-Lobatto nodes, weights and the LGL Vandermonde
    matrix. The LGL nodes are the zeros of (1-x^2)*P'_N(x). Useful for numerical
    integration and spectral methods.
    Parameters
    ----------
    n : integer, requesting an nth-order Gauss-quadrature rule on [-1, 1]
    Returns
    -------
    (nodes, weights) : tuple, representing the quadrature nodes and weights.
                       Note: (n+1) nodes and weights are returned.

    Example
    -------
    >>> from lglnodes import *
    >>> (nodes, weights) = lglnodes(3)
    >>> print(str(nodes) + "   " + str(weights))
    [-1.        -0.4472136  0.4472136  1.       ]   [0.16666667 0.83333333 0.83333333 0.16666667]
    Notes
    -----
    Reference on LGL nodes and weights:
      C. Canuto, M. Y. Hussaini, A. Quarteroni, T. A. Tang, "Spectral Methods
      in Fluid Dynamics," Section 2.3. Springer-Verlag 1987
    Written by Greg von Winckel - 04/17/2004
        Contact: gregvw@chtm.unm.edu
    Translated and modified into Python by Jacob Schroder - 9/15/2018
    '''

    w = np.zeros((n+1,))
    x = np.zeros((n+1,))
    xold = np.zeros((n+1,))

    # The Legendre Vandermonde Matrix
    P = np.zeros((n+1,n+1))

    # Use the Chebyshev-Gauss-Lobatto nodes as the first guess
    for i in range(n+1):
        x[i] = -np.cos(np.pi*i / n)


    # Compute P using the recursion relation
    # Compute its first and second derivatives and
    # update x using the Newton-Raphson method.

    xold = 2.0

    for i in range(100):
        xold = x

        P[:,0] = 1.0
        P[:,1] = x

        for k in range(2,n+1):
            P[:,k] = ( (2*k-1)*x*P[:,k-1] - (k-1)*P[:,k-2] ) / k

        x = xold - ( x*P[:,n] - P[:,n-1] )/( (n+1)*P[:,n])

        if (max(np.abs(x - xold)) < epss ):
            break

    w = 2.0 / ( (n*(n+1))*(P[:,n]**2))

    return x, w


def lgwt(N):
    # lgwt.m
    # This script is for computing definite integrals using Legendre-Gauss 
    # Quadrature. Computes the Legendre-Gauss nodes and weights  on an interval
    # [a,b] with truncation order N
    # Suppose you have a continuous function f(x) which is defined on [a,b]
    # which you can evaluate at any x in [a,b]. Simply evaluate it at all of
    # the values contained in the x vector to obtain a vector f. Then compute
    # the definite integral using sum(f.*w);
    a=-1; b=1
    N=N-1
    N1=N+1; N2=N+2
    xu=np.linspace(-1,1,N1)
    # Initial guess
    zz = np.linspace(0,N,N1)
    y=np.cos((2*zz+1)*np.pi/(2*N+2))+(0.27/N1)*np.sin(np.pi*xu*N/N2)
    # Legendre-Gauss Vandermonde Matrix
    L=np.zeros((N1,N2))
    # Derivative of LGVM
    Lp=np.zeros(N1)
    # Compute the zeros of the N+1 Legendre Polynomial
    # using the recursion relation and the Newton-Raphson method
    y0=2.*np.ones(N1)
    # Iterate until new points are uniformly within epsilon of old points
    while max(np.abs(y-y0))>1e-15:
        
        
        L[:,0]=1
        
        L[:,1]=y
        
        for k in range(2,N1+1):
            L[:,k]=( (2*k-1)*y*L[:,k-1]-(k-1)*L[:,k-2] )/k

     
        Lp=(N2)*( L[:,N1-1]-y*L[:,N2-1] )/(1-y**2)
        
        y0=y
        y=y0-L[:,N2-1]/Lp
        
    # Linear map from[-1,1] to [a,b]
    x=(a*(1-y)+b*(1+y))/2
    # Compute the weights
    w=(b-a)/((1-y**2)*Lp**2)*(N2/N1)**2

    return x,w


def nodes_weights(N, nodes_type):
    """
    Nodes and weights of associated quadrature in [0,1]
    """
    if nodes_type=="equispaced":
        nn,ww = equispaced(N)
    elif nodes_type=="gaussLobatto":
        nn,ww = lglnodes(N-1)
    elif nodes_type=="gaussLegendre":
        nn,ww = lgwt(N)
    else:
        raise ValueError("nodes_type does not exists")
    nn = nn/2.+0.5
    ww = ww/2.
    return nn, ww