"""Cartesian geometry description for structured meshes"""

import numpy as np


class CartesianGeometry:
    """
    Defines an hexaedral geometry with a cartesian grid
    xL are all coordinates of left faces and xR are coordinates of right faces, so that the domain is \prod_k [xL[k],xR[k]]
    N_elem_dir are the number of element for each direction
    BC Boundary conditions (0 periodic, 1 dirichlet) in order x[0]=xL[0], x[1]=xL[1], ..., x[0]=xR[0], x[1]=xR[0], ... x[dim-1]=xR[dim-1] 
    """
    def __init__(self, xL, xR, N_elem_dir, geometry_folder, BC=None):
        assert len(xL) == len(xR) and len(xL) == len(N_elem_dir)
        self.geometry_folder = geometry_folder
        self.dim = np.shape(xL)[0]
        self.xL  = xL
        self.xR  = xR
        self.domainLength_dir = xR - xL
        self.N_elem_dir       = N_elem_dir
        self.N_elem           = np.prod(self.N_elem_dir)
        self.dx = self.domainLength_dir/self.N_elem_dir
        self.dx_min = np.min(self.dx)
        self.xx = {}
        for k in range(self.dim):
            self.xx[k] = np.linspace(self.xL[k],self.xR[k],self.N_elem_dir[k]+1)
        if BC is None:
            self.BC = np.zeros(2*self.dim, dtype=np.int32) # periodic BC
        else:
            self.BC = BC
        for k in range(self.dim):
            if self.BC[k]==0:                             # Periodic BC 
                assert self.BC[k] == self.BC[k+self.dim]  # Check that is periodic also on the other side
                self.xx[k] = self.xx[k][:-1]