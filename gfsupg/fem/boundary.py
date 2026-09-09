"""Boundary-index helpers, periodic/Neumann-aware 1D sparse assembly, and Dirichlet BC container"""

import numpy as np


def boundary_index_dir(ix, direction, geom, n_dof_dir):
    """ Actual implementation of boundary conditions for matrices: we look for the appropriate index periodic/Neumann"""
    if geom.BC[direction] == 0: #periodic
        return ix%n_dof_dir
    else: #Neumann (and Dirichlet doesn't really matter)
        return max(min(ix,n_dof_dir - 1),0)

def assemble_1D_sparse_matrix(xx_dofs_dir, FEM1D_dir, stencil_dir, geom, n_dof_dir, direction):
    """Assemble of the sparse matrix in form of 4-tensor index_i, index_j, values 
    such that matrix[index_i[ix,iy,jx_stencil,jy_stencil], index_j[ix,iy,jx_stencil,jy_stencil]] += values[ix,iy,jx_stencil,jy_stencil]
    where ix, iy are the dof indexes of the rows and
    jx_stencil and jy_stencil are the dofs of the columns (only in the stencil of the matrix)
    """
    Na, Nb  = len(xx_dofs_dir), 3*FEM1D_dir.degree

    index_i = np.zeros((Na,Nb), dtype=np.int64)
    index_j = np.zeros((Na,Nb), dtype=np.int64)
    values  = np.zeros((Na,Nb))

    for ix, xi in enumerate(xx_dofs_dir):
        ix_cell = ix//FEM1D_dir.degree
        ix_dof  = ix%FEM1D_dir.degree

        index_i[ix,:] = ix

        # 1D formalism
        jxl = (ix_cell - 1)*FEM1D_dir.degree 
        jxr = (ix_cell + 2)*FEM1D_dir.degree 

        for jx_stencil, jx in enumerate(range(jxl,jxr)):
            j = boundary_index_dir(jx,direction, geom, n_dof_dir) 
            index_j[ix,jx_stencil] = j
            values[ix,jx_stencil]  = stencil_dir[ix_dof,jx_stencil]
    
    return index_i, index_j, values

class Dirichlet_BC_set:
    """Container for Dirichlet boundary indexes and values per variable."""

    def __init__(self, indexes, dirichlet_vector):
        self.indexes = indexes
        self.dirichlet_vector = dirichlet_vector
        self.vars  = list(dirichlet_vector.keys())


