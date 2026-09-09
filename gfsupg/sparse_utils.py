"""Small sparse-matrix utility routines used by the FEM assembly"""

import numpy as np
from scipy.sparse import dia_matrix
import scipy.sparse as sparse


def get_stencil_indexes(i_cell, degree):
    """Return left/right stencil bounds for a 1D cell index."""

    jl = (i_cell-1)*degree + degree
    jr = (i_cell+2)*degree + degree
    
    return jl, jr

def invert_lumped_matrix(lump):
    """Return diagonal inverse of a lumped mass matrix in sparse format."""

    ll = dia_matrix(lump)
    dd = 1.0/ll.diagonal()
    siz = len(dd)

    return dia_matrix((dd.reshape((1,-1)),np.array([0])),shape=(siz,siz))

def put_zero_row_in_csr(A, i):
    """Set to zero all entries of row `i` in a CSR sparse matrix."""

    if type(A) == sparse.csr.csr_matrix:
        A.data[A.indptr[i]:A.indptr[i+1]] = 0
    else:
        raise ValueError("The type of the matrix is not csr to put to zero the row")
        

def delete_row_in_coo(A, i):
    """Delete row `i` from a COO sparse matrix and return a new COO matrix."""

    idx_row = A.row==i
    idx_higher_rows = A.row>i

    new_data = np.copy(A.data)
    new_col = np.copy(A.col)
    new_row = np.copy(A.row)

    new_data=np.delete(new_data,idx_row)
    new_col = np.delete(new_col, idx_row)
    new_row[idx_higher_rows] -= 1
    new_row= np.delete(new_row,idx_row)

    return sparse.coo_matrix((new_data,(new_row,new_col)),shape=(A.shape[0]-1,A.shape[1]))