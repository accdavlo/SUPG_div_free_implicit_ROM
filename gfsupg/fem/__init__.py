from .geometry import CartesianGeometry
from .fem1d import FiniteElement1D
from .fem2d import Scipy2DFEM
from .boundary import boundary_index_dir, assemble_1D_sparse_matrix, Dirichlet_BC_set
from ..sparse_utils import (
    get_stencil_indexes, invert_lumped_matrix,
    put_zero_row_in_csr, delete_row_in_coo,
)