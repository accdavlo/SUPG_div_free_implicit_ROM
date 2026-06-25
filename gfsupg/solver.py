"""Core finite-element operators and DeC time integration for 2D acoustics.

This module provides:
- Cartesian geometry and 1D/2D FEM operator assembly.
- Discrete differential operators for divergence/curl-aware formulations.
- DeC space-time solver with SUPG/OSS stabilization variants.
- Utility routines for sparse-matrix boundary handling.

The implementation focuses on structured Cartesian meshes and high-order
Lagrange bases, with both standard and global-flux (GF) formulations.
"""

import numpy as np
from scipy.sparse import coo_matrix,csr_matrix, lil_matrix, dia_matrix
import scipy.sparse as sp
import time
import pickle, os


from scipy.sparse import hstack, vstack, identity
import scipy.sparse as sparse

from scipy.optimize import LinearConstraint, minimize

from .quadr import lagrange_basis, lagrange_basis_deriv, nodes_weights
import matplotlib.pyplot as plt


class CartesianGeometry:
    """
    Defines an hexaedral geometry with a cartesian grid
    xL are all coordinates of left faces and xR are coordinates of right faces, so that the domain is \prod_k [xL[k],xR[k]]
    N_elem_dir are the number of element for each 
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
            self.BC = np.zeros(2*self.dim, dtype=np.int32)   # periodic BC
        else:
            self.BC = BC
        for k in range(self.dim):
            if self.BC[k]==0:                               # Periodic BC 
                assert self.BC[k]==self.BC[k+self.dim]      # Check that is periodic also on the other side
                self.xx[k] = self.xx[k][:-1]


class FiniteElement1D:
    """
    Defines the one dimensional Finite ELement matrices
    Inputs:
    degree of polynomials
    nodes_type among equispaced, gaussLobatto
    quadrature_type among equispaced, gaussLobatto, gaussLegendre
    Class elements:
    mass     matrix M_ij  = int phi_i phi_j
    deriv_i  matrix D_ij  = int phi_i' phi_j
    deriv_j  matrix D_ij  = int phi_i phi_j'
    deriv_ij matrix D_ij  = int phi_i' phi_j'
    """
    def __init__(self, degree, nodes_type="gaussLobatto", quad_type = "gaussLegendre"):
        self.degree = degree
        self.N_dof  = self.degree + 1

        self.nodes,      self.weights      = nodes_weights(self.N_dof, nodes_type)
        self.quad_nodes, self.quad_weights = nodes_weights(self.N_dof, quad_type)
        self.assemble_matrices()
        self.assemble_stencil_matrices()

    def assemble_matrices(self):
        self.matrix_names = ("mass","lump_mass","eval_mat","int_mat", "mass_bar",
                             "deriv_i","deriv_j","deriv_ij",
                             "deriv_ij_tilde","der_int_tilde", \
                             "deriv_i_tilde","mass_int",\
                             "deriv_j_bar")

        self.matrix    = {}
        for matrix_name in self.matrix_names:
            self.matrix[matrix_name] = np.zeros((self.N_dof, self.N_dof))


        self.phi_quad = np.zeros((self.N_dof, len(self.quad_nodes)))
        self.phi_der_quad = np.zeros((self.N_dof, len(self.quad_nodes)))
        for i in range(self.N_dof):
            self.phi_quad[i,:]     = lagrange_basis(self.nodes, self.quad_nodes, i)
            self.phi_der_quad[i,:] = lagrange_basis_deriv(self.nodes, self.quad_nodes, i)

        phi_quad = self.phi_quad
        phi_der_quad = self.phi_der_quad

        for i in range(self.N_dof):            
            for j in range(self.N_dof):
                for iq in range(len(self.quad_nodes)):
                    wq = self.quad_weights[iq]
                    self.matrix["mass"][i,j]    += wq * phi_quad[i,iq]    * phi_quad[j,iq]
                    self.matrix["deriv_i"][i,j] += wq * phi_der_quad[i,iq]* phi_quad[j,iq]
                    self.matrix["deriv_j"][i,j] += wq * phi_quad[i,iq]    * phi_der_quad[j,iq]
                    self.matrix["deriv_ij"][i,j]+= wq * phi_der_quad[i,iq]* phi_der_quad[j,iq]
            self.matrix["lump_mass"][i,i] = np.sum(self.matrix["mass"][i,:])

        for i in range(self.N_dof):            
            nodes_i   = self.nodes[i]*self.quad_nodes
            weights_i = self.nodes[i]*self.quad_weights
            for j in range(self.N_dof):
                basis_quad = lagrange_basis(self.nodes, nodes_i, j)
                for iq in range(len(self.quad_nodes)):
                    self.matrix["int_mat"][i,j] +=  weights_i[iq]*basis_quad[iq]
        
        for j in range(self.N_dof):
            basis_nodes = lagrange_basis(self.nodes, self.nodes, j)
            self.matrix["eval_mat"][:,j] =  basis_nodes
        
        self.matrix["deriv_ij_tilde"] = self.matrix["deriv_ij"]@self.matrix["eval_mat"]
        self.matrix["der_int_tilde"]  = self.matrix["deriv_j"] @self.matrix["int_mat"]
        self.matrix["deriv_i_tilde"]  = self.matrix["deriv_ij"]@self.matrix["int_mat"]
        self.matrix["deriv_j_bar"]    = self.matrix["deriv_j"] @self.matrix["eval_mat"]
        self.matrix["mass_bar"]       = self.matrix["mass"] @self.matrix["eval_mat"]
        self.matrix["mass_int"]       = self.matrix["mass"] @self.matrix["int_mat"]


    def assemble_stencil_matrices(self):
        """
        Dof shared between cells belong to the right cell
                cell k         cell k+1
         |-----------------|-----------------|
        dof0  dof1  dof2  dof0  dof1  dof2  dof0
        Hence, dof0 of cell k communicates up to cell k-1 and cell k+1
        Assuming connectivity of only two cells to right and one to the left
        """
        self.stencil_cells_length = 3

        self.stencil_long = {}
        self.stencil      = {}
        for matrix_name in self.matrix_names:
            self.stencil_long[matrix_name] = np.zeros((self.degree, self.stencil_cells_length, self.degree))
            self.stencil[matrix_name] = np.zeros((self.degree, self.stencil_cells_length*self.degree))

       
        # Contributions from the same cell
        for matrix_name in self.matrix_names:
            for i in range(self.degree):
                self.stencil_long[matrix_name][i,0,:] = \
                    self.matrix[matrix_name][i,:self.degree]    

                #  Contributions from the cell on the right
                self.stencil_long[matrix_name][i,1,0] = \
                    self.matrix[matrix_name][i,self.degree]    
            
            # Add contributions from cell k-1
            i=0
            for j in range(self.degree):
                self.stencil_long[matrix_name][i,-1,j] +=\
                    self.matrix[matrix_name][self.degree,j]

            # Add contribution of last dof of cell k-1
            i=0
            self.stencil_long[matrix_name][i,0,0] += self.matrix[matrix_name][self.degree,self.degree]

            # Putting stencil on one line

            self.stencil[matrix_name][:,:self.degree]                =  self.stencil_long[matrix_name][:,-1,:]
            self.stencil[matrix_name][:,self.degree:2*self.degree]   =  self.stencil_long[matrix_name][:,0,:]
            self.stencil[matrix_name][:,2*self.degree:3*self.degree] =  self.stencil_long[matrix_name][:,1,:]


class Scipy2DFEM:
    """Assemble and store 2D FEM operators on a Cartesian tensor-product mesh.

    Parameters
    ----------
    geom:
        Cartesian geometry descriptor.
    FEM1Dx, FEM1Dy:
        1D finite-element definitions in x/y. If `FEM1Dy` is `None`, the same
        discretization as x is used in y.
    folder:
        Optional folder used for reading/writing serialized operators.
    force_matrix_assembly:
        If `True`, forces assembly even when serialized operators exist.
    save_operators:
        If `True`, assembled operators are serialized to disk.
    """

    def __init__(self, geom, FEM1Dx, FEM1Dy=None, folder = None, force_matrix_assembly = False, save_operators = False):
        
        self.save_operators = save_operators
        if self.save_operators and folder is None:
            self.save_operators = False
        
        if not self.save_operators:
            self.force_matrix_assembly = True

        if folder is not None:
            save_operators_path = geom.geometry_folder+'FEM2D_operators_ord%d_N%04d.pkl'%(FEM1Dx.degree+1,geom.N_elem_dir[0])

        self.FEM1Dx = FEM1Dx
        if FEM1Dy is None:
            self.FEM1Dy = FEM1Dx
        else:
            self.FEM1Dy = FEM1Dy
        self.folder = folder

        self.geom = geom
        self.xx_dofs = dict()
        self.n_dof_dir = dict()
        for k in range(geom.dim):
            if k==0:
                FEM1D = self.FEM1Dx
            elif k==1:
                FEM1D = self.FEM1Dy
            self.xx_dofs[k]   = np.reshape([x_cell + geom.dx[k]*FEM1D.nodes[:-1] for x_cell in geom.xx[k]],-1)
            if self.geom.BC[0]!=0: # non periodic case
                if FEM1D.degree>1:
                    self.xx_dofs[k] = np.delete(self.xx_dofs[k], np.s_[-FEM1D.degree+1:])
            self.n_dof_dir[k] = len(self.xx_dofs[k] ) 
        
        self.rectify_mesh()
        self.vect_matr_maps()

        # Save also in matrix form the coordinates
        self.xx_mat = dict()
        for k in range(self.geom.dim):
            self.xx_mat[k] = self.vect_to_mat(self.mesh_points[:,k])

        if self.folder is not None and os.path.exists(save_operators_path) and not force_matrix_assembly:
            self.define_matrices()
            print("Loading matrices")
            with open(save_operators_path, 'rb') as inp:
                self.operator = pickle.load(inp)
            
            computed_new_matrix = False
            tic = time.time()
            for matrix in self.matrices_definition.keys():
                if matrix not in self.operator.keys():
                    computed_new_matrix = True
                    print("Assembling Matrix %s"%(matrix), end="\r")
                    self.operator[matrix] = self.build_matrix_kron(\
                self.FEM1Dx.stencil[self.matrices_definition[matrix]["matrix_x"]],\
                self.FEM1Dy.stencil[self.matrices_definition[matrix]["matrix_y"]])*\
                self.matrices_definition[matrix]["coefficient"]
            toc = time.time()-tic
            
            
            if computed_new_matrix:
                print("Assembled new matrices in %1.3f seconds"%(toc), end="\r")
                if self.save_operators:
                    with open(save_operators_path, 'wb') as outp:
                        pickle.dump(self.operator, outp, pickle.HIGHEST_PROTOCOL)
            return
        else:
            print("Assembling matrices", end = "\r")
            tic = time.time()
            self.build_matrices_kron()
            toc = time.time()-tic
            print("Assembled matrices in %1.3f seconds"%toc)
            if self.folder is not None and self.save_operators:
                with open(save_operators_path, 'wb') as outp:
                    pickle.dump(self.operator, outp, pickle.HIGHEST_PROTOCOL)

    def rectify_mesh(self):
        mesh_points = []
        for xi in self.xx_dofs[0]:
            for yi in self.xx_dofs[1]:
                mesh_points.append([xi,yi])
        self.mesh_points = np.array(mesh_points)
        self.n_dof_tot = len(self.mesh_points)

        mesh_vertexes = []
        for xi in self.geom.xx[0]:
            for yi in self.geom.xx[1]:
                mesh_vertexes.append([xi,yi])
        self.mesh_vertexes = np.array(mesh_vertexes)
        self.n_vertex_tot = len(self.mesh_vertexes)

    def vect_matr_maps(self):

        self.vect_2_mat = np.zeros((self.n_dof_tot, 2),dtype=np.int64)
        self.mat_2_vect = np.zeros((self.n_dof_dir[0],self.n_dof_dir[1]), dtype=np.int64)
        global_idx = -1
        for ix, xi in enumerate(self.xx_dofs[0]):
            for iy, yi in enumerate(self.xx_dofs[1]):
                global_idx +=1
                self.vect_2_mat[global_idx,:] = [ix,iy]
                self.mat_2_vect[ix,iy]        = global_idx

        self.vect_2_mat_vertex = np.zeros((self.n_vertex_tot, 2),dtype=np.int64)
        self.mat_2_vect_vertex = np.zeros((self.geom.N_elem_dir[0]+1,self.geom.N_elem_dir[1]+1), dtype=np.int64)
        global_idx = -1
        for ix, xi in enumerate(self.geom.xx[0]):
            for iy, yi in enumerate(self.geom.xx[1]):
                global_idx +=1
                self.vect_2_mat_vertex[global_idx,:] = [ix,iy]
                self.mat_2_vect_vertex[ix,iy]        = global_idx

        self.dirichlet_indexes = dict()
        self.dirichlet_indexes["north"] = self.mat_2_vect[-1,:]
        self.dirichlet_indexes["south"] = self.mat_2_vect[0,:]
        self.dirichlet_indexes["east"]  = self.mat_2_vect[:,-1]
        self.dirichlet_indexes["west"]  = self.mat_2_vect[:,0]
        self.dirichlet_indexes["all"]   = np.array(list(set(np.concatenate([\
                                             self.dirichlet_indexes["north"],\
                                             self.dirichlet_indexes["east"],\
                                             self.dirichlet_indexes["south"],\
                                             self.dirichlet_indexes["west"]]))))
        for bc_item in self.dirichlet_indexes.keys():   
            self.dirichlet_indexes[bc_item].sort()

    def apply_dirichlet_bc_matrix(self, boundaries, matrix):
        for boundary in boundaries:
            for i in self.dirichlet_indexes[boundary]:
                put_zero_row_in_csr(matrix, i)
                matrix[i,i]=1.0

    def apply_dirichlet_bc_rhs(self, boundaries, rhs, value):
        for boundary in boundaries:
            for i in self.dirichlet_indexes[boundary]:
                rhs[i]=value[i]

    def vect_to_mat(self, u_vec):
        u_mat = np.zeros((self.n_dof_dir[0],self.n_dof_dir[1]))
        for i in range(self.n_dof_tot):
            u_mat[self.vect_2_mat[i,0],self.vect_2_mat[i,1]] =\
                u_vec[i]
        return u_mat
    
    def vect_to_mat_vertex(self, u_vec):
        u_mat = np.zeros((self.geom.N_elem_dir[0]+1,self.geom.N_elem_dir[1]+1))
        for i in range(self.n_vertex_tot):
            u_mat[self.vect_2_mat_vertex[i,0],self.vect_2_mat_vertex[i,1]] =\
                u_vec[i]
        return u_mat

    def mat_to_vect(self, u_mat):
        u_vec = np.zeros(self.n_dof_tot)
        for i in range(self.n_dof_tot):
            u_vec[i] = u_mat[self.vect_2_mat[i,0],self.vect_2_mat[i,1]]
        return u_vec

    def mat_to_vect_vertex(self, u_mat):
        u_vec = np.zeros(self.n_vertex_tot)
        for i in range(self.n_vertex_tot):
            u_vec[i] = u_mat[self.vect_2_mat_vertex[i,0],self.vect_2_mat_vertex[i,1]]
        return u_vec

    def define_matrices(self):
        self.matrices_definition = dict()
        self.matrices_definition["mass"] = {
            "matrix_x"    : "mass",
            "matrix_y"    : "mass",
            "coefficient" : self.geom.dx[0]*self.geom.dx[1]
        }

        self.matrices_definition["lump_mass"] = {
            "matrix_x"    : "lump_mass",
            "matrix_y"    : "lump_mass",
            "coefficient" : self.geom.dx[0]*self.geom.dx[1]
        }
        
        self.matrices_definition["IDx"] = {
            "matrix_x"    : "deriv_j",
            "matrix_y"    : "mass",
            "coefficient" : self.geom.dx[1]
        }
        
        self.matrices_definition["IDy"] = {
            "matrix_x"    : "mass",
            "matrix_y"    : "deriv_j",
            "coefficient" : self.geom.dx[0]
        }
        
        self.matrices_definition["DxI"] = {
            "matrix_x"    : "deriv_i",
            "matrix_y"    : "mass",
            "coefficient" : self.geom.dx[1]
        }
        
        self.matrices_definition["DyI"] = {
            "matrix_x"    : "mass",
            "matrix_y"    : "deriv_i",
            "coefficient" : self.geom.dx[0]
        }
        
        self.matrices_definition["IDxy"] = {
            "matrix_x"    : "deriv_j",
            "matrix_y"    : "deriv_j",
            "coefficient" : 1.0
        }
        
        self.matrices_definition["DxDx"] = {
            "matrix_x"    : "deriv_ij",
            "matrix_y"    : "mass",
            "coefficient" :1.0/self.geom.dx[0]*self.geom.dx[1]
        }
        
        self.matrices_definition["DyDy"] = {
            "matrix_x"    : "mass",
            "matrix_y"    : "deriv_ij",
            "coefficient" : self.geom.dx[0]/self.geom.dx[1]
        }
        
        self.matrices_definition["DxDy"] = {
            "matrix_x"    : "deriv_i",
            "matrix_y"    : "deriv_j",
            "coefficient" : 1.0
        }
                
        self.matrices_definition["DyDx"] = {
            "matrix_x"    : "deriv_j",
            "matrix_y"    : "deriv_i",
            "coefficient" : 1.0
        }
        
        self.matrices_definition["DxDxy"] = {
            "matrix_x"    : "deriv_ij",
            "matrix_y"    : "deriv_j",
            "coefficient" : 1.0/self.geom.dx[0]
        }
        
        self.matrices_definition["DyDxy"] = {
            "matrix_x"    : "deriv_j",
            "matrix_y"    : "deriv_ij",
            "coefficient" : 1.0/self.geom.dx[1]
        }
        

        self.matrices_definition["DxDx_tilde"] = {
            "matrix_x"    : "deriv_ij_tilde",
            "matrix_y"    : "der_int_tilde",
            "coefficient" : 1.0/self.geom.dx[0]*self.geom.dx[1]
        }
        
        self.matrices_definition["DyDy_tilde"] = {
            "matrix_x"    : "der_int_tilde",
            "matrix_y"    : "deriv_ij_tilde",
            "coefficient" : self.geom.dx[0]/self.geom.dx[1]
        }
        
        self.matrices_definition["DxDy_tilde"] = {
            "matrix_x"    : "deriv_i_tilde",
            "matrix_y"    : "deriv_j_bar",
            "coefficient" : 1.0
        }
        
        self.matrices_definition["DyDx_tilde"] = {
            "matrix_x"    : "deriv_j_bar",
            "matrix_y"    : "deriv_i_tilde",
            "coefficient" : 1.0
        }
        
        self.matrices_definition["IDx_tilde"] = {
            "matrix_x"    : "deriv_j_bar",
            "matrix_y"    : "der_int_tilde",
            "coefficient" : self.geom.dx[1]
        }
        
        self.matrices_definition["IDy_tilde"] = {
            "matrix_x"    : "der_int_tilde",
            "matrix_y"    : "deriv_j_bar",
            "coefficient" : self.geom.dx[0]
        }
        
        self.matrices_definition["mass_tilde_x"] = {
            "matrix_x"    : "der_int_tilde",
            "matrix_y"    : "mass_bar",
            "coefficient" : self.geom.dx[0]*self.geom.dx[1]
        }
        
        self.matrices_definition["mass_tilde_y"] = {
            "matrix_x"    : "mass_bar",
            "matrix_y"    : "der_int_tilde",
            "coefficient" : self.geom.dx[0]*self.geom.dx[1]
        }
                
        self.matrices_definition["DxI_tilde"] = {
            "matrix_x"    : "deriv_i_tilde",
            "matrix_y"    : "mass_bar",
            "coefficient" : self.geom.dx[1]
        }
                
        self.matrices_definition["DyI_tilde"] = {
            "matrix_x"    : "mass_bar",
            "matrix_y"    : "deriv_i_tilde",
            "coefficient" : self.geom.dx[0]
        }
                        
        self.matrices_definition["int_y"] = {
            "matrix_x"    : "mass",
            "matrix_y"    : "int_mat",
            "coefficient" : self.geom.dx[0]*self.geom.dx[1]**2
        }

        self.matrices_definition["int_x"] = {
            "matrix_x"    : "int_mat",
            "matrix_y"    : "mass",
            "coefficient" : self.geom.dx[0]**2.*self.geom.dx[1]
        }
                       
        self.matrices_definition["int_y_tilde"] = {
            "matrix_x"    : "der_int_tilde",
            "matrix_y"    : "mass_int",
            "coefficient" : self.geom.dx[0]*self.geom.dx[1]**2
        }

        self.matrices_definition["int_x_tilde"] = {
            "matrix_x"    : "mass_int",
            "matrix_y"    : "der_int_tilde",
            "coefficient" : self.geom.dx[0]**2.*self.geom.dx[1]
        }

        self.matrices_definition["DxM_tilde"] = {
            "matrix_x"    : "deriv_i_tilde",
            "matrix_y"    : "der_int_tilde",
            "coefficient" : self.geom.dx[1]
        }

        self.matrices_definition["DyM_tilde"] = {
            "matrix_x"    : "der_int_tilde",
            "matrix_y"    : "deriv_i_tilde", 
            "coefficient" : self.geom.dx[0]
        }

        self.matrices_definition["mass_tilde"] = {
            "matrix_x"    : "der_int_tilde",
            "matrix_y"    : "der_int_tilde", 
            "coefficient" : self.geom.dx[0]*self.geom.dx[1]
        }

        self.matrices_definition["Dx_int"] = {
            "matrix_x"    : "deriv_i",
            "matrix_y"    : "int_mat", 
            "coefficient" : self.geom.dx[1]**2
        }

        self.matrices_definition["Dy_int"] = {
            "matrix_x"    : "int_mat",
            "matrix_y"    : "deriv_i", 
            "coefficient" : self.geom.dx[0]**2
        }

        self.matrices_definition["Dx_int_tilde"] = {
            "matrix_x"    : "deriv_i_tilde",
            "matrix_y"    : "mass_int", 
            "coefficient" : self.geom.dx[1]**2
        }

        self.matrices_definition["Dy_int_tilde"] = {
            "matrix_x"    : "mass_int",
            "matrix_y"    : "deriv_i_tilde", 
            "coefficient" : self.geom.dx[0]**2
        }

        self.matrices_definition["mass_tilde_tilde"] = {
            "matrix_x"    : "der_int_tilde",
            "matrix_y"    : "der_int_tilde", 
            "coefficient" : self.geom.dx[0]*self.geom.dx[1]
        }


    def build_matrix_kron(self, stencil_x, stencil_y):
        mat_x = self.build_1D_matrix(stencil_x, 0)
        mat_y = self.build_1D_matrix(stencil_y, 1)
        return sp.kron(mat_x, mat_y)

    def build_1D_matrix(self, stencil_dir, direction):
        """1D matrix assembly"""

        matrix = lil_matrix((self.n_dof_dir[direction],self.n_dof_dir[direction]), dtype=np.float64)

        
        if direction==0:
            FEM1D = self.FEM1Dx
        elif direction==1:
            FEM1D = self.FEM1Dy
        index_i, index_j, values = assemble_1D_sparse_matrix(self.xx_dofs[direction],
                                                             FEM1D, 
                                                             stencil_dir, 
                                                             self.geom, 
                                                             self.n_dof_dir[direction], 
                                                             direction )
                        
        for ix in range(len(self.xx_dofs[direction])):
            for jx_stencil in range(3*FEM1D.degree ):
                i = index_i[ix,jx_stencil]
                j = index_j[ix,jx_stencil]
                v = values[ix,jx_stencil]

                matrix[i,j] += v

        matrix = csr_matrix(matrix)
        matrix.eliminate_zeros()
        return matrix
    
    def build_matrices_kron(self):
        self.define_matrices()

        tot_mat = len(self.matrices_definition)
        
        self.operator = dict() 
        for i, matrix in enumerate(self.matrices_definition):
            print("Assembling Matrices  %02d/%d"%(i,tot_mat+1), end="\r")
            self.operator[matrix] = self.build_matrix_kron(\
                self.FEM1Dx.stencil[self.matrices_definition[matrix]["matrix_x"]],\
                self.FEM1Dy.stencil[self.matrices_definition[matrix]["matrix_y"]])*\
                self.matrices_definition[matrix]["coefficient"]

        print("Assembling Matrices  %02d/%d"%(tot_mat,tot_mat+1), end="\r")
        self.operator["inv_lump"]   = invert_lumped_matrix(self.operator["lump_mass"])
        print("Assembling Matrices  %02d/%d"%(tot_mat+1,tot_mat+1), end="\r")
        
    def evaluate_function(self,funct):
        vfunc = np.vectorize(funct)
        return vfunc(self.mesh_points[:,0],self.mesh_points[:,1])
    
    def evaluate_function_vertex(self,funct):
        vfunc = np.vectorize(funct)
        return vfunc(self.mesh_vertexes[:,0],self.mesh_vertexes[:,1])

    def from_vector_to_vertex(self, q_vec):
        q_mat = self.vect_to_mat(q_vec)
        q_ver = self.mat_to_vect_vertex(q_mat[::self.FEM1Dx.degree,::self.FEM1Dy.degree])
        return q_ver
    
    def from_vector_to_vertex_matrix(self, q_vec):
        q_mat = self.vect_to_mat(q_vec)
        return q_mat[::self.FEM1Dx.degree,::self.FEM1Dy.degree]
    
    def compute_discrete_divergence(self, q):
        return (self.operator["IDx_tilde"]@q["u"]+self.operator["IDy_tilde"]@q["v"])/self.geom.dx[0]/self.geom.dx[1]

    def compute_wrong_discrete_divergence(self, q):
        return (self.operator["IDx"]@q["u"]+self.operator["IDy"]@q["v"])/self.geom.dx[0]/self.geom.dx[1]

    def compute_discrete_divergence_residual(self, q, p_source):
        return (self.operator["IDx_tilde"]@q["u"]\
               +self.operator["IDy_tilde"]@q["v"]\
               -self.operator["mass_tilde"]@p_source\
                )/self.geom.dx[0]/self.geom.dx[1]

    def compute_GF_residual(self, q, source, problem="acoustics"):
        res = dict()
        if problem=="acoustics":
            for var in ("p","u","v"):
                res[var] = np.zeros_like(q["u"])
            res["p"] = (self.operator["IDx_tilde"]@q["u"]\
                +self.operator["IDy_tilde"]@q["v"]\
                -self.operator["mass_tilde"]@source["p"]\
                    )/self.geom.dx[0]/self.geom.dx[1]
            res["u"] = (self.operator["IDx"]@q["p"]\
                        -self.operator["mass_tilde_x"]@source["u"]\
                        )/self.geom.dx[0]
            res["v"] = (self.operator["IDy"]@q["p"]\
                        -self.operator["mass_tilde_y"]@source["v"]\
                        )/self.geom.dx[1]
        else:
            raise NotImplementedError("Equation not implemented for GF residuals in Scipy2DFEM")
        return res



    def compute_noGF_residual(self, q, source, problem="acoustics"):
        res = dict()
        if problem=="acoustics":
            for var in ("p","u","v"):
                res[var] = np.zeros_like(q["u"])
            res["p"] = (self.operator["IDx"]@q["u"]\
                +self.operator["IDy"]@q["v"]\
                -self.operator["mass"]@source["p"]\
                    )/self.geom.dx[0]/self.geom.dx[1]
            res["u"] = (self.operator["IDx"]@q["p"]\
                        -self.operator["mass"]@source["u"]\
                        )/self.geom.dx[0]
            res["v"] = (self.operator["IDy"]@q["p"]\
                        -self.operator["mass"]@source["v"]\
                        )/self.geom.dx[1]
        else:
            raise NotImplementedError("Equation not implemented for noGF residuals in Scipy2DFEM")
        return res


    def compute_discrete_curl(self, q):
        return self.operator["inv_lump"]@(self.operator["IDy"]@q["u"]-self.operator["IDx"]@q["v"])

    def compute_discrete_curl_involution(self, q, alpha, dx,dy):
        """Discrete involution with stabilization terms that should be preserved
        consistent with a curl"""
        op = self.operator
        K_u = dx*op["DxDx"]@q["u"]+dy*op["DyDy"]@q["u"]
        K_v = dx*op["DxDx"]@q["v"]+dy*op["DyDy"]@q["v"]
        omega_q = \
             op["IDxy"]@(op["IDy"]@q["u"]) - alpha**2*dy*op["DyDxy"]@K_u\
            -op["IDxy"]@(op["IDx"]@q["v"]) + alpha**2*dx*op["DxDxy"]@K_v\
            +alpha*(-dx*op["IDy"]@(op["DxDxy"]@q["p"]) + dy*op["IDx"]@(op["DyDxy"]@q["p"]))
        
        return self.operator["inv_lump"]@(self.operator["inv_lump"]@omega_q)

    def compute_discrete_curl_involution_all(self, qall, alpha, dx,dy):
        """Computing the curl involution on all solutions (e.g. in time)"""
        if len(qall["u"].shape)==1:
            return self.compute_discrete_curl_involution(qall,alpha,dx,dy)
        else:
            (M, Ndof) = qall["u"].shape
            curl_inv = dict()
            for var in ("u","v","p"):
                curl_inv[var] = np.zeros((M,Ndof))
            for m in range(M):
                q_one = dict()                
                for var in ("u","v","p"):
                    q_one[var] = qall[var][m,:]
                z = self.compute_discrete_curl_involution(q_one,alpha,dx,dy)
                for var in ("u","v","p"):
                    curl_inv[var][m,:] = z[var]
            return curl_inv
                

    def compute_discrete_divergence_norms(self, qs):
        Nt = qs["u"].shape[0]
        div_norms = np.zeros(Nt)
        for it in range(Nt):
            disc_div = (self.operator["IDx_tilde"]@qs["u"][it,:]+self.operator["IDy_tilde"]@qs["v"][it,:])/self.geom.dx[0]/self.geom.dx[1]
            div_norms[it] = np.linalg.norm(disc_div)/np.sqrt(self.n_dof_tot)
        return div_norms
    
    def compute_wrong_discrete_divergence_norms(self, qs):
        Nt = qs["u"].shape[0]
        div_norms = np.zeros(Nt)
        for it in range(Nt):
            disc_div = (self.operator["IDx"]@qs["u"][it,:]+self.operator["IDy"]@qs["v"][it,:])/self.geom.dx[0]/self.geom.dx[1]
            div_norms[it] = np.linalg.norm(disc_div)/np.sqrt(self.n_dof_tot)
        return div_norms
    

    def divfree_projection_optimization(self, IC_vect, problem, method = "trust-constr"):
        div_oper = hstack([self.operator["IDx_tilde"],self.operator["IDy_tilde"]] )
        div_oper = div_oper.tocoo()

        if (problem.BC==np.ones(4,dtype=np.int64)).all():
            dirichlet_BC = True
        else:
            dirichlet_BC = False


        if dirichlet_BC:
            for i in np.sort(self.dirichlet_indexes["all"])[::-1]:
                div_oper = delete_row_in_coo(div_oper,i)

            # Add dirichlet BC constraints
            # U constraint
            u_bc_idxs =self.dirichlet_indexes["all"]##[:-1] 
            n_bc = len(u_bc_idxs)
            data = np.ones(n_bc)
            row = np.arange(n_bc)
            col = u_bc_idxs
            u_matrix_constraint = sparse.coo_matrix((data, (row,col)), shape=(n_bc, 2*self.n_dof_tot))
            u_rhs_constraint = IC_vect["u"][u_bc_idxs]

            # V constraint
            v_bc_idxs =self.dirichlet_indexes["all"]##[:-1] 
            n_bc = len(v_bc_idxs)
            data = np.ones(n_bc)
            row = np.arange(n_bc)
            col = np.array(v_bc_idxs,dtype=np.int64)+self.n_dof_tot
            v_matrix_constraint = sparse.coo_matrix((data, (row,col)), shape=(n_bc, 2*self.n_dof_tot))
            v_rhs_constraint = IC_vect["v"][v_bc_idxs]


        dx_div_oper = hstack([self.operator["DxDx_tilde"],self.operator["DxDy_tilde"]])
        dy_div_oper = hstack([self.operator["DyDx_tilde"],self.operator["DyDy_tilde"]])
        grad_div_oper = vstack([dx_div_oper, dy_div_oper])

        if problem.source is not None:

            source_p = self.evaluate_function(problem.source["p"])
        else:
            source_p = None

        if source_p is None:
            source_p_term = np.zeros(grad_div_oper.shape[0])
        else:
            source_p_term = np.concatenate([self.operator["DxM_tilde"]@source_p ,\
                                    self.operator["DyM_tilde"]@source_p ] )

        mass_oper = vstack([
            hstack([   self.operator["mass"], 0.*self.operator["mass"]]),\
            hstack([0.*self.operator["mass"],    self.operator["mass"]])
        ])



        if source_p is None:
            rhs_constraint_div = np.zeros(self.n_dof_tot)
        else:
            rhs_constraint_div = self.operator["mass_tilde"]@source_p

            
        ## keeping the zero constraints
        #if dirichlet_BC:
        #    rhs_constraint_div[FEM2D.dirichlet_indexes["all"]]=0.
        
        if dirichlet_BC:
            rhs_constraint_div = np.delete(rhs_constraint_div,self.dirichlet_indexes["all"])

            lhs_constraint = vstack([div_oper,u_matrix_constraint,v_matrix_constraint])
            rhs_constraint = np.hstack([rhs_constraint_div,u_rhs_constraint,v_rhs_constraint])
            linear_constraint = LinearConstraint(lhs_constraint,  rhs_constraint, rhs_constraint)
        else:
            linear_constraint = LinearConstraint(div_oper,rhs_constraint_div,rhs_constraint_div)


        u = IC_vect["u"]
        v = IC_vect["v"]

        u_ic = np.concatenate([u,v])
        
        mass_mass = mass_oper.T @ mass_oper
        grad_div_grad_div = grad_div_oper.T@grad_div_oper

        # pen_grad_div = 1e5
        # pen_div = 1e3

        def target_function(u):
            return np.sum((mass_oper@(u-u_ic))**2) #+ pen_div*np.sum((div_oper@u)**2) #+ pen_grad_div*np.sum((grad_div_oper@u-source_p_term)**2)
        def target_function_dir(u):
            return 2.*mass_mass@(u-u_ic) #+ pen_grad_div*2.* grad_div_oper.T@( grad_div_oper@u-source_p_term)
        def target_function_hess(u):
            return 2.*mass_mass #+pen_grad_div*2. * grad_div_grad_div



        # Different methods # trust-constr #SLSQP #"lagrange"#


        u_guess = np.zeros_like(u_ic)
        if method == 'trust-constr':
            res = minimize(target_function, u_guess, method=method,
                        jac=target_function_dir, hess=target_function_hess,
                    constraints=[linear_constraint], tol=1e-10,# constr_violation=1e-10,
                    options={'verbose':3}) #trust-constr #SLSQP #constr_violation
            q_vec = dict()
            q_vec["u"]       = res.x[:len(u)]
            q_vec["v"]       = res.x[len(u):]
            q_vec["p"]       = IC_vect["p"]
        elif method == 'SLSQP':
            my_constraint = dict()
            my_constraint["type"] = 'eq'
            my_constraint["fun"] = lambda x: linear_constraint.A@x-linear_constraint.lb
            #my_constraint["jac"] = lambda x: linear_constraint.A
            res = minimize(target_function, u_ic, method=method,
                        #jac=target_function_dir, hess=target_function_hess,
                    constraints=[my_constraint]) #trust-constr #SLSQP
            q_vec = dict()
            q_vec["u"]       = res.x[:len(u)]
            q_vec["v"]       = res.x[len(u):]
            q_vec["p"]       = IC_vect["p"]
        elif method == 'lagrange':
            N_unkn = linear_constraint.A.shape[1]
            N_constraints = linear_constraint.A.shape[0]
            AAA = hstack([   mass_oper, linear_constraint.A.T])
            BBB =hstack([linear_constraint.A,  sparse.csr_matrix((N_constraints, N_constraints), dtype=np.float64)])
            print(AAA.shape, BBB.shape)
            lagrange_mat = vstack([AAA,BBB])

            lagrange_rhs = np.concatenate([np.zeros(N_unkn), linear_constraint.lb- linear_constraint.A@u_ic])
            x = sparse.linalg.spsolve(lagrange_mat, lagrange_rhs)

            q_vec = dict()
            q_vec["u"]       = x[:len(u)]+u
            q_vec["v"]       = x[len(u):2*len(u)]+v
            q_vec["p"]       = IC_vect["p"]


        if problem.coriolis is not None or problem.coriolis_non_uniform is not None:
            # Well prepare pressure
            print("Preparing also pressure")
            q_vec["p"] = self.well_prepare_p_coriolis(q_vec, problem)
        return q_vec


    def divfree_projection_integration(self, IC_vect, IC_fun, problem):
        FEM1Dx = self.FEM1Dx
        FEM1Dy = self.FEM1Dy
        geom   = self.geom

        u = IC_vect["u"]
        v = IC_vect["v"]

        u_mat = self.vect_to_mat(u)
        v_mat = self.vect_to_mat(v)

        u_mario = np.empty_like(u_mat)
        v_mario = np.empty_like(v_mat)



        if problem.source is not None:
            source_p = self.evaluate_function(lambda x,y: problem.source["p"](x,y,0.))

        # Solve I^y u(x_m) l = int y_0^y_l u^ex(x_m,y) dy

        
        HO_quad_nodes, HO_quad_weights = nodes_weights(10, "gaussLegendre")

        RHS = np.zeros(FEM1Dy.degree)
        for m, x in enumerate(self.xx_dofs[0]):
            u_mario[m, 0] = IC_fun["u"](x, geom.xL[1])

            # Solve I^y u(x_m) l = int y_0^y_l u^ex(x_m,y) dy
            for j_cell in range(geom.N_elem_dir[1]):
                j_cell_global = j_cell*FEM1Dy.degree      
                j1_cell_global = (j_cell+1)*FEM1Dy.degree
                for j_dof in range(1,FEM1Dy.degree+1):
                    j_dof_global = j_cell_global+j_dof 
                    dy = self.xx_dofs[1][j_dof_global] - geom.xx[1][j_cell]
                    quad_nodes = HO_quad_nodes*dy +geom.xx[1][j_cell]
                    quad_weights =  HO_quad_weights*dy
                    RHS[j_dof-1] = np.sum(np.vectorize(IC_fun["u"])(x, quad_nodes) * quad_weights)
                
                dy = geom.xx[1][j_cell+1]-geom.xx[1][j_cell]
                RHS = RHS/dy- FEM1Dy.matrix["int_mat"][1:,0]*u_mario[m, j_cell_global]
                u_mario[ m, j_cell_global+1:j1_cell_global+1 ] = np.linalg.solve(FEM1Dy.matrix["int_mat"][1:,1:],RHS)




        # Solve I^x v(y_l) m = int x_0^x_m v^ex(x,y_l) dx

        RHS = np.zeros(FEM1Dx.degree)
        for l, y in enumerate(self.xx_dofs[1]):
            v_mario[0,l] = IC_fun["v"](geom.xL[0], y )

            # Solve I^x v(y_l) m = int x_0^x_m v^ex(x,y_l) dx
            for j_cell in range(geom.N_elem_dir[0]):
                j_cell_global = j_cell*FEM1Dx.degree      
                j1_cell_global = (j_cell+1)*FEM1Dx.degree
                for j_dof in range(1,FEM1Dx.degree+1):
                    j_dof_global = j_cell_global+j_dof 
                    dx = self.xx_dofs[0][j_dof_global] - geom.xx[0][j_cell]
                    quad_nodes = HO_quad_nodes*dx +geom.xx[0][j_cell]
                    quad_weights =  HO_quad_weights*dx
                    RHS[j_dof-1] = np.sum(np.vectorize(IC_fun["v"])(quad_nodes, y) * quad_weights)
                
                dx = geom.xx[0][j_cell+1]-geom.xx[0][j_cell]
                RHS = RHS/dx- FEM1Dx.matrix["int_mat"][1:,0]*v_mario[ j_cell_global, l]
                v_mario[ j_cell_global+1:j1_cell_global+1, l ] = np.linalg.solve(\
                    FEM1Dx.matrix["int_mat"][1:,1:],RHS)


        q_vec = dict()
        q_vec["u"]       = self.mat_to_vect(u_mario)
        q_vec["v"]       = self.mat_to_vect(v_mario)
        q_vec["p"]       = IC_vect["p"]
        if problem.coriolis is not None or problem.coriolis_non_uniform is not None:
            # Well prepare pressure
            print("Preparing also pressure")
            q_vec["p"] = self.well_prepare_p_coriolis(q_vec, problem)

        return q_vec
    
    def compute_sources(self,q_vec, problem, time=0.):
        source = dict()
        if problem.source is not None:
            for var in problem.vars:
                source[var] = self.evaluate_function(lambda x,y: problem.source[var](x,y,time))
        else:
            for var in problem.vars:
                source[var] = self.evaluate_function(lambda x,y: 0.)


        if problem.coriolis_non_uniform is not None:
            cor_nu = self.evaluate_function(problem.coriolis_non_uniform)
        else:
            cor_nu = self.evaluate_function(lambda x,y: 0.)

        coriolis = problem.coriolis

        source["u"] = q_vec["v"]*(coriolis+cor_nu) - problem.friction*q_vec["u"]+source["u"]
        source["v"] = -q_vec["u"]*(coriolis+cor_nu)- problem.friction*q_vec["v"]+source["v"]
        
        return source
    
    def well_prepare_p_coriolis(self, q_vec, problem):
        """For coriolis systems, the pressure at equilibrium balances the coriolis source
        In particular \partial_x p  = c v and \partial_y p = -c u
        So p = \int^x c v dx + const(y) and p = -\int^y c u dy + const(x)
        Choosing any average of the two makes it work
        """
        p0 = q_vec["p"][0]
        a = 0.5

        p_mat = self.vect_to_mat(q_vec["p"])

        p_new = np.zeros(p_mat.shape)
        Ixv   = np.zeros(p_mat.shape)
        Iyu   = np.zeros(p_mat.shape)


        source = self.compute_sources(q_vec,problem)

        source_u = self.vect_to_mat(source["u"])
        source_v = self.vect_to_mat(source["v"])

        # Computing I^x S_u
        for l, y in enumerate(self.xx_dofs[1]):
        # Compute I^x S_u(y_l) m = int x_0^x_m S_u(x,y_l) dx
            for j_cell in range(self.geom.N_elem_dir[0]):
                j_cell_global = j_cell*self.FEM1Dx.degree      
                j1_cell_global = (j_cell+1)*self.FEM1Dx.degree

                if self.geom.BC[0] == 0 and (j_cell==self.geom.N_elem_dir[0]-1):  # periodic
                    if self.FEM1Dx.degree>1:
                        dx = self.geom.xR[0]-self.geom.xx[0][j_cell]
                        Ixv[ j_cell_global+1:j1_cell_global, l ] =Ixv[ j_cell_global,l]+\
                            dx*self.FEM1Dx.matrix["int_mat"][1:-1,:]@ \
                            np.concatenate([source_u[ j_cell_global:j1_cell_global, l], [source_u[0,l]] ])
                else:
                    dx = self.geom.xx[0][j_cell+1]-self.geom.xx[0][j_cell]
                    Ixv[ j_cell_global+1:j1_cell_global+1, l ] =Ixv[ j_cell_global,l]+\
                        dx*self.FEM1Dx.matrix["int_mat"][1:,:]@ source_u[ j_cell_global:j1_cell_global+1, l]
                

            

        # Computing I^y S_v
        for l, x in enumerate(self.xx_dofs[0]):
            # Compute I^y S_v(x_l) m = int y_0^y_m S_v(x_l,y) dy
            for j_cell in range(self.geom.N_elem_dir[1]):
                j_cell_global = j_cell*self.FEM1Dy.degree      
                j1_cell_global = (j_cell+1)*self.FEM1Dy.degree

                if self.geom.BC[1] == 0 and (j_cell==self.geom.N_elem_dir[1]-1):  # periodic in y
                    if self.FEM1Dy.degree>1:
                        dy = self.geom.xR[1]-self.geom.xx[1][j_cell]
                        Iyu[ l, j_cell_global+1:j1_cell_global ] =Iyu[ l,j_cell_global]+\
                            dy*self.FEM1Dy.matrix["int_mat"][1:-1,:]@ \
                            np.concatenate( [source_v[ l, j_cell_global:j1_cell_global], [source_v[l,0]] ] )
                else:
                    dy = self.geom.xx[1][j_cell+1]-self.geom.xx[1][j_cell]
                    Iyu[ l, j_cell_global+1:j1_cell_global+1 ] =Iyu[ l,j_cell_global]+\
                        dy*self.FEM1Dy.matrix["int_mat"][1:,:]@ source_v[ l, j_cell_global:j1_cell_global+1]
            

        # optimize a 
        # min_a || p0 + a Ixv+ (1-a) Iyu -p||^2=||a (Ixv-Iyu)+ p0 + Iyu -p||^2
        #  2 (Ixv-Iyu) * [a(Ixv-Iyu) + p0+Iyu-p |=0
        # a =  -(Ixv-Iyu)*(p0+Iyu-p)/ ((Ixv-Iyu)*(Ixv-Iyu))

        # a = - np.sum((Ixv-Iyu)*(p0+Iyu-p_mat))/(np.sum((Ixv-Iyu)*(Ixv-Iyu)))
        # p_new[:,:] = p0 + a*Ixv + (1-a)*Iyu
        # p_new[:,:] = p0 + Ixv + Iyu
        
        p_aux_y = np.zeros_like(Ixv)
        for i in range(len(self.xx_dofs[0])):
            p_aux_y[i,:] = p_mat[0,:]-p0-Ixv[0,:]

        p_aux_x = np.zeros_like(Ixv)
        for j in range(len(self.xx_dofs[1])):
            p_aux_x[:,j] = p_mat[:,0]-p0-Iyu[:,0]


        # p_aux_y = np.zeros_like(Ixv)
        # for i in range(len(self.xx_dofs[0])):
        #     p_aux_y[i,:] = Iyu[0,:]-Ixv[0,:]

        # p_aux_x = np.zeros_like(Ixv)
        # for j in range(len(self.xx_dofs[1])):
        #     p_aux_x[:,j] = Ixv[:,0]-Iyu[:,0]


        p_new = p0 + a*(Ixv + p_aux_y)+(1-a)*(Iyu+p_aux_x)

        p_vec = self.mat_to_vect(p_new)
        
        return p_vec


def boundary_index_dir(ix, direction, geom, n_dof_dir):
    """ Actual implementation of boundary conditions for matrices: we look for the appropriate index periodic/neumann"""
    if geom.BC[direction]==0: #periodic
        return ix%n_dof_dir
    else: #neumann (and dirichlet doesn't really matter)
        return max(min(ix,n_dof_dir-1),0)


def assemble_1D_sparse_matrix(xx_dofs_dir, FEM1D_dir, stencil_dir, geom, n_dof_dir, direction):
    """Assemble of the sparse matrix in form of 4-tensor index_i, index_j, values such that matrix[index_i[ix,iy,jx_stencil,jy_stencil], index_j[ix,iy,jx_stencil,jy_stencil]]+=values[ix,iy,jx_stencil,jy_stencil]
    where ix, iy are the dof indexes of the rows and
    jx_stencil and jy_stencil are the dofs of the columns (only in the stencil of the matrix)
    """
    Na, Nb  = len(xx_dofs_dir),  3*FEM1D_dir.degree

    index_i = np.zeros((Na,Nb), dtype=np.int64)
    index_j = np.zeros((Na,Nb), dtype=np.int64)
    values  = np.zeros((Na,Nb))

    for ix, xi in enumerate(xx_dofs_dir):
        ix_cell = ix//FEM1D_dir.degree
        ix_dof  = ix %FEM1D_dir.degree
            

        index_i[ix,:] = ix

        # 1D formalism
        jxl = (ix_cell-1)*FEM1D_dir.degree 
        jxr = (ix_cell+2)*FEM1D_dir.degree 

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


class DeC:
    """Deferred Correction (DeC) temporal quadrature coefficients.

    Attributes
    ----------
    theta:
        Integration coefficients mapping sub-step residuals to each node.
    beta:
        Sub-node positions in the normalized time interval [0, 1].
    """

    def __init__(self, M_sub, n_iter, nodes_type):
        self.n_subNodes = M_sub+1
        self.M_sub = M_sub
        self.n_iter = n_iter
        self.nodes_type = nodes_type
        self.compute_theta_DeC()
        self.name = "DeC_"+self.nodes_type
    
    def compute_theta_DeC(self):
        nodes, w = nodes_weights(self.n_subNodes,self.nodes_type)
        int_nodes, int_w = nodes_weights(self.n_subNodes,"gaussLobatto")
        # generate theta coefficients 
        self.theta = np.zeros((self.n_subNodes,self.n_subNodes))
        self.beta = np.zeros(self.n_subNodes)
        for m in range(self.n_subNodes):
            self.beta[m] = nodes[m]
            nodes_m = int_nodes*(nodes[m])
            w_m = int_w*(nodes[m])
            for r in range(self.n_subNodes):
                self.theta[m,r] = np.sum(lagrange_basis(nodes,nodes_m,r)*w_m)
        return self.theta, self.beta
    
  
class DeCSpaceTimeSUPGSolver:
    """Space-time DeC solver for 2D acoustics with SUPG/OSS stabilization.

    The solver advances the state `(u, v, p)` on a fixed structured mesh using
    high-order finite elements in space and DeC iterations in time.
    """

    def __init__(self, problem, FEM2D, DeC, GF=False, stab = "SUPG", trick_second_der = False):
        self.FEM2D   = FEM2D
        self.geom    = self.FEM2D.geom 
        self.GF      = GF
        self.DeC     = DeC
        self.problem = problem
        self.set_ic()
        self.CFL     = 0.1
        if self.FEM2D.FEM1Dx.degree>=5:
            self.set_CFL( 1./2./(2.*self.FEM2D.FEM1Dx.degree+1))
        self.Nt_save = 35
        self.Nt_max  = 1000000
        self.stab    = stab
        self.trick_second_der = trick_second_der
        self.set_second_derivative_operators()

    def set_second_derivative_operators(self):
        """If trick second der is used, then instead of the second derivative operator
        we set the composition of the first derivative operators with 
        the inverse of the lumped mass matrix to match the kernels of the central part"""
        op = self.FEM2D.operator
        if self.trick_second_der:
            op["DxDx2"] = op["DxI"]@op["inv_lump"]@op["IDx"]
            op["DyDy2"] = op["DyI"]@op["inv_lump"]@op["IDy"]
            op["DxDx2_tilde"] = \
                op["DxI"]@\
                op["inv_lump"]@\
                op["mass_tilde_y"]@\
                op["inv_lump"]@\
                op["IDx"]
            op["DyDy2_tilde"] = \
                op["DyI"]@\
                op["inv_lump"]@\
                op["mass_tilde_x"]@\
                op["inv_lump"]@\
                op["IDy"]
        else:
            op["DxDx2"] = op["DxDx"]
            op["DyDy2"] = op["DyDy"]
            op["DxDx2_tilde"] = op["DxDx_tilde"]
            op["DyDy2_tilde"] = op["DyDy_tilde"]

        op["DxDx3"] = op["DxI"]@op["inv_lump"]@op["IDx"]
        op["DyDy3"] = op["DyI"]@op["inv_lump"]@op["IDy"]


        op["DxDx3_tilde"] = \
            op["DxI"]@\
            op["inv_lump"]@\
            op["mass_tilde_y"]@\
            op["inv_lump"]@\
            op["IDx"]
        op["DyDy3_tilde"] = \
            op["DyI"]@\
            op["inv_lump"]@\
            op["mass_tilde_x"]@\
            op["inv_lump"]@\
            op["IDy"]
        
        op["ZxMy"] =   op["DxDx"]-op["DxDx3"]
        op["ZyMx"] =   op["DyDy"]-op["DyDy3"]
        
        op["ZxMy_tilde"] =   op["DxDx_tilde"]-op["DxDx3_tilde"]
        op["ZyMx_tilde"] =   op["DyDy_tilde"]-op["DyDy3_tilde"]
        
        op["DxZy_int"] =   op["DyDx_tilde"]-\
                                            op["DyI"]@\
                                            op["inv_lump"]@\
                                            op["IDx_tilde"]
                                            
        op["DyZx_int"] =   op["DxDy_tilde"]-\
                                            op["DxI"]@\
                                            op["inv_lump"]@\
                                            op["IDy_tilde"]

        op["My_tilde_Zx_int"] =  op["DxM_tilde"] - \
                       op["DxI"]@op["inv_lump"]@op["mass_tilde_tilde"] 

        
        op["Mx_tilde_Zy_int"] =  op["DyM_tilde"]- \
                       op["DyI"]@op["inv_lump"]@op["mass_tilde_tilde"]    

        op["Zx_int_My"] = op["DxI_tilde"]\
                       -op["DxI"]@op["inv_lump"]@op["mass_tilde_x"]
        op["Zy_int_Mx"] = op["DyI_tilde"]\
                       -op["DyI"]@op["inv_lump"]@op["mass_tilde_y"]                       

    def set_ic(self):
        if hasattr(self.problem,"perturbation") and hasattr(self.problem,"steady_state_test"):
            if "num" in self.problem.name:
                # Load numerical solution at the steady state
                base_folder=self.problem.steady_state_test.folderName
                order = self.FEM2D.FEM1Dx.degree+1
                N     = self.FEM2D.geom.N_elem_dir[0]
                if self.GF:
                    filename = "/final_sol_SUPG_GF_ord_%d_N_%04d.pkl"%(order,N)
                else:
                    filename = "/final_sol_SUPG_ord_%d_N_%04d.pkl"%(order,N)
                if os.path.isfile(base_folder+filename):
                    with open(base_folder+filename, 'rb') as handle:
                        final_sol = pickle.load(handle)
                else:
                    raise ValueError("File not found %s\n You should first run the numerical long time simulation"%(base_folder+filename))
                self.ic_vect = final_sol[0]

                if self.problem.perturbation:
                    disc_divergence = np.mean(np.abs(self.FEM2D.compute_discrete_divergence(self.ic_vect)))
                    print("Initial discrete divergence mean %g"%disc_divergence)

                self.ic_no_pert = dict()
                for var in self.problem.vars:
                    self.ic_no_pert[var] = np.copy(self.ic_vect[var])
                    self.ic_vect[var] += self.FEM2D.evaluate_function(\
                                        self.problem.perturbation[var])
            elif "opt" in self.problem.name:
                # Compute the equilibrium
                problem_base = self.problem.steady_state_test 
                ic_vect_guess = dict()
                for var in self.problem.vars:
                    ic_vect_guess[var] = self.FEM2D.evaluate_function(\
                                        problem_base.ics[var])

                self.ic_no_pert = self.FEM2D.divfree_projection_optimization( ic_vect_guess, problem_base)
                
                self.ic_vect = dict()
                for var in self.problem.vars:
                    self.ic_vect[var] = np.copy(self.ic_no_pert[var])
                    self.ic_vect[var] += self.FEM2D.evaluate_function(\
                                        self.problem.perturbation[var])
    
            elif "int" in self.problem.name:
                # Compute the equilibrium
                problem_base = self.problem.steady_state_test
                ic_vect_guess = dict()
                for var in self.problem.vars:
                    ic_vect_guess[var] = self.FEM2D.evaluate_function(\
                                        problem_base.ics[var])
                self.ic_no_pert = self.FEM2D.divfree_projection_integration( ic_vect_guess, problem_base.ics, coriolis= self.problem.coriolis)

                self.ic_vect = dict()
                for var in self.problem.vars:
                    self.ic_vect[var] = np.copy(self.ic_no_pert[var])
                    self.ic_vect[var] += self.FEM2D.evaluate_function(\
                                        self.problem.perturbation[var])

        else:
            self.ic_vect = dict()
            for var in self.problem.vars:
                self.ic_vect[var] = self.FEM2D.evaluate_function(\
                                    self.problem.ics[var])

            if hasattr(self.problem,"perturbation"):
                disc_divergence = np.mean(np.abs(self.FEM2D.compute_discrete_divergence(self.ic_vect)))
                print("Initial discrete divergence mean %g"%disc_divergence)
            
                self.ic_no_pert = dict()
                for var in self.problem.vars:
                    self.ic_no_pert[var] = np.copy(self.ic_vect[var])
                    self.ic_no_pert[var] -= self.FEM2D.evaluate_function(\
                                        self.problem.perturbation[var])


    def set_CFL(self, CFL):
        self.CFL = CFL
        print("CFL number = %g"%self.CFL)
        
    def set_save_slabs(self, Nt_save):
        self.Nt_save = max(Nt_save,3)
    def set_Nt_max(self, Nt_max):
        self.Nt_max = Nt_max

    def solver_set_parameters(self, stab_coeff = None, with_error = False, \
              with_error_vertex = False, GF=None, CFL = None, \
              stab = None, trick_second_der = False) :
        if CFL is not None:
            self.set_CFL(CFL)
        if GF is not None:
            self.GF = GF

        if with_error:
            error = np.zeros(len(self.problem.vars))
        else:
            error = None
        if with_error_vertex:
            error_vertex = np.zeros(len(self.problem.vars))
        else:
            error_vertex = None

        if stab is not None:
            self.stab = stab

        if self.GF:
            method_name = self.stab+"_GF"
            error_name = "errors_"+self.stab+"_GF"
        else:
            method_name = self.stab
            error_name = "errors_"+self.stab

        if self.problem.equations=="acoustics":
            if self.GF:
                get_residual = define_GF_residuals
                if self.stab == "SUPG":
                    get_stabilization = SUPG_GF_stabilization
                elif self.stab =="OSS":
                    get_stabilization = OSS_GF_stabilization
                curl_stabilization = OSS_GF_curl_stabilization
            else:
                get_residual = define_residuals
                if self.stab == "SUPG":
                    get_stabilization = SUPG_stabilization
                elif self.stab =="OSS":
                    get_stabilization = OSS_stabilization
                curl_stabilization = OSS_curl_stabilization
        else:
            raise NotImplementedError("Equations %s not implemented in solve in DeCSpaceTimeSolver"%self.problem.equations)

        if stab_coeff is None:
            if self.stab == "SUPG":
                if self.FEM2D.FEM1Dx.degree <= 5:
                    al = 0.05  # 5*10**-self.FEM2D.FEM1Dx.degree
                else:
                    al = 0.02
            elif self.stab == "OSS":
                if self.FEM2D.FEM1Dx.degree <= 2:
                    al = 1e-2  # 5*10**-self.FEM2D.FEM1Dx.degree
                else:
                    al = 4e-2
        else:
            al = stab_coeff
        self.stab_coeff = al
        self.stab_curl_coeff = 1e-4

        if trick_second_der!=self.trick_second_der:
            self.trick_second_der = trick_second_der
            self.set_second_derivative_operators()

        return error, error_vertex, method_name, error_name, get_residual, get_stabilization, curl_stabilization


    def solve(self, stab_coeff = None, with_error = False, \
              with_error_vertex = False, GF=None, CFL = None, \
              save_sol = False, stab = None, trick_second_der = False, curl_stab_flag = False):
        """Run a full transient simulation.

        Parameters
        ----------
        stab_coeff:
            Optional stabilization coefficient. If `None`, defaults are chosen
            from polynomial degree and stabilization type.
        with_error, with_error_vertex:
            If enabled and an exact solution is available, compute errors 
            varying in time averaging on all dofs or on vertex dofs.
        GF:
            Optional override for global-flux formulation flag.
        CFL:
            Optional override for time-step scaling.
        save_sol:
            If truthy, writes final state and diagnostics to a pickle file.
        stab:
            Optional override for stabilization method (`SUPG` or `OSS`).
        trick_second_der:
            If changed, uses other second-derivative-related operators.
        curl_stab_flag:
            Adds optional curl-based OSS stabilization for velocity components.

        Returns
        -------
        q_save, tt_save, comp_time, error, error_vertex
            Saved solution snapshots, saved times, wall time, and optional
            error arrays.
        """

        error, error_vertex, method_name, error_name, get_residual, get_stabilization, curl_stabilization = \
                self.solver_set_parameters(stab_coeff, with_error, with_error_vertex, \
                                           GF, CFL, stab, trick_second_der)
        

        dt_save = self.problem.T_fin/(self.Nt_save-2)

        q_save = dict() 

        it = 0
        t=0.
        t_save  = 0.
        it_save = 0
        L2 = dict()
        for var in self.problem.vars: # ("u", "v", "p")
            L2[var] = np.zeros(self.FEM2D.n_dof_tot)
            q_save[var] = np.zeros((self.Nt_save, self.FEM2D.n_dof_tot))
        tt_save = np.zeros(self.Nt_save)

        q_prev = dict()
        q_now  = dict()
        q      = np.zeros((len(self.problem.vars), self.FEM2D.n_dof_tot)) 
        for var in self.problem.vars:
            q_prev[var] = np.zeros((self.DeC.n_subNodes, self.FEM2D.n_dof_tot))
            q_now[var]  = np.zeros((self.DeC.n_subNodes, self.FEM2D.n_dof_tot))
            for i in range(self.DeC.n_subNodes):
                q_now[var][i,:] = self.ic_vect[var]

        for var in self.problem.vars:
            q_save[var][it_save,:] = q_now[var][-1,:]
        tt_save[it_save] = t

        source = dict()
        if self.problem.source is not None:
            for var in self.problem.vars:
                source[var] = self.FEM2D.evaluate_function(lambda x,y: self.problem.source[var](x,y,0.))
        else:
            for var in self.problem.vars:
                source[var] = self.FEM2D.evaluate_function(lambda x,y: 0.)

        if self.problem.coriolis_non_uniform is not None:
            cor_nu = self.FEM2D.evaluate_function(self.problem.coriolis_non_uniform)
        else:
            cor_nu = self.FEM2D.evaluate_function(lambda x,y: 0.)

        if self.problem.dirichlet is not None:
            dirichlet_BC = dict()
            for bc_item in self.problem.dirichlet.keys():
                BC_values = dict()
                idxs = self.FEM2D.dirichlet_indexes[bc_item]
                for var in self.problem.dirichlet[bc_item]:
                    BC_values[var] = self.ic_vect[var][idxs]
                dirichlet_BC[bc_item] = Dirichlet_BC_set(idxs, BC_values)

        else:
            dirichlet_BC = None

        sub_sources = dict()
        for ivar, var in enumerate(self.problem.vars):
            sub_sources[var] = np.zeros_like(q_now[var])

        #MZ main loop  
        tic = time.time()
        while (t<self.problem.T_fin and it<self.Nt_max):
            # Set dt
            dt =  self.CFL* self.geom.dx_min#self.CFL * self.problem.max_dt(q, self.geom.dx)
            c = self.problem.c 

            # Initialize variables
            for ivar, var in enumerate(self.problem.vars):
                q[ivar,:] = q_now[var][-1,:]
                for i in range(self.DeC.M_sub):
                    q_now[var][i,:] = q_now[var][-1,:] # previous timestep last update
                if self.problem.source is not None:
                    for i in range(self.DeC.M_sub+1):
                        sub_sources[var][i,:] = self.FEM2D.evaluate_function(lambda x,y: self.problem.source[var](x,y,t+dt*self.DeC.beta[i]))

            print("Iteration %07d, time %1.5f, max vars %1.3f  %1.3f  %1.3f ,  min vars %1.3f  %1.3f  %1.3f "%(it,t,\
                    np.max(q[0,:]),np.max(q[1,:]),np.max(q[2,:]),\
                    np.min(q[0,:]),np.min(q[1,:]),np.min(q[2,:])  ) , end="\r")

            for k in range(self.DeC.n_iter):
                # Update variables
                for var in self.problem.vars:
                    q_prev[var][:,:] = q_now[var][:,:]

                # Compute L2 high order space time discretization of the residual
                # And update of q_now
                DeC_one_step(self.problem, self.DeC, self.FEM2D, dt, self.stab_coeff,\
                             self.stab_curl_coeff, q_prev, L2, q_now, sub_sources = sub_sources,\
                             coriolis_not_uni = cor_nu,\
                             get_residual=get_residual,\
                             get_stabilization=get_stabilization,\
                             curl_stabilization=curl_stabilization,\
                             dirichlet_BC=dirichlet_BC,
                             curl_stab_flag = curl_stab_flag)
            
            for ivar, var in enumerate(self.problem.vars):
                q[ivar,:] = q_now[var][-1,:]

            it+=1
            t=t+dt
            t_save+=dt
            if t_save > dt_save:
                it_save+=1
                t_save = 0.
                for var in self.problem.vars:
                    q_save[var][it_save,:] = q_now[var][-1,:]
                tt_save[it_save] = t

        print("Iteration %07d, time %1.5f, max vars %1.3f  %1.3f  %1.3f ,  min vars %1.3f  %1.3f  %1.3f "%(it,t,\
                    np.max(q[0,:]),np.max(q[1,:]),np.max(q[2,:]),\
                    np.min(q[0,:]),np.min(q[1,:]),np.min(q[2,:])  ))
            

        # Final step to save
        it_save+=1
        Nt_save = it_save
        for var in self.problem.vars:
            q_save[var][it_save,:] = q_now[var][-1,:]
            q_save[var] = q_save[var][:Nt_save+1,:]
        tt_save[it_save] = t
        tt_save = tt_save[:Nt_save+1] 
        comp_time = time.time() - tic 
        print("Simulation over in %1.2f seconds"%comp_time)


        if (with_error or with_error_vertex) and self.problem.exact is not None:
            print("Computing exact solution and error")
            for it_save, t in enumerate(tt_save):
                if it_save == 0:
                    continue
                # Computing error
                dt_tmp = (tt_save[it_save]- tt_save[it_save-1])/self.problem.T_fin
                for ivar, var in enumerate(self.problem.vars):
                    if with_error:
                        ex = self.FEM2D.evaluate_function(lambda x,y: self.problem.exact[var](x,y,t))
                        error[ivar] += np.linalg.norm(q_save[var][it_save,:]-ex)/np.sqrt(self.FEM2D.n_dof_tot)*dt_tmp
                    if with_error_vertex:
                        ex = self.FEM2D.evaluate_function_vertex(lambda x,y: self.problem.exact[var](x,y,t))
                        sol_vertex = self.FEM2D.from_vector_to_vertex(q_save[var][it_save,:])
                        error_vertex[ivar] += np.linalg.norm(sol_vertex-ex)/np.sqrt(len(ex))*dt_tmp

        if save_sol is not None:
            q_final = dict()
            for var in self.problem.vars:
                q_final[var] = q_save[var][-1]

            sol_to_save = [q_final, tt_save[-1], comp_time, error, error_vertex ]
            # Open a file and use dump()
            savefile_name = self.problem.folderName+"/final_sol_"+method_name+"_ord_%d_N_%04d.pkl"%(self.FEM2D.FEM1Dx.degree+1,self.FEM2D.geom.N_elem_dir[0])
            with open(savefile_name, 'wb') as file:
                # A new file will be created
                pickle.dump(sol_to_save, file)
        
        
        print("")
        return q_save, tt_save, comp_time, error, error_vertex
    
class ImplicitEuler(DeCSpaceTimeSUPGSolver):
    
    def build_whole_q_vector(self, q:dict, vect_q:np.ndarray)->None:
        """
        Builds a whole vector stacking along dimension 0 the arrays in q.
        """
        curr_i = 0
        for k in self.problem.vars:
            size_q = q[k].shape[1]
            vect_q[:, curr_i:curr_i+size_q] = q[k]
            curr_i += size_q
        

    def split_whole_q_vector(self, q:dict, vect_q:np.ndarray):
        """
        Splits vect_q into dictionnary q.
        It supposes that vect_q is just q stacked along dimension 0.
        """
        curr_i = 0
        for k in self.problem.vars:
            size_q = q[k].shape[1]
            q[k][:,:] = vect_q[:, curr_i:curr_i+size_q]
            curr_i += size_q


    def build_whole_matrices(self,a,dx, dirichlet_BC = None):
        A_C=sparse.csr_matrix((self.FEM2D.n_dof_tot*3,self.FEM2D.n_dof_tot*3))
        A_SU=sparse.csr_matrix((self.FEM2D.n_dof_tot*3,self.FEM2D.n_dof_tot*3))
        Eps_CGFq=sparse.csr_matrix((self.FEM2D.n_dof_tot*3,self.FEM2D.n_dof_tot*3))
        Eps_SUGFq=sparse.csr_matrix((self.FEM2D.n_dof_tot*3,self.FEM2D.n_dof_tot*3))
        zero = sparse.csr_matrix((self.FEM2D.n_dof_tot,self.FEM2D.n_dof_tot))

        A_C = vstack([hstack([self.FEM2D.operator["mass"], zero, zero]), \
                      hstack([zero, self.FEM2D.operator["mass"], zero]),\
                      hstack([zero, zero, self.FEM2D.operator["mass"]])])
        A_SU = a * dx * vstack([hstack([zero, zero, self.FEM2D.operator["DxI"]]), \
                       hstack([zero, zero, self.FEM2D.operator["DyI"]]),\
                       hstack([self.FEM2D.operator["DxI"], self.FEM2D.operator["DyI"], zero])])
        Eps_CGFq = vstack([hstack([zero, zero, self.FEM2D.operator["IDx"]]), \
                           hstack([zero, zero, self.FEM2D.operator["IDy"]]),\
                           hstack([self.FEM2D.operator["IDx_tilde"], self.FEM2D.operator["IDy_tilde"], zero])])
        Eps_SUGFq = a * dx * vstack([hstack([self.FEM2D.operator["DxDx_tilde"], self.FEM2D.operator["DxDy_tilde"], zero ]), \
                                     hstack([self.FEM2D.operator["DyDx_tilde"], self.FEM2D.operator["DyDy_tilde"], zero ]),\
                                     hstack([zero, zero, self.FEM2D.operator["DxDx"] + self.FEM2D.operator["DyDy"]])])

        if dirichlet_BC is not None:
            for bc_item in dirichlet_BC.keys():
                    for i in dirichlet_BC[bc_item].indexes:
                        A_C = delete_row_in_coo_and_keep_diag_one(A_C, i)
                        A_C = delete_row_in_coo_and_keep_diag_one(A_C, i + self.FEM2D.n_dof_tot)
                        A_C = delete_row_in_coo_and_keep_diag_one(A_C, i + 2*self.FEM2D.n_dof_tot)
        if dirichlet_BC is not None:
            for bc_item in dirichlet_BC.keys():
                    for i in dirichlet_BC[bc_item].indexes:
                        A_SU = put_zero_row_in_coo(A_SU, i)
                        A_SU = put_zero_row_in_coo(A_SU, i + self.FEM2D.n_dof_tot)
                        A_SU = put_zero_row_in_coo(A_SU, i + 2*self.FEM2D.n_dof_tot)
        if dirichlet_BC is not None:
            for bc_item in dirichlet_BC.keys():
                    for i in dirichlet_BC[bc_item].indexes:
                        Eps_CGFq = put_zero_row_in_coo(Eps_CGFq, i)
                        Eps_CGFq = put_zero_row_in_coo(Eps_CGFq, i + self.FEM2D.n_dof_tot)
                        Eps_CGFq = put_zero_row_in_coo(Eps_CGFq, i + 2*self.FEM2D.n_dof_tot)
        if dirichlet_BC is not None:
            for bc_item in dirichlet_BC.keys():
                    for i in dirichlet_BC[bc_item].indexes:
                        Eps_SUGFq = put_zero_row_in_coo(Eps_SUGFq, i)
                        Eps_SUGFq = put_zero_row_in_coo(Eps_SUGFq, i + self.FEM2D.n_dof_tot)
                        Eps_SUGFq = put_zero_row_in_coo(Eps_SUGFq, i + 2*self.FEM2D.n_dof_tot)

        return A_C+A_SU, Eps_CGFq + Eps_SUGFq

        
        
        



    def solve(self, stab_coeff = None, with_error = False, \
              with_error_vertex = False, GF=None, CFL = None, \
              save_sol = False, stab = None, trick_second_der = False, curl_stab_flag = False):
        """Run a full transient simulation.

        Parameters
        ----------
        stab_coeff:
            Optional stabilization coefficient. If `None`, defaults are chosen
            from polynomial degree and stabilization type.
        with_error, with_error_vertex:
            If enabled and an exact solution is available, compute errors 
            varying in time averaging on all dofs or on vertex dofs.
        GF:
            Optional override for global-flux formulation flag.
        CFL:
            Optional override for time-step scaling.
        save_sol:
            If truthy, writes final state and diagnostics to a pickle file.
        stab:
            Optional override for stabilization method (`SUPG` or `OSS`).
        trick_second_der:
            If changed, uses other second-derivative-related operators.
        curl_stab_flag:
            Adds optional curl-based OSS stabilization for velocity components.

        Returns
        -------
        q_save, tt_save, comp_time, error, error_vertex
            Saved solution snapshots, saved times, wall time, and optional
            error arrays.
        """

        error, error_vertex, method_name, error_name, get_residual, get_stabilization, curl_stabilization = \
                self.solver_set_parameters(stab_coeff, with_error, with_error_vertex, \
                                           GF, CFL, stab, trick_second_der)

        dt_save = self.problem.T_fin/(self.Nt_save-2)

        q_save = dict() 

        it = 0
        t=0.
        t_save  = 0
        it_save = 0
        L2 = dict()
        for var in self.problem.vars: # ("u", "v", "p")
            L2[var] = np.zeros(self.FEM2D.n_dof_tot)
            q_save[var] = np.zeros((self.Nt_save, self.FEM2D.n_dof_tot))
        tt_save = np.zeros(self.Nt_save)

        q_prev = dict()
        q_now  = dict()
        q      = np.zeros((len(self.problem.vars), self.FEM2D.n_dof_tot)) 
        for var in self.problem.vars:
            q_prev[var] = np.zeros((2, self.FEM2D.n_dof_tot))
            q_now[var]  = np.zeros((2, self.FEM2D.n_dof_tot))
            for i in range(2): #range(self.DeC.n_subNodes):
                q_now[var][i,:] = self.ic_vect[var]

        size_array = sum(np.array([q_prev[k].shape[1] for k in self.problem.vars]))            
        vect_q = np.empty((q_now['u'].shape[0], size_array))

        for var in self.problem.vars:
            q_save[var][it_save,:] = q_now[var][-1,:]
        tt_save[it_save] = t

        source = dict()
        if self.problem.source is not None:
            for var in self.problem.vars:
                source[var] = self.FEM2D.evaluate_function(lambda x,y: self.problem.source[var](x,y,0.))
        else:
            for var in self.problem.vars:
                source[var] = self.FEM2D.evaluate_function(lambda x,y: 0.)

        if self.problem.coriolis_non_uniform is not None:
            cor_nu = self.FEM2D.evaluate_function(self.problem.coriolis_non_uniform)
        else:
            cor_nu = self.FEM2D.evaluate_function(lambda x,y: 0.)

        if self.problem.dirichlet is not None:
            dirichlet_BC = dict()
            for bc_item in self.problem.dirichlet.keys():
                BC_values = dict()
                idxs = self.FEM2D.dirichlet_indexes[bc_item]
                for var in self.problem.dirichlet[bc_item]:
                    BC_values[var] = self.ic_vect[var][idxs]
                dirichlet_BC[bc_item] = Dirichlet_BC_set(idxs, BC_values)

        else:
            dirichlet_BC = None

        sub_sources = dict()
        for ivar, var in enumerate(self.problem.vars):
            sub_sources[var] = np.zeros_like(q_now[var])


        tic = time.time()

        #Define big matrices
        A, B = self.build_whole_matrices(self.stab_coeff, self.geom.dx_min, dirichlet_BC)
        
        #Enfore Dirichlet conditions
        if dirichlet_BC is not None:
            for bc_item in dirichlet_BC.keys():
                for m in range(2):
                    for var in dirichlet_BC[bc_item].vars:
                        q_prev[var][m,dirichlet_BC[bc_item].indexes] =\
                            dirichlet_BC[bc_item].dirichlet_vector[var]

        while (t<self.problem.T_fin and it<self.Nt_max):
            # Set dt
            dt =  self.CFL* self.geom.dx_min#self.CFL * self.problem.max_dt(q, self.geom.dx)
            c = self.problem.c

            # Initialize variables
            for ivar, var in enumerate(self.problem.vars):
                q[ivar,:] = q_now[var][-1,:]
                for i in range(self.DeC.M_sub):
                    q_now[var][i,:] = q_now[var][-1,:] # previous timestep last update
                if self.problem.source is not None:
                    for i in range(self.DeC.M_sub+1):
                        sub_sources[var][i,:] = self.FEM2D.evaluate_function(lambda x,y: self.problem.source[var](x,y,t+dt*self.DeC.beta[i]))

            print("Iteration %07d, time %1.5f, max vars %1.3f  %1.3f  %1.3f ,  min vars %1.3f  %1.3f  %1.3f "%(it,t,\
                    np.max(q[0,:]),np.max(q[1,:]),np.max(q[2,:]),\
                    np.min(q[0,:]),np.min(q[1,:]),np.min(q[2,:])  ) , end="\r")

            # Update variables
            for var in self.problem.vars:
                q_prev[var][:,:] = q_now[var][:,:]

            # Compute L2 high order space time discretization of the residual
            # And update of q_now
            self.implicitEuler_one_step(dt, A, B, \
                         q_prev, vect_q, q_now, sub_sources = sub_sources,\
                         coriolis_not_uni = cor_nu,\
                         dirichlet_BC=dirichlet_BC,
                         curl_stab_flag = curl_stab_flag)
            
            for ivar, var in enumerate(self.problem.vars):
                q[ivar,:] = q_now[var][-1,:]

            it+=1
            t=t+dt
            t_save+=dt
            if t_save > dt_save:
                it_save+=1
                t_save = 0.
                for var in self.problem.vars:
                    q_save[var][it_save,:] = q_now[var][-1,:]
                tt_save[it_save] = t

        print("Iteration %07d, time %1.5f, max vars %1.3f  %1.3f  %1.3f ,  min vars %1.3f  %1.3f  %1.3f "%(it,t,\
                    np.max(q[0,:]),np.max(q[1,:]),np.max(q[2,:]),\
                    np.min(q[0,:]),np.min(q[1,:]),np.min(q[2,:])  ))
            

        # Final step to save
        it_save+=1
        Nt_save = it_save
        for var in self.problem.vars:
            q_save[var][it_save,:] = q_now[var][-1,:]
            q_save[var] = q_save[var][:Nt_save+1,:]
        tt_save[it_save] = t
        tt_save = tt_save[:Nt_save+1] 
        comp_time = time.time() - tic 
        print("Simulation over in %1.2f seconds"%comp_time)


        if (with_error or with_error_vertex) and self.problem.exact is not None:
            print("Computing exact solution and error")
            for it_save, t in enumerate(tt_save):
                if it_save == 0:
                    continue
                # Computing error
                dt_tmp = (tt_save[it_save]- tt_save[it_save-1])/self.problem.T_fin
                for ivar, var in enumerate(self.problem.vars):
                    if with_error:
                        ex = self.FEM2D.evaluate_function(lambda x,y: self.problem.exact[var](x,y,t))
                        error[ivar] += np.linalg.norm(q_save[var][it_save,:]-ex)/np.sqrt(self.FEM2D.n_dof_tot)*dt_tmp
                    if with_error_vertex:
                        ex = self.FEM2D.evaluate_function_vertex(lambda x,y: self.problem.exact[var](x,y,t))
                        sol_vertex = self.FEM2D.from_vector_to_vertex(q_save[var][it_save,:])
                        error_vertex[ivar] += np.linalg.norm(sol_vertex-ex)/np.sqrt(len(ex))*dt_tmp

        if save_sol is not None:
            q_final = dict()
            for var in self.problem.vars:
                q_final[var] = q_save[var][-1]

            sol_to_save = [q_final, tt_save[-1], comp_time, error, error_vertex ]
            # Open a file and use dump()
            savefile_name = self.problem.folderName+"/final_sol_"+method_name+"_ord_%d_N_%04d.pkl"%(self.FEM2D.FEM1Dx.degree+1,self.FEM2D.geom.N_elem_dir[0])
            with open(savefile_name, 'wb') as file:
                # A new file will be created
                pickle.dump(sol_to_save, file)
        
        
        print("")
        return q_save, tt_save, comp_time, error, error_vertex

    def implicitEuler_one_step(self, dt, A, B, q_prev, vect_q, q_now,\
                           sub_sources, coriolis_not_uni, \
                           dirichlet_BC = None, curl_stab_flag=False):
        """Perform one implicit Euler correction sweep over all sub-nodes.
    
        This routine assembles sources, residuals, and stabilization terms for each
        node and applies one implicit update using `inv_lump`.
        """
    
        # Compute L2 high order space time discretization of the residual
    
        c   = self.problem.c
        cor = self.problem.coriolis
        op  = self.FEM2D.operator
        fric = self.problem.friction
        stab_curl_coeff = self.stab_curl_coeff

        self.build_whole_q_vector(q_prev, vect_q)

        #Define RHS
        RHS = A @ vect_q[0,:]
        vect_q[1,:] = sparse.linalg.spsolve(A+dt*B, RHS) 
        #Just out of curiosity: we could try different solvers. 
        # spsolve is a LU solver, so not necessarily the best for big systems...
        #vect_q[1,:] = sparse.linalg.bicgstab(A+dt*B, RHS) 
        #vect_q[1,:], _ = sparse.linalg.gmres(A+dt*B, RHS) 
        #vect_q[1,:], _ = sparse.linalg.gmres(A, (A-dt*B)@vect_q[0,:])
        #RHS = (A - 0.5*dt*B)@ vect_q[0,:] #Let's try Crank-Nicholson?
        ##vect_q[1,:], _ = sparse.linalg.gmres((A+0.5*dt*B), RHS)
        #vect_q[1,:] = sparse.linalg.spsolve(A+0.5*dt*B, RHS) 

        self.split_whole_q_vector(q_now, vect_q)

        if dirichlet_BC is not None:
            for bc_item in dirichlet_BC.keys():
                for m in range(2):
                    for var in dirichlet_BC[bc_item].vars:
                        q_now[var][m,dirichlet_BC[bc_item].indexes] =\
                            dirichlet_BC[bc_item].dirichlet_vector[var]

def define_sources(all_sources, q_prev, sub_sources, theta_m, cor, coriolis_not_uni, fric):
    """Build momentum and pressure source terms in semi-discrete form.

    Signs follow the convention used in `DeC_one_step`, where the assembled
    source terms are moved to the left-hand side of the residual equations.
    """

    all_sources["u"][:] = -cor* (theta_m @ q_prev["v"])\
            - theta_m @ (q_prev["v"]*coriolis_not_uni)\
            + fric* (theta_m @ q_prev["u"])\
            - theta_m@sub_sources["u"]
    all_sources["v"][:] = cor* (theta_m @ q_prev["u"])\
            + theta_m @ (q_prev["u"]*coriolis_not_uni)\
            + fric* (theta_m @ q_prev["v"])\
            - theta_m@sub_sources["v"]
    all_sources["p"][:] = - theta_m@sub_sources["p"]
    return all_sources


def define_residuals(galer_residuals, q_prev,all_sources,m,op,c,dx_min , al, theta_m, dt):

    """Assemble Galerkin residuals (no stabilization) 
    for the standard (non-GF) formulation."""

    galer_residuals["u"][:] = op["mass"]@(q_prev["u"][m,:]-q_prev["u"][0,:])/dt\
        +c*   op["IDx"] @(theta_m @ q_prev["p"] )\
        +     op["mass"]@all_sources["u"]

    galer_residuals["v"][:] = op["mass"]@(q_prev["v"][m,:]-q_prev["v"][0,:])/dt\
        +c  * op["IDy"] @(theta_m @ q_prev["p"] )\
        +     op["mass"]@all_sources["v"]
        
    galer_residuals["p"][:] = op["mass"]@(q_prev["p"][m,:]-q_prev["p"][0,:])/dt\
        +c*op["IDx"]@(theta_m @ q_prev["u"] )\
        +c*op["IDy"]@(theta_m @ q_prev["v"] )\
        + op["mass"]@all_sources["p"]

    return galer_residuals

def define_GF_residuals(galer_residuals, q_prev,all_sources,m,op,c,dx_min , al, theta_m, dt):

    """Assemble Galerkin residuals for the global-flux (GF) formulation."""

    galer_residuals["u"][:] = op["mass"]@(q_prev["u"][m,:]-q_prev["u"][0,:])/dt\
        +c*op["IDx"]@(theta_m @ q_prev["p"] )\
        +  op["mass_tilde_x"]@all_sources["u"]
    
    galer_residuals["v"][:] = op["mass"]@(q_prev["v"][m,:]-q_prev["v"][0,:])/dt\
        +c*op["IDy"]@(theta_m @ q_prev["p"] )\
        +  op["mass_tilde_y"]@all_sources["v"]
    
    galer_residuals["p"][:] = op["mass"]@(q_prev["p"][m,:]-q_prev["p"][0,:])/dt\
        +c*op["IDx_tilde"] @(theta_m @ q_prev["u"] )\
        +c*op["IDy_tilde"] @(theta_m @ q_prev["v"] )\
        +  op["mass_tilde"]@all_sources["p"]

    return galer_residuals

def SUPG_stabilization(all_stabs, q_prev,all_sources,m,op,c,dx_min , al, theta_m, dt):
          
    """Compute SUPG stabilization contributions for the standard formulation."""

    all_stabs["u"][:] = \
            al*dx_min*op["DxI"] @(q_prev["p"][m,:]-q_prev["p"][0,:])/dt\
        +c*al*dx_min*op["DxDx2"]@(theta_m@q_prev["u"])\
        +c*al*dx_min*op["DxDy"]@(theta_m@q_prev["v"])\
        +  al*dx_min*op["DxI"] @all_sources["p"]

    all_stabs["v"][:] = \
            al*dx_min*op["DyI"] @(q_prev["p"][m,:]-q_prev["p"][0,:])/dt\
        +c*al*dx_min*op["DyDx"]@(theta_m@q_prev["u"])\
        +c*al*dx_min*op["DyDy2"]@(theta_m@q_prev["v"])\
        +  al*dx_min*op["DyI"] @all_sources["p"]
    
    
    all_stabs["p"][:] = \
           al*dx_min*op["DxI"] @(q_prev["u"][m,:]-q_prev["u"][0,:])/dt\
        +  al*dx_min*op["DyI"] @(q_prev["v"][m,:]-q_prev["v"][0,:])/dt\
        +c*al*dx_min*op["DxDx2"]@(theta_m@q_prev["p"])\
        +c*al*dx_min*op["DyDy2"]@(theta_m@q_prev["p"])\
        +  al*dx_min*op["DxI"]@all_sources["u"]\
        +  al*dx_min*op["DyI"]@all_sources["v"]
    
    return all_stabs


def SUPG_GF_stabilization(all_stabs, q_prev,all_sources,m,op,c,dx_min , al, theta_m, dt):
    """Compute SUPG stabilization contributions for the GF formulation."""

    all_stabs["u"][:] = \
           al*dx_min*op["DxI"]@(q_prev["p"][m,:]-q_prev["p"][0,:])/dt\
        +c*al*dx_min*op["DxDx2_tilde"]@(theta_m@q_prev["u"])\
        +c*al*dx_min*op["DxDy_tilde"]@(theta_m@q_prev["v"])\
        +  al*dx_min*op["DxM_tilde"]@all_sources["p"]
    
    all_stabs["v"][:] = \
           al*dx_min*op["DyI"]       @(q_prev["p"][m,:]-q_prev["p"][0,:])/dt\
        +c*al*dx_min*op["DyDx_tilde"]@(theta_m@q_prev["u"])\
        +c*al*dx_min*op["DyDy2_tilde"]@(theta_m@q_prev["v"])\
        +  al*dx_min*op["DyM_tilde"] @all_sources["p"]
    
    all_stabs["p"][:] = \
           al*dx_min*op["DxI"]      @(q_prev["u"][m,:]-q_prev["u"][0,:])/dt\
        +  al*dx_min*op["DyI"]      @(q_prev["v"][m,:]-q_prev["v"][0,:])/dt\
        +c*al*dx_min*op["DxDx2"]     @(theta_m@q_prev["p"])\
        +c*al*dx_min*op["DyDy2"]     @(theta_m@q_prev["p"])\
        +  al*dx_min*op["DxI_tilde"]@all_sources["u"]\
        +  al*dx_min*op["DyI_tilde"]@all_sources["v"]
    
    return all_stabs


def OSS_stabilization(all_stabs, q_prev,all_sources,m,op,c,dx_min , al, theta_m, dt):
          
    """Compute OSS stabilization for the standard formulation."""

    all_stabs["u"][:] = \
         c*al*dx_min*op["ZxMy"]@(theta_m@q_prev["u"])

    all_stabs["v"][:] = \
         c*al*dx_min*op["ZyMx"]@(theta_m@q_prev["v"])
    
    theta_p = theta_m@q_prev["p"]
    all_stabs["p"][:] = \
         c*al*dx_min*op["ZxMy"]@theta_p\
        +c*al*dx_min*op["ZyMx"]@theta_p
    
    return all_stabs



def OSS_GF_stabilization(all_stabs, q_prev,all_sources,m,op,c,dx_min , al, theta_m, dt):
    

    """Compute OSS stabilization for the GF formulation."""

    all_stabs["u"][:] = \
         c*al*dx_min*(op["ZxMy_tilde"]@(theta_m@q_prev["u"])\
                     +op["DyZx_int"]@(theta_m@q_prev["v"])\
                     +op["My_tilde_Zx_int"]@all_sources["p"]\
                    #  +1e-3*(op["DxDx2_tilde"]@(theta_m@q_prev["u"])\
                    #         +op["DxDy_tilde"]@(theta_m@q_prev["v"])\
                    #         -op["DxM_tilde"]@all_sources["p"]
                    #        )
                     )

    all_stabs["v"][:] = \
         c*al*dx_min*(op["ZyMx_tilde"]@(theta_m@q_prev["v"])\
                     +op["DxZy_int"]@(theta_m@q_prev["u"])\
                     +op["Mx_tilde_Zy_int"]@all_sources["p"]
                     )
    
    all_stabs["p"][:] = \
         c*al*dx_min*(op["ZxMy"]@(theta_m@q_prev["p"])\
                     +op["ZyMx"]@(theta_m@q_prev["p"])\
                     +op["Zx_int_My"]@all_sources["u"]\
                     +op["Zy_int_Mx"]@all_sources["v"]
                    )
    
    return all_stabs



def OSS_curl_stabilization(all_stabs, q_prev,all_sources,m,op,c,dx_min , al, theta_m, dt):
    """Compute optional curl-targeted OSS stabilization (standard operators)."""

    beta_m = np.sum(theta_m)
    dtu = (q_prev["u"][m]-q_prev["u"][0])/dt/beta_m - all_sources["u"]
    dtv = (q_prev["v"][m]-q_prev["v"][0])/dt/beta_m - all_sources["v"]

    all_stabs["u"][:] = c*al*dx_min*op["ZyMx"]@dtu
    
    all_stabs["v"][:] = c*al*dx_min*op["ZxMy"]@dtv
    
    return all_stabs


def OSS_GF_curl_stabilization(all_stabs, q_prev,all_sources,m,op,c,dx_min , al, theta_m, dt):
    """Compute optional curl-targeted OSS stabilization (GF operators)."""

    beta_m = np.sum(theta_m)
    dtu = (q_prev["u"][m]-q_prev["u"][0])/dt/beta_m - all_sources["u"]
    dtv = (q_prev["v"][m]-q_prev["v"][0])/dt/beta_m - all_sources["v"]

    all_stabs["u"][:] = c*al*dx_min*(op["ZyMx_tilde"]@dtu-op["DxZy_int"]@dtv)
    
    all_stabs["v"][:] = c*al*dx_min*(op["ZxMy_tilde"]@dtv-op["DyZx_int"]@dtv)
    
    return all_stabs


def DeC_one_step(problem, DeC, FEM2D, dt, al, stab_curl_coeff, q_prev, L2, q_now,\
                       sub_sources, coriolis_not_uni, get_residual, get_stabilization, curl_stabilization,\
                       dirichlet_BC = None, curl_stab_flag=False):
    """Perform one DeC correction sweep over all sub-nodes.

    This routine assembles sources, residuals, and stabilization terms for each
    DeC sub-node and applies one explicit update using `inv_lump`.
    """

    # Compute L2 high order space time discretization of the residual

    c   = problem.c
    cor = problem.coriolis
    op  = FEM2D.operator
    fric = problem.friction

    all_sources   = dict()
    gal_residuals = dict()
    all_stabs     = dict()
    for var in problem.vars:
        all_sources[var]   = np.empty(q_prev[var][0,:].shape)
        gal_residuals[var] = np.empty(q_prev[var][0,:].shape)
        all_stabs[var]     = np.empty(q_prev[var][0,:].shape)


    for m in range(1,DeC.n_subNodes):

        # Carefull with the signs! source_u,_v,_p are meant on the LHS, while the other source was on the RHS
        define_sources(all_sources, q_prev, sub_sources, DeC.theta[m,:], cor, coriolis_not_uni, fric)
        get_residual(gal_residuals, q_prev,all_sources,m,op,c,FEM2D.geom.dx_min, al, DeC.theta[m,:], dt)
        get_stabilization(all_stabs, q_prev,all_sources,m,op,c,FEM2D.geom.dx_min, al, DeC.theta[m,:], dt)

        for var in problem.vars:
            L2[var][:] = gal_residuals[var]+ all_stabs[var]

        if curl_stab_flag:
            curl_stabilization(all_stabs,q_prev,all_sources,m,op,c,FEM2D.geom.dx_min, stab_curl_coeff, DeC.theta[m,:], dt)
            for var in ["u","v"]:
                L2[var][:] += all_stabs[var]


        for var in problem.vars:
            q_now[var][m,:] = q_prev[var][m,:] - dt*op["inv_lump"]@L2[var][:]

        if dirichlet_BC is not None:
            for bc_item in dirichlet_BC.keys():
                for var in dirichlet_BC[bc_item].vars:
                    q_now[var][m,dirichlet_BC[bc_item].indexes] =\
                        dirichlet_BC[bc_item].dirichlet_vector[var]



def get_stencil_indexes(i_cell, degree):
    """Return left/right stencil bounds for a 1D cell index."""

    jl = (i_cell-1)*degree +degree
    jr = (i_cell+2)*degree +degree
    return jl, jr



def invert_lumped_matrix(lump):
    """Return diagonal inverse of a lumped mass matrix in sparse format."""

    ll = dia_matrix(lump)
    dd = 1./ll.diagonal()
    siz = len(dd)
    return dia_matrix((dd.reshape((1,-1)),np.array([0])), shape=(siz,siz))

def put_zero_row_in_csr(A, i):
    """Set to zero all entries of row `i` in a CSR sparse matrix."""

    if type(A)==sparse.csr.csr_matrix:
        A.data[A.indptr[i]:A.indptr[i+1]] = 0
    else:
        raise ValueError("The type of the matrix is not csr to put to zero the row")
        
def put_zero_row_in_coo(A, i):
    """Delete row `i` from a COO sparse matrix and return a new COO matrix."""

    idx_row = A.row==i
    idx_higher_rows = A.row>i

    new_data = np.copy(A.data)
    new_col = np.copy(A.col)
    new_row = np.copy(A.row)

    new_data=np.delete(new_data,idx_row)
    new_col = np.delete(new_col, idx_row)
    new_row= np.delete(new_row,idx_row)

    return sparse.coo_matrix((new_data,(new_row,new_col)), shape = (A.shape[0],A.shape[1]))

def delete_row_in_coo(A, i):
    """Delete row `i` from a COO sparse matrix and return a new COO matrix."""

    idx_row = A.row==i
    idx_higher_rows = A.row>i

    new_data = np.copy(A.data)
    new_col = np.copy(A.col)
    new_row = np.copy(A.row)

    new_data=np.delete(new_data,idx_row)
    new_col = np.delete(new_col, idx_row)
    new_row[idx_higher_rows]-=1
    new_row= np.delete(new_row,idx_row)

    return sparse.coo_matrix((new_data,(new_row,new_col)), shape = (A.shape[0]-1,A.shape[1]))

def delete_row_in_coo_and_keep_diag_one(A, i):
    """Delete row `i` from a COO sparse matrix and return a new COO matrix."""

    idx_row = (A.row==i) & (A.col != i)
    diag_i = (A.row==i) & (A.col == i)
    idx_higher_rows = A.row>i
    
    new_data = np.copy(A.data)
    new_col = np.copy(A.col)
    new_row = np.copy(A.row)
    
    new_data[diag_i] = 1.0 
    new_data=np.delete(new_data,idx_row)
    new_col = np.delete(new_col, idx_row)
    #new_row[idx_higher_rows]-=1
    new_row= np.delete(new_row,idx_row)

    return sparse.coo_matrix((new_data,(new_row,new_col)), shape = (A.shape[0],A.shape[1]))



