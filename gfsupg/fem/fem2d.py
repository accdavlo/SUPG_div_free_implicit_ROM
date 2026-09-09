"""2D FEM operator assembly on a Cartesian tensor-product mesh (Scipy2DFEM)"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
import scipy.sparse as sp
import time
import pickle, os

from scipy.sparse import hstack, vstack, identity

from scipy.optimize import LinearConstraint, minimize

from .boundary import assemble_1D_sparse_matrix
from ..sparse_utils import put_zero_row_in_csr, invert_lumped_matrix, delete_row_in_coo


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

    def __init__(self, geom, FEM1Dx, FEM1Dy=None, folder=None, force_matrix_assembly=False, save_operators=False):
        
        self.save_operators = save_operators
        if self.save_operators and folder is None:
            self.save_operators = False
        
        if not self.save_operators:
            self.force_matrix_assembly = True

        if folder is not None:
            save_operators_path = geom.geometry_folder + 'FEM2D_operators_ord%d_N%04d.pkl'%(FEM1Dx.degree+1,geom.N_elem_dir[0])

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
            if k == 0:
                FEM1D = self.FEM1Dx
            elif k == 1:
                FEM1D = self.FEM1Dy
            self.xx_dofs[k] = np.reshape([x_cell + geom.dx[k]*FEM1D.nodes[:-1] for x_cell in geom.xx[k]],-1)
            if self.geom.BC[0] != 0: # non periodic case
                if FEM1D.degree > 1:
                    self.xx_dofs[k] = np.delete(self.xx_dofs[k], np.s_[-FEM1D.degree+1:])
            self.n_dof_dir[k] = len(self.xx_dofs[k]) 
        
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
            toc = time.time() - tic
            
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
            toc = time.time() - tic
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
        self.mat_2_vect_vertex = np.zeros((self.geom.N_elem_dir[0] + 1,self.geom.N_elem_dir[1] + 1), dtype=np.int64)
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
                matrix[i,i] = 1.0

    def apply_dirichlet_bc_rhs(self, boundaries, rhs, value):
        for boundary in boundaries:
            for i in self.dirichlet_indexes[boundary]:
                rhs[i] = value[i]

    def vect_to_mat(self, u_vec):
        u_mat = np.zeros((self.n_dof_dir[0],self.n_dof_dir[1]))
        for i in range(self.n_dof_tot):
            u_mat[self.vect_2_mat[i,0],self.vect_2_mat[i,1]] =\
                u_vec[i]
        return u_mat
    
    def vect_to_mat_vertex(self, u_vec):
        u_mat = np.zeros((self.geom.N_elem_dir[0] + 1,self.geom.N_elem_dir[1] + 1))
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
        
        if direction == 0:
            FEM1D = self.FEM1Dx
        elif direction == 1:
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

    def build_matrices_MOR(self, ROM):
        basis = ROM.basis
        n_dof_rb = ROM.n_rb
        self.n_dof_rb = dict()
        self.operator_MOR = dict()
        if ROM.variable_split == "u,v,p":
            for var in tuple(basis.keys()):
                self.operator_MOR[var] = dict()
                self.n_dof_rb[var] = n_dof_rb[var]
                for var_bis in tuple(basis.keys()):
                    self.operator_MOR[var][var_bis] = dict()
                    for i, matrix in enumerate(self.operator):
                        #print("Assembling MOR Matrices  %02d/%d"%(i,tot_mat+1), end="\r")
                        self.operator_MOR[var][var_bis][matrix] = basis[var].T @ self.operator[matrix] @ basis[var_bis]
        elif ROM.variable_split=="uv,p":
            # We look at the SUPG operators I need
            zero = sp.csr_matrix((self.n_dof_tot,self.n_dof_tot))
            for ivar in ROM.vars:
                self.operator_MOR[ivar] = dict()
                self.n_dof_rb[ivar] = n_dof_rb[ivar]
                for jvar in ROM.vars:
                    self.operator_MOR[ivar][jvar] = dict()
            self.operator_MOR["uv"]["uv"]["mass"] = basis["uv"].T @ \
            vstack([ hstack([self.operator["mass"], zero]),
                     hstack([zero, self.operator["mass"]]) ]) @ basis["uv"]
            
            self.operator_MOR["uv"]["uv"]["mass_tilde_xy"] = basis["uv"].T @ \
            vstack([ hstack([self.operator["mass_tilde_x"], zero]),
                     hstack([zero, self.operator["mass_tilde_y"]]) ]) @ basis["uv"]
            
            self.operator_MOR["p"]["p"]["mass"] = basis["p"].T @ self.operator["mass"] @ basis["p"]

            self.operator_MOR["p"]["p"]["mass_tilde"] = basis["p"].T @ self.operator["mass_tilde"] @ basis["p"]

            self.operator_MOR["uv"]["p"]["IGrad"] = basis["uv"].T @ \
                vstack([self.operator["IDx_tilde"], self.operator["IDy_tilde"]]) @ basis["p"]

            self.operator_MOR["p"]["uv"]["IDiv"] = basis["p"].T @ \
                hstack([self.operator["IDx"], self.operator["IDy"]]) @ basis["uv"]

            self.operator_MOR["p"]["uv"]["IDiv_tilde"] = basis["p"].T @ \
                hstack([self.operator["IDx_tilde"], self.operator["IDy_tilde"]]) @ basis["uv"]

            self.operator_MOR["uv"]["p"]["GradI"] = basis["uv"].T @ \
                vstack([self.operator["DxI"], self.operator["DyI"]]) @ basis["p"]

            self.operator_MOR["p"]["uv"]["DivI"] = basis["p"].T @ \
                hstack([self.operator["DxI"], self.operator["DyI"]]) @ basis["uv"]

            self.operator_MOR["p"]["uv"]["DivI_tilde"] = basis["p"].T @ \
                hstack([self.operator["DxI_tilde"], self.operator["DyI_tilde"]]) @ basis["uv"]

            self.operator_MOR["uv"]["uv"]["GradDiv"] = basis["uv"].T @ \
                vstack([hstack([self.operator["DxDx2"], self.operator["DxDy"]]),
                        hstack([self.operator["DyDx"], self.operator["DyDy2"]])]) @ basis["uv"]

            self.operator_MOR["uv"]["uv"]["GradDiv_tilde"] = basis["uv"].T @ \
                vstack([hstack([self.operator["DxDx2_tilde"], self.operator["DxDy_tilde"]]),
                        hstack([self.operator["DyDx_tilde"], self.operator["DyDy2_tilde"]])]) @ basis["uv"]

            self.operator_MOR["p"]["p"]["DivGrad"] = basis["p"].T @ \
                (self.operator["DxDx2"] + self.operator["DyDy2"]) @ basis["p"]

            self.operator_MOR["p"]["p"]["DivGrad_tilde"] = basis["p"].T @ \
                (self.operator["DxDx2_tilde"] + self.operator["DyDy2_tilde"]) @ basis["p"]
            
            self.operator_MOR["uv"]["p"]["GradM_tilde"] = basis["uv"].T @ \
                vstack([self.operator["DxM_tilde"], self.operator["DyM_tilde"]]) @ basis["p"]

            self.operator_MOR["uv"]["uv"]["perp"] = np.linalg.inv(basis["uv"].T@basis["uv"]) @\
                basis["uv"].T @ \
                np.vstack([basis["uv"][self.n_dof_tot:,:], -basis["uv"][:self.n_dof_tot,:]])

            self.operator_MOR["uv"]["uv"]["ZgradMgrad"] = basis["uv"].T @ \
                vstack([hstack([self.operator["ZxMy"], zero ]),
                        hstack([zero, self.operator["ZyMx"]])]) @ basis["uv"]
            
            self.operator_MOR["uv"]["uv"]["ZgradMgrad_tilde"] = basis["uv"].T @ \
                vstack([hstack([self.operator["ZxMy_tilde"], self.operator["DyZx_int"]]),\
                        hstack([self.operator["DxZy_int"], self.operator["ZyMx_tilde"]])]) @ basis["uv"]
            
            self.operator_MOR["p"]["p"]["ZdivMdiv"] = basis["p"].T @ \
                (self.operator["ZxMy"] + self.operator["ZyMx"]) @ basis["p"]

            self.operator_MOR["uv"]["p"]["Mgrad_tilde_Zgrad_int"] = basis["uv"].T @ \
            vstack([self.operator["My_tilde_Zx_int"], self.operator["Mx_tilde_Zy_int"]]) @ basis["p"]

            self.operator_MOR["p"]["uv"]["Zgrad_int_Mdiv"] = basis["p"].T @ \
            hstack([self.operator["Zx_int_My"], self.operator["Zy_int_Mx"]]) @ basis["uv"]

            self.operator_MOR["p"]["p"]["inv_lump"] = basis["p"].T @ self.operator["inv_lump"] @ basis["p"]
            self.operator_MOR["uv"]["uv"]["inv_lump"] = basis["uv"].T @ \
                vstack([hstack([self.operator["inv_lump"],zero]),\
                        hstack([zero, self.operator["inv_lump"]])]) @ basis["uv"]

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
        norm_res=dict()
        if problem == "acoustics":
            for var in ("p","u","v"):
                res[var] = np.zeros_like(q["u"])
            res["p"] = (self.operator["IDx_tilde"]@q["u"] \
                + self.operator["IDy_tilde"]@q["v"] \
                - self.operator["mass_tilde"]@source["p"]) \
                / self.geom.dx[0]/self.geom.dx[1]
            res["u"] = (self.operator["IDx"]@q["p"] \
                - self.operator["mass_tilde_x"]@source["u"]) \
                / self.geom.dx[0]
            res["v"] = (self.operator["IDy"]@q["p"]\
                - self.operator["mass_tilde_y"]@source["v"]) \
                / self.geom.dx[1]
        else:
            raise NotImplementedError("Equation not implemented for GF residuals in Scipy2DFEM")
    
        for var in ("p","u","v"):
            norm_res[var] = np.linalg.norm(res[var]*self.geom.dx[0]*self.geom.dx[1], 1)
        
        return res, norm_res

    # GO: Is here the right place for these functions?
    def compute_GF_residual_MOR(self, ROM, q, source, basis, problem="acoustics"):
        res_rb = dict()
        norm_res_rb = dict()

        if ROM.variable_split == "u,v,p":
            for var in ("p","u","v"):
                res_rb[var] = np.zeros_like(q[var])
            
            res_rb["p"] = (self.operator_MOR["p"]["u"]["IDx_tilde"]@q["u"] \
                + self.operator_MOR["p"]["v"]["IDy_tilde"]@q["v"] \
                - basis["p"].T@(self.operator["mass_tilde"]@source["p"])) \
                / self.geom.dx[0]/self.geom.dx[1]
            
            res_rb["u"] = (self.operator_MOR["u"]["p"]["IDx"]@q["p"] \
                - basis["u"].T@(self.operator["mass_tilde_x"]@source["u"])) \
                / self.geom.dx[0]
            
            res_rb["v"] = (self.operator_MOR["v"]["p"]["IDy"]@q["p"] \
                - basis["v"].T@(self.operator["mass_tilde_y"]@source["v"])) \
                / self.geom.dx[1]
    
        elif ROM.variable_split=="uv,p":
            res_rb["p"] = (self.operator_MOR["p"]["uv"]["IDiv_tilde"]@q["uv"]\
                - self.operator_MOR["p"]["p"]["mass_tilde"]@source["p"]) \
                / self.geom.dx[0]/self.geom.dx[1]
            
            res_rb["uv"] = (self.operator_MOR["uv"]["p"]["IGrad"]@ q["p"]\
                - self.operator_MOR["uv"]["uv"]["mass_tilde_xy"]@source["uv"]) \
                / self.geom.dx[0]
        for var in ROM.vars:
            norm_res_rb[var] = np.linalg.norm(res_rb[var]*self.geom.dx[0]*self.geom.dx[1],1)
        
        return res_rb, norm_res_rb

    def compute_noGF_residual(self, q, source, problem="acoustics"):
        res = dict()
        if problem == "acoustics":
            for var in ("p","u","v"):
                res[var] = np.zeros_like(q["u"])
            res["p"] = (self.operator["IDx"]@q["u"] \
                + self.operator["IDy"]@q["v"] \
                - self.operator["mass"]@source["p"]) \
                / self.geom.dx[0]/self.geom.dx[1]
            res["u"] = (self.operator["IDx"]@q["p"] \
                - self.operator["mass"]@source["u"]) \
                / self.geom.dx[0]
            res["v"] = (self.operator["IDy"]@q["p"] \
                - self.operator["mass"]@source["v"]) \
                / self.geom.dx[1]
        else:
            raise NotImplementedError("Equation not implemented for noGF residuals in Scipy2DFEM")
        
        return res

    def compute_discrete_curl(self, q):
        return self.operator["inv_lump"]@(self.operator["IDy"]@q["u"] - self.operator["IDx"]@q["v"])

    def compute_discrete_curl_involution(self, q, alpha, dx,dy):
        """Discrete involution with stabilization terms that should be preserved
        consistent with a curl"""
        op = self.operator
        K_u = dx*op["DxDx"]@q["u"] + dy*op["DyDy"]@q["u"]
        K_v = dx*op["DxDx"]@q["v"] + dy*op["DyDy"]@q["v"]
        omega_q = \
             op["IDxy"]@(op["IDy"]@q["u"]) - alpha**2*dy*op["DyDxy"]@K_u \
            -op["IDxy"]@(op["IDx"]@q["v"]) + alpha**2*dx*op["DxDxy"]@K_v \
            +alpha*(-dx*op["IDy"]@(op["DxDxy"]@q["p"]) + dy*op["IDx"]@(op["DyDxy"]@q["p"]))
        
        return self.operator["inv_lump"]@(self.operator["inv_lump"]@omega_q)

    def compute_discrete_curl_involution_all(self, qall, alpha, dx,dy):
        """Computing the curl involution on all solutions (e.g. in time)"""
        if len(qall["u"].shape) == 1:
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
            disc_div = (self.operator["IDx_tilde"]@qs["u"][it,:] \
                + self.operator["IDy_tilde"]@qs["v"][it,:]) \
                / self.geom.dx[0]/self.geom.dx[1]
            div_norms[it] = np.linalg.norm(disc_div)/np.sqrt(self.n_dof_tot)
        
        return div_norms
    
    def compute_wrong_discrete_divergence_norms(self, qs):
        Nt = qs["u"].shape[0]
        div_norms = np.zeros(Nt)
        for it in range(Nt):
            disc_div = (self.operator["IDx"]@qs["u"][it,:] \
                + self.operator["IDy"]@qs["v"][it,:]) \
                / self.geom.dx[0]/self.geom.dx[1]
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
            u_bc_idxs = self.dirichlet_indexes["all"]##[:-1] 
            n_bc = len(u_bc_idxs)
            data = np.ones(n_bc)
            row = np.arange(n_bc)
            col = u_bc_idxs
            u_matrix_constraint = sp.coo_matrix((data, (row,col)), shape=(n_bc, 2*self.n_dof_tot))
            u_rhs_constraint = IC_vect["u"][u_bc_idxs]

            # V constraint
            v_bc_idxs = self.dirichlet_indexes["all"]##[:-1] 
            n_bc = len(v_bc_idxs)
            data = np.ones(n_bc)
            row = np.arange(n_bc)
            col = np.array(v_bc_idxs,dtype=np.int64)+self.n_dof_tot
            v_matrix_constraint = sp.coo_matrix((data, (row,col)), shape=(n_bc, 2*self.n_dof_tot))
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
                                    self.operator["DyM_tilde"]@source_p])

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
        #    rhs_constraint_div[FEM2D.dirichlet_indexes["all"]] = 0.0
        
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
            return np.sum((mass_oper@(u - u_ic))**2)
            #+ pen_div*np.sum((div_oper@u)**2) #+ pen_grad_div*np.sum((grad_div_oper@u-source_p_term)**2)
        def target_function_dir(u):
            return 2.0*mass_mass@(u - u_ic)
            #+ pen_grad_div*2.* grad_div_oper.T@( grad_div_oper@u-source_p_term)
        def target_function_hess(u):
            return 2.0*mass_mass
            #+pen_grad_div*2. * grad_div_grad_div

        # Different methods # trust-constr #SLSQP #"lagrange"

        u_guess = np.zeros_like(u_ic)
        if method == 'trust-constr':
            res = minimize(target_function, u_guess, method=method,
                           jac=target_function_dir, hess=target_function_hess,
                           constraints=[linear_constraint], tol=1e-10, #constr_violation=1e-10,
                           options={'verbose':3}) #trust-constr #SLSQP #constr_violation
            q_vec = dict()
            q_vec["u"] = res.x[:len(u)]
            q_vec["v"] = res.x[len(u):]
            q_vec["p"] = IC_vect["p"]
        elif method == 'SLSQP':
            my_constraint = dict()
            my_constraint["type"] = 'eq'
            my_constraint["fun"] = lambda x: linear_constraint.A@x - linear_constraint.lb
            #my_constraint["jac"] = lambda x: linear_constraint.A
            res = minimize(target_function, u_ic, method=method,
                           #jac=target_function_dir, hess=target_function_hess,
                           constraints=[my_constraint]) #trust-constr #SLSQP
            q_vec = dict()
            q_vec["u"] = res.x[:len(u)]
            q_vec["v"] = res.x[len(u):]
            q_vec["p"] = IC_vect["p"]
        elif method == 'lagrange':
            N_unkn = linear_constraint.A.shape[1]
            N_constraints = linear_constraint.A.shape[0]
            AAA = hstack([mass_oper, linear_constraint.A.T])
            BBB =hstack([linear_constraint.A,  sp.csr_matrix((N_constraints, N_constraints), dtype=np.float64)])
            print(AAA.shape, BBB.shape)
            lagrange_mat = vstack([AAA,BBB])

            lagrange_rhs = np.concatenate([np.zeros(N_unkn), linear_constraint.lb- linear_constraint.A@u_ic])
            x = sp.linalg.spsolve(lagrange_mat, lagrange_rhs)

            q_vec = dict()
            q_vec["u"] = x[:len(u)]+u
            q_vec["v"] = x[len(u):2*len(u)]+v
            q_vec["p"] = IC_vect["p"]

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
                j1_cell_global = (j_cell + 1)*FEM1Dy.degree
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
                v_mario[j_cell_global+1:j1_cell_global+1, l] = np.linalg.solve(\
                    FEM1Dx.matrix["int_mat"][1:,1:],RHS)

        q_vec = dict()
        q_vec["u"] = self.mat_to_vect(u_mario)
        q_vec["v"] = self.mat_to_vect(v_mario)
        q_vec["p"] = IC_vect["p"]
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

        source["u"] = q_vec["v"]*(coriolis+cor_nu)  - problem.friction*q_vec["u"] + source["u"]
        source["v"] = -q_vec["u"]*(coriolis+cor_nu) - problem.friction*q_vec["v"] + source["v"]
        
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
                    if self.FEM1Dx.degree > 1:
                        dx = self.geom.xR[0]-self.geom.xx[0][j_cell]
                        Ixv[j_cell_global+1:j1_cell_global, l] = Ixv[j_cell_global,l] + \
                            dx*self.FEM1Dx.matrix["int_mat"][1:-1,:]@ \
                            np.concatenate([source_u[j_cell_global:j1_cell_global, l], [source_u[0,l]]])
                else:
                    dx = self.geom.xx[0][j_cell+1]-self.geom.xx[0][j_cell]
                    Ixv[j_cell_global+1:j1_cell_global+1, l] = Ixv[j_cell_global,l] + \
                        dx*self.FEM1Dx.matrix["int_mat"][1:,:]@ source_u[j_cell_global:j1_cell_global+1, l]        

        # Computing I^y S_v
        for l, x in enumerate(self.xx_dofs[0]):
            # Compute I^y S_v(x_l) m = int y_0^y_m S_v(x_l,y) dy
            for j_cell in range(self.geom.N_elem_dir[1]):
                j_cell_global = j_cell*self.FEM1Dy.degree      
                j1_cell_global = (j_cell + 1)*self.FEM1Dy.degree

                if self.geom.BC[1] == 0 and (j_cell == self.geom.N_elem_dir[1] - 1):  # periodic in y
                    if self.FEM1Dy.degree > 1:
                        dy = self.geom.xR[1]-self.geom.xx[1][j_cell]
                        Iyu[l, j_cell_global+1:j1_cell_global ] = Iyu[l,j_cell_global] + \
                            dy*self.FEM1Dy.matrix["int_mat"][1:-1,:]@ \
                            np.concatenate([source_v[l, j_cell_global:j1_cell_global], [source_v[l,0]]])
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
            p_aux_y[i,:] = p_mat[0,:] - p0 - Ixv[0,:]

        p_aux_x = np.zeros_like(Ixv)
        for j in range(len(self.xx_dofs[1])):
            p_aux_x[:,j] = p_mat[:,0] - p0 - Iyu[:,0]

        # p_aux_y = np.zeros_like(Ixv)
        # for i in range(len(self.xx_dofs[0])):
        #     p_aux_y[i,:] = Iyu[0,:]-Ixv[0,:]

        # p_aux_x = np.zeros_like(Ixv)
        # for j in range(len(self.xx_dofs[1])):
        #     p_aux_x[:,j] = Ixv[:,0]-Iyu[:,0]

        p_new = p0 + a*(Ixv + p_aux_y)+(1-a)*(Iyu+p_aux_x)

        p_vec = self.mat_to_vect(p_new)
        
        return p_vec

