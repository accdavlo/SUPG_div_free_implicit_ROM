"""1D finite-element operator assembly (mass/derivative matrices and stencils)"""

import numpy as np

from ..quadr import lagrange_basis, lagrange_basis_deriv, nodes_weights


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
        self.matrix_names = ("mass","lump_mass","eval_mat","int_mat", "mass_bar", \
                             "deriv_i","deriv_j","deriv_ij", \
                             "deriv_ij_tilde","der_int_tilde", \
                             "deriv_i_tilde","mass_int",\
                             "deriv_j_bar")

        self.matrix = {}
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
                    self.matrix["mass"][i,j]     += wq * phi_quad[i,iq]     * phi_quad[j,iq]
                    self.matrix["deriv_i"][i,j]  += wq * phi_der_quad[i,iq] * phi_quad[j,iq]
                    self.matrix["deriv_j"][i,j]  += wq * phi_quad[i,iq]     * phi_der_quad[j,iq]
                    self.matrix["deriv_ij"][i,j] += wq * phi_der_quad[i,iq] * phi_der_quad[j,iq]
            self.matrix["lump_mass"][i,i] = np.sum(self.matrix["mass"][i,:])

        for i in range(self.N_dof):            
            nodes_i   = self.nodes[i]*self.quad_nodes
            weights_i = self.nodes[i]*self.quad_weights
            for j in range(self.N_dof):
                basis_quad = lagrange_basis(self.nodes, nodes_i, j)
                for iq in range(len(self.quad_nodes)):
                    self.matrix["int_mat"][i,j] += weights_i[iq]*basis_quad[iq]
        
        for j in range(self.N_dof):
            basis_nodes = lagrange_basis(self.nodes, self.nodes, j)
            self.matrix["eval_mat"][:,j] = basis_nodes
        
        self.matrix["deriv_ij_tilde"] = self.matrix["deriv_ij"] @ self.matrix["eval_mat"]
        self.matrix["der_int_tilde"]  = self.matrix["deriv_j"]  @ self.matrix["int_mat"]
        self.matrix["deriv_i_tilde"]  = self.matrix["deriv_ij"] @ self.matrix["int_mat"]
        self.matrix["deriv_j_bar"]    = self.matrix["deriv_j"]  @ self.matrix["eval_mat"]
        self.matrix["mass_bar"]       = self.matrix["mass"]     @ self.matrix["eval_mat"]
        self.matrix["mass_int"]       = self.matrix["mass"]     @ self.matrix["int_mat"]


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

