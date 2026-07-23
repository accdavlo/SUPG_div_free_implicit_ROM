from .solver import DeCSpaceTimeSUPGSolver, Dirichlet_BC_set, define_sources
import numpy as np
import scipy.sparse as sp
from scipy.sparse import hstack, vstack
import time
import pickle, os

def delete_row_in_coo_and_keep_diag_one(A, i):
    """Delete row `i` from a COO sparse matrix and return a new COO matrix."""

    idx_row = (A.row==i) & (A.col != i)
    diag_i = (A.row==i) & (A.col == i)
    
    new_data = np.copy(A.data)
    new_col = np.copy(A.col)
    new_row = np.copy(A.row)
    
    new_data[diag_i] = 1.0 
    new_data=np.delete(new_data,idx_row)
    new_col = np.delete(new_col, idx_row)
    #new_row[idx_higher_rows]-=1
    new_row= np.delete(new_row,idx_row)

    return sp.coo_matrix((new_data,(new_row,new_col)), shape = (A.shape[0],A.shape[1]))

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

    return sp.coo_matrix((new_data,(new_row,new_col)), shape = (A.shape[0],A.shape[1]))

class ImplicitEuler(DeCSpaceTimeSUPGSolver):
    def build_whole_q_vector(self, q:dict, vect_q:np.ndarray)->None:
        """
        Builds a whole vector stacking along dimension 0 the arrays in q.
        """
        curr_i = 0
        for var in q:
            size_q = q[var].shape[1]
            vect_q[:, curr_i:curr_i+size_q] = q[var]
            curr_i += size_q

    def split_whole_q_vector(self, q:dict, vect_q:np.ndarray):
        """
        Splits vect_q into dictionnary q.
        It supposes that vect_q is just q stacked along dimension 0.
        """
        curr_i = 0
        for var in q:
            size_q = q[var].shape[1]
            q[var][:,:] = vect_q[:, curr_i:curr_i+size_q]
            curr_i += size_q

    def build_whole_matrices(self,a,dx, dirichlet_BC = None):
        A_C=sp.csr_matrix((self.FEM2D.n_dof_tot*3,self.FEM2D.n_dof_tot*3))
        A_SU=sp.csr_matrix((self.FEM2D.n_dof_tot*3,self.FEM2D.n_dof_tot*3))
        Eps_CGFq=sp.csr_matrix((self.FEM2D.n_dof_tot*3,self.FEM2D.n_dof_tot*3))
        Eps_SUGFq=sp.csr_matrix((self.FEM2D.n_dof_tot*3,self.FEM2D.n_dof_tot*3))
        zero = sp.csr_matrix((self.FEM2D.n_dof_tot,self.FEM2D.n_dof_tot))

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
    
    def build_whole_matrices_MOR(self,ROM,a,dx, dirichlet_BC = None):
        n_rb = ROM.n_rb

        if ROM.variable_split == "u,v,p":
            A_C = np.vstack([np.hstack([self.FEM2D.operator_MOR["u"]["u"]["mass"], np.zeros((n_rb["u"], n_rb["v"])), np.zeros((n_rb["u"], n_rb["p"]))]), \
                        np.hstack([np.zeros((n_rb["v"], n_rb["u"])), self.FEM2D.operator_MOR["v"]["v"]["mass"], np.zeros((n_rb["v"], n_rb["p"]))]),\
                        np.hstack([np.zeros((n_rb["p"], n_rb["u"])), np.zeros((n_rb["p"], n_rb["v"])), self.FEM2D.operator_MOR["p"]["p"]["mass"]])])
            A_SU = a * dx * np.vstack([np.hstack([np.zeros((n_rb["u"], n_rb["u"])), np.zeros((n_rb["u"], n_rb["v"])), self.FEM2D.operator_MOR["u"]["p"]["DxI"]]), \
                        np.hstack([np.zeros((n_rb["v"], n_rb["u"])), np.zeros((n_rb["v"], n_rb["v"])), self.FEM2D.operator_MOR["v"]["p"]["DyI"]]),\
                        np.hstack([self.FEM2D.operator_MOR["p"]["u"]["DxI"], self.FEM2D.operator_MOR["p"]["v"]["DyI"], np.zeros((n_rb["p"], n_rb["p"]))])])
            Eps_CGFq = np.vstack([np.hstack([np.zeros((n_rb["u"], n_rb["u"])), np.zeros((n_rb["u"], n_rb["v"])), self.FEM2D.operator_MOR["u"]["p"]["IDx"]]), \
                            np.hstack([np.zeros((n_rb["v"], n_rb["u"])), np.zeros((n_rb["v"], n_rb["v"])), self.FEM2D.operator_MOR["v"]["p"]["IDy"]]),\
                            np.hstack([self.FEM2D.operator_MOR["p"]["u"]["IDx_tilde"], self.FEM2D.operator_MOR["p"]["v"]["IDy_tilde"], np.zeros((n_rb["p"], n_rb["p"]))])])
            Eps_SUGFq = a * dx * np.vstack([np.hstack([self.FEM2D.operator_MOR["u"]["u"]["DxDx_tilde"], self.FEM2D.operator_MOR["u"]["v"]["DxDy_tilde"], np.zeros((n_rb["u"], n_rb["p"])) ]), \
                                        np.hstack([self.FEM2D.operator_MOR["v"]["u"]["DyDx_tilde"], self.FEM2D.operator_MOR["v"]["v"]["DyDy_tilde"], np.zeros((n_rb["v"], n_rb["p"])) ]),\
                                        np.hstack([np.zeros((n_rb["p"], n_rb["u"])), np.zeros((n_rb["p"], n_rb["v"])), self.FEM2D.operator_MOR["p"]["p"]["DxDx"] + self.FEM2D.operator_MOR["p"]["p"]["DyDy"]])])
        elif ROM.variable_split == "uv,p":
            A_C = np.vstack([np.hstack([self.FEM2D.operator_MOR["uv"]["uv"]["mass"],  np.zeros((n_rb["uv"], n_rb["p"]))]), \
                        np.hstack([np.zeros((n_rb["p"], n_rb["uv"])), self.FEM2D.operator_MOR["p"]["p"]["mass"]])])
            A_SU = a * dx * np.vstack([np.hstack([np.zeros((n_rb["uv"], n_rb["uv"])), self.FEM2D.operator_MOR["uv"]["p"]["GradI"]]), \
                        np.hstack([self.FEM2D.operator_MOR["p"]["uv"]["DivI"], np.zeros((n_rb["p"], n_rb["p"]))])])
            Eps_CGFq = np.vstack([np.hstack([np.zeros((n_rb["uv"], n_rb["uv"])),self.FEM2D.operator_MOR["uv"]["p"]["IGrad"]]), \
                            np.hstack([self.FEM2D.operator_MOR["p"]["uv"]["IDiv_tilde"], np.zeros((n_rb["p"], n_rb["p"]))])])

            Eps_SUGFq = a * dx * np.vstack([np.hstack([self.FEM2D.operator_MOR["uv"]["uv"]["GradDiv_tilde"], np.zeros((n_rb["uv"], n_rb["p"])) ]), \
                        np.hstack([np.zeros((n_rb["p"], n_rb["uv"])), self.FEM2D.operator_MOR["p"]["p"]["DivGrad"] ])])    

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
        vect_source = np.empty((1, size_array))

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
        S = self.define_matrix_sources_implicit(self.problem.coriolis, cor_nu, self.problem.friction, \
                                                self.FEM2D.operator, self.stab_coeff, self.geom.dx_min, dirichlet_BC)
        vect_source = np.empty((1, size_array))
        vect_source_all = np.empty((q_now['u'].shape[0], size_array))

        
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
            self.implicitEuler_one_step(dt, A, B, S, \
                         q_prev, vect_q, q_now, vect_source, sub_sources = sub_sources,\
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
    
    def solve_MOR(self, ROM, stab_coeff=None, with_error=False, \
                  with_error_vertex=False, GF=None, CFL=None, \
                  save_sol=False, stab=None, curl_stab_flag=False):
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

        error, error_vertex, method_name, error_name, get_residual, get_stabilization, curl_stabilzation = \
                self.solver_set_parameters(stab_coeff, with_error, with_error_vertex, \
                                           GF, CFL, stab)

        dt_save = self.problem.T_fin/(self.Nt_save-2)

        q_save = dict() 

        it = 0
        t = 0.
        t_save  = 0
        it_save = 0
        L2 = dict()
        for var in ROM.vars: # ("u", "v", "p")
            L2[var] = np.zeros(self.FEM2D.n_dof_rb[var])
            q_save[var] = np.zeros((self.Nt_save, self.FEM2D.n_dof_rb[var]))
        tt_save = np.zeros(self.Nt_save)

        q_prev = dict()
        q_now  = dict()
        q      = dict()
        ic_ROM = ROM.project_onto_ROM(self.ic_vect)
        for var in ROM.vars:
            q_prev[var] = np.zeros((2, self.FEM2D.n_dof_rb[var]))
            q_now[var]  = np.zeros((2, self.FEM2D.n_dof_rb[var]))
            for i in range(2):
                q_now[var][i,:] = ic_ROM[var]


        size_array = sum(np.array([q_prev[k].shape[1] for k in ROM.vars]))            
        vect_q = np.empty((2, size_array))

        for var in ROM.vars:
            q_save[var][it_save,:] = q_now[var][-1,:]
        tt_save[it_save] = t

        source_FOM = dict()
        if self.problem.source is not None:
            for var in self.problem.vars:
                source_FOM[var] =  self.FEM2D.evaluate_function(lambda x,y: self.problem.source[var](x,y,0.))
        else:
            for var in self.problem.vars:
                source_FOM[var] = self.FEM2D.evaluate_function(lambda x,y: 0.)
        
        source = ROM.project_onto_ROM(source_FOM)

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
        for ivar, var in enumerate(ROM.vars):
            sub_sources[var] = np.zeros_like(q_now[var])


        tic = time.time()

        #Define big matrices
        A, B = self.build_whole_matrices_MOR(ROM,self.stab_coeff, self.geom.dx_min, dirichlet_BC)

        while (t<self.problem.T_fin and it<self.Nt_max):
            # Set dt
            dt =  self.CFL* self.geom.dx_min#self.CFL * self.problem.max_dt(q, self.geom.dx)
            c = self.problem.c

            # Initialize variables
            for ivar, var in enumerate(ROM.vars):
                for i in range(self.DeC.M_sub):
                    q_now[var][i,:] = q_now[var][-1,:] # previous timestep last update
            for i in range(self.DeC.M_sub+1):
                if self.problem.source is not None:
                    for var in self.problem.vars:
                        source_FOM[var] =  self.FEM2D.evaluate_function(lambda x,y: self.problem.source[var](x,y,t+dt*self.DeC.beta[i]))
                    source_ROM = ROM.project_onto_ROM(source_FOM)
                    for ivar, var in enumerate(ROM.vars): 
                        sub_sources[var][i,:] = source_ROM[var]

            # Create space-separated strings of the max and min values for all keys in q
            max_str = "  ".join(f"{np.max(q[k]):1.3f}" for k in q)
            min_str = "  ".join(f"{np.min(q[k]):1.3f}" for k in q)

            # Print everything using a modern f-string
            print(f"Iteration {it:07d}, time {t:1.5f}, max vars {max_str} ,  min vars {min_str}", end="\r")

            # Update variables
            for var in ROM.vars:
                q_prev[var][:,:] = q_now[var][:,:]

            # Compute L2 high order space time discretization of the residual
            # And update of q_now
            self.implicitEuler_one_step(dt, A, B, \
                         q_prev, vect_q, q_now, vect_source = np.empty((1, size_array)),
                         vect_sources_all = np.empty((2, size_array)),
                         sub_sources = sub_sources,\
                         coriolis_not_uni = cor_nu,\
                         dirichlet_BC=dirichlet_BC,
                         curl_stab_flag = curl_stab_flag)


            it+=1
            t=t+dt
            t_save+=dt
            if t_save > dt_save:
                it_save+=1
                t_save = 0.
                for var in ROM.vars:
                    q_save[var][it_save,:] = q_now[var][-1,:]
                tt_save[it_save] = t

        # Print everything 
        max_str = "  ".join(f"{np.max(q[k]):1.3f}" for k in q)
        min_str = "  ".join(f"{np.min(q[k]):1.3f}" for k in q)
        print(f"Iteration {it:07d}, time {t:1.5f}, max vars {max_str} ,  min vars {min_str}", end="\r")
            

        # Final step to save
        it_save+=1
        Nt_save = it_save
        for var in ROM.vars:
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
                if with_error:
                    reconstruct = ROM.reconstruct_from_ROM(q_now)
                    for ivar, var in enumerate(self.problem.vars):
                        ex = self.FEM2D.evaluate_function(lambda x,y: self. problem.exact[var](x,y,t))
                        error[ivar] += np.linalg.norm(reconstruct[var][-1,:]-ex)/(np.linalg.norm(ex) + 1e-10)*dt_tmp


        if save_sol is not None:
            q_final = dict()
            for var in ROM.vars:
                q_final[var] = q_save[var][-1]

            sol_to_save = [q_final, tt_save[-1], comp_time, error, error_vertex ]
            # Open a file and use dump()
            savefile_name = self.problem.folderName+"/final_sol_MOR_"+method_name+"_ord_%d_N_%04d.pkl"%(self.FEM2D.FEM1Dx.degree+1,self.FEM2D.geom.N_elem_dir[0])
            with open(savefile_name, 'wb') as file:
                # A new file will be created
                pickle.dump(sol_to_save, file)
        
        
        print("")

        return q_save, tt_save, comp_time, error, error_vertex

    def implicitEuler_one_step(self, dt, A, B, S, q_prev, vect_q, q_now,\
                           vect_source, sub_sources, \
                           dirichlet_BC = None, curl_stab_flag=False):
        """Perform one implicit Euler correction sweep over all sub-nodes.
    
        This routine assembles sources, residuals, and stabilization terms for each
        node and applies one implicit update using `inv_lump`.
        """
    
        # Compute L2 high order space time discretization of the residual
    
        stab_curl_coeff = self.stab_curl_coeff

        all_sources   = dict()
        for var in q_prev:
            all_sources[var] = np.empty(q_prev[var][0,:].shape)

        self.build_whole_q_vector(q_prev, vect_q)
        self.define_vector_sources_implicit(sub_sources, vect_source, self.DeC.theta[1,:],\
                                             self.FEM2D.operator, self.stab_coeff, self.geom.dx_min)
        
        #Define RHS
        RHS = A @ vect_q[0,:] + dt*vect_source.squeeze()
        if sp.issparse(A):
            vect_q[1,:] = sp.linalg.spsolve(A+dt*(B+S), RHS)
        else:
            vect_q[1,:] = np.linalg.solve(A+dt*(B+S), RHS)    
        #Just out of curiosity: we could try different solvers. 
        # spsolve is a LU solver, so not necessarily the best for big systems...
        #vect_q[1,:] = sp.linalg.bicgstab(A+dt*B, RHS) 
        #vect_q[1,:], _ = sp.linalg.gmres(A+dt*B, RHS) 
        #vect_q[1,:], _ = sp.linalg.gmres(A, (A-dt*B)@vect_q[0,:])
        #RHS = (A - 0.5*dt*B)@ vect_q[0,:] #Let's try Crank-Nicholson?
        ##vect_q[1,:], _ = sp.linalg.gmres((A+0.5*dt*B), RHS)
        #vect_q[1,:] = sp.linalg.spsolve(A+0.5*dt*B, RHS) 

        self.split_whole_q_vector(q_now, vect_q)

        if dirichlet_BC is not None:
            for bc_item in dirichlet_BC.keys():
                for m in range(2):
                    for var in dirichlet_BC[bc_item].vars:
                        q_now[var][m,dirichlet_BC[bc_item].indexes] =\
                            dirichlet_BC[bc_item].dirichlet_vector[var]

    def define_matrix_sources_implicit(self, cor, coriolis_not_uni, fric, op, al, dx_min, dirichlet_BC = None):
        """Build momentum and pressure source terms in semi-discrete form.
    
        Signs follow the convention used in `DeC_one_step`, where the assembled
        source terms are moved to the left-hand side of the residual equations.
        """
        ndof = self.FEM2D.n_dof_tot
        S = sp.csr_matrix((ndof*3,ndof*3))
        zero = sp.csr_matrix((ndof,ndof))
        cor_mat = (cor*sp.eye(ndof)+sp.diags(coriolis_not_uni))
    
        S = vstack([hstack([fric*op["mass_tilde_x"], -op["mass_tilde_x"]@cor_mat, zero]), \
                      hstack([op["mass_tilde_y"]@cor_mat, fric*op["mass_tilde_y"], zero]),\
                      hstack([al*dx_min*(fric*op["DxI_tilde"] + op["DyI_tilde"]@cor_mat), al*dx_min*(fric*op["DyI_tilde"] - op["DxI_tilde"]@cor_mat), zero])])
        
        if dirichlet_BC is not None:
            for bc_item in dirichlet_BC.keys():
                    for i in dirichlet_BC[bc_item].indexes:
                        S = put_zero_row_in_coo(S, i)
                        S = put_zero_row_in_coo(S, i + self.FEM2D.n_dof_tot)
                        S = put_zero_row_in_coo(S, i + 2*self.FEM2D.n_dof_tot)
    
        return S

    def define_vector_sources_implicit(self, sub_sources, vect_sources, theta_m, op, al, dx_min):
        all_sources = dict()
        all_sources['u'] = -theta_m@sub_sources['u']
        all_sources['v'] = -theta_m@sub_sources['v']
        all_sources['p'] = -theta_m@sub_sources['p']
        #self.build_whole_q_vector(all_sources, vect_sources)
        vect_sources[:] = -np.hstack([op["mass_tilde_x"]@all_sources['u'] + al*dx_min*op["DxM_tilde"]@all_sources['p'],
                                   op["mass_tilde_y"]@all_sources['v'] + al*dx_min*op["DyM_tilde"]@all_sources['p'],
                                   op["mass_tilde"]@all_sources['p']] + al*dx_min*(op["DxI_tilde"]@all_sources['u'] + op["DyI_tilde"]@all_sources['v']))


def define_residuals_implicit(galer_residuals, q_prev,m,op,c,dx_min , al, theta_m, dt):

    """Assemble Galerkin residuals (no stabilization) 
    for the standard (non-GF) formulation."""

    galer_residuals["u"][:] = op["mass"]@(q_prev["u"][m,:]-q_prev["u"][0,:])/dt\
        +c*   op["IDx"] @(theta_m @ q_prev["p"] )

    galer_residuals["v"][:] = op["mass"]@(q_prev["v"][m,:]-q_prev["v"][0,:])/dt\
        +c  * op["IDy"] @(theta_m @ q_prev["p"] )
        
    galer_residuals["p"][:] = op["mass"]@(q_prev["p"][m,:]-q_prev["p"][0,:])/dt\
        +c*op["IDx"]@(theta_m @ q_prev["u"] )\
        +c*op["IDy"]@(theta_m @ q_prev["v"] )

    return galer_residuals

def define_residuals_MOR_implicit(galer_residuals, q_prev,m,op,c,dx_min , al, theta_m, dt):

    """Assemble Galerkin residuals (no stabilization) 
    for the standard (non-GF) formulation."""

    galer_residuals["u"][:] = op["u"]["u"]["mass"]@(q_prev["u"][m,:]-q_prev["u"][0,:])/dt\
        +c*   op["u"]["p"]["IDx"] @(theta_m @ q_prev["p"] )

    galer_residuals["v"][:] = op["v"]["v"]["mass"]@(q_prev["v"][m,:]-q_prev["v"][0,:])/dt\
        +c  * op["v"]["p"]["IDy"] @(theta_m @ q_prev["p"] )
        
    galer_residuals["p"][:] = op["p"]["p"]["mass"]@(q_prev["p"][m,:]-q_prev["p"][0,:])/dt\
        +c*op["p"]["u"]["IDx"]@(theta_m @ q_prev["u"] )\
        +c*op["p"]["v"]["IDy"]@(theta_m @ q_prev["v"] )

    return galer_residuals

def define_GF_residuals_implicit(galer_residuals, q_prev,m,op,c,dx_min , al, theta_m, dt):

    """Assemble Galerkin residuals for the global-flux (GF) formulation when used with implicit DeC."""

    galer_residuals["u"][:] = op["mass"]@(q_prev["u"][m,:]-q_prev["u"][0,:])/dt\
        +c*op["IDx"]@(theta_m @ q_prev["p"])
    
    galer_residuals["v"][:] = op["mass"]@(q_prev["v"][m,:]-q_prev["v"][0,:])/dt\
        +c*op["IDy"]@(theta_m @ q_prev["p"])
    
    galer_residuals["p"][:] = op["mass"]@(q_prev["p"][m,:]-q_prev["p"][0,:])/dt\
        +c*op["IDx_tilde"] @(theta_m @ q_prev["u"] )\
        +c*op["IDy_tilde"] @(theta_m @ q_prev["v"] )

    return galer_residuals

def define_GF_residuals_MOR_implicit(galer_residuals, q_prev,m,op,c,dx_min , al, theta_m, dt):

    """Assemble Galerkin residuals for the global-flux (GF) formulation."""

    galer_residuals["u"][:] = op["u"]["u"]["mass"]@(q_prev["u"][m,:]-q_prev["u"][0,:])/dt\
        +c*op["u"]["p"]["IDx"]@(theta_m @ q_prev["p"] )
    
    galer_residuals["v"][:] = op["v"]["v"]["mass"]@(q_prev["v"][m,:]-q_prev["v"][0,:])/dt\
        +c*op["v"]["p"]["IDy"]@(theta_m @ q_prev["p"] )
    
    galer_residuals["p"][:] = op["p"]["p"]["mass"]@(q_prev["p"][m,:]-q_prev["p"][0,:])/dt\
        +c*op["p"]["u"]["IDx_tilde"] @(theta_m @ q_prev["u"] )\
        +c*op["p"]["v"]["IDy_tilde"] @(theta_m @ q_prev["v"] )

    return galer_residuals

class ImplicitDec(ImplicitEuler):
    def solver_set_parameters(self, stab_coeff=None, with_error=False, \
              with_error_vertex=False, GF=None, CFL=None, \
              stab=None, trick_second_der = False) :

        error, error_vertex, method_name, error_name, get_residual, get_stabilization, curl_stabilization = \
            super().solver_set_parameters(stab_coeff, with_error, with_error_vertex, GF, CFL, stab, trick_second_der)
        if self.problem.equations == "acoustics":
            if self.GF:
                get_residual = define_GF_residuals_implicit
            else:
                get_residual = define_residuals_implicit
        else:
            raise NotImplementedError("Equations %s not implemented in solve in ImplicitDec"%self.problem.equations)

        return error, error_vertex, method_name, error_name, get_residual, get_stabilization, curl_stabilization

    def solver_set_parameters_MOR(self, stab_coeff=None, with_error=False, \
              with_error_vertex=False, GF=None, CFL=None, \
              stab=None, trick_second_der = False) :

        error, error_vertex, method_name, error_name, get_residual, get_stabilization, curl_stabilization = \
            super().solver_set_parameters(stab_coeff, with_error, with_error_vertex, GF, CFL, stab, trick_second_der)
        if self.problem.equations == "acoustics":
            if self.GF:
                get_residual = define_GF_residuals_MOR_implicit
            else:
                get_residual = define_residuals_MOR_implicit
        else:
            raise NotImplementedError("Equations %s not implemented in solve in ImplicitDec"%self.problem.equations)

        return error, error_vertex, method_name, error_name, get_residual, get_stabilization, curl_stabilization

    def build_whole_q_vector(self, q:dict, vect_q:np.ndarray, m:int)->None:
        """
        Builds a whole vector stacking along dimension 0 the arrays in q.
        """
        curr_i = 0
        for var in q:
            size_q = q[var].shape[1]
            vect_q[0, curr_i:curr_i+size_q] = q[var][m, :]
            curr_i += size_q

    def split_whole_q_vector(self, q:dict, vect_q:np.ndarray, m:int):
        """
        Splits vect_q into dictionnary q.
        It supposes that vect_q is just q stacked along dimension 0.
        """
        curr_i = 0
        for var in q:
            size_q = q[var].shape[1]
            q[var][m,:] = vect_q[:, curr_i:curr_i+size_q]
            curr_i += size_q
    
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
            L2[var] = np.zeros((1,self.FEM2D.n_dof_tot))
            q_save[var] = np.zeros((self.Nt_save, self.FEM2D.n_dof_tot))
        tt_save = np.zeros(self.Nt_save)

        q_prev = dict()
        q_now  = dict()
        q      = np.zeros((len(self.problem.vars), self.FEM2D.n_dof_tot)) 
        for var in self.problem.vars:
            q_prev[var] = np.zeros((self.DeC.n_subNodes, self.FEM2D.n_dof_tot))
            q_now[var]  = np.zeros((self.DeC.n_subNodes, self.FEM2D.n_dof_tot))
            for i in range(self.DeC.n_subNodes): #range(self.DeC.n_subNodes):
                q_now[var][i,:] = self.ic_vect[var]

        size_array = sum(np.array([q_prev[k].shape[1] for k in self.problem.vars]))            
        vect_q = np.empty(size_array)
        vect_L2 = np.empty((1,size_array))

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
        _, B, L = self.build_whole_matrices(self.stab_coeff, self.geom.dx_min, dirichlet_BC)
        S = self.define_matrix_sources_implicit(self.problem.coriolis, cor_nu, self.problem.friction)
        
        #Enfore Dirichlet conditions
        if dirichlet_BC is not None:
            for bc_item in dirichlet_BC.keys():
                for m in range(self.DeC.n_subNodes):
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

            for k in range(self.DeC.n_iter):
                # Update variables
                for var in self.problem.vars:
                    q_prev[var][:,:] = q_now[var][:,:]

                # Compute L2 high order space time discretization of the residual
                # And update of q_now
                self.implicitDeC_one_step(dt, L, B, S, q_prev, vect_q, L2, vect_L2, q_now,\
                             sub_sources = sub_sources,\
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
                        error[ivar] += np.linalg.norm(q_save[var][it_save,:]-ex)/np.sqrt(np.linalg.norm(ex) + 1e-10)*dt_tmp
                    if with_error_vertex:
                        ex = self.FEM2D.evaluate_function_vertex(lambda x,y: self.problem.exact[var](x,y,t))
                        sol_vertex = self.FEM2D.from_vector_to_vertex(q_save[var][it_save,:])
                        error_vertex[ivar] += np.linalg.norm(sol_vertex-ex)/np.sqrt(np.linalg.norm(ex) + 1e-10)*dt_tmp

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

    def build_whole_matrices(self,a,dx, dirichlet_BC = None):
        A, B = super().build_whole_matrices(a,dx, dirichlet_BC)

        L=sp.csr_matrix((self.FEM2D.n_dof_tot*3,self.FEM2D.n_dof_tot*3))
        zero = sp.csr_matrix((self.FEM2D.n_dof_tot,self.FEM2D.n_dof_tot))

        L = vstack([hstack([self.FEM2D.operator["lump_mass"], zero, zero]), \
                      hstack([zero, self.FEM2D.operator["lump_mass"], zero]),\
                      hstack([zero, zero, self.FEM2D.operator["lump_mass"]])])
        
        if dirichlet_BC is not None:
            for bc_item in dirichlet_BC.keys():
                    for i in dirichlet_BC[bc_item].indexes:
                        L = delete_row_in_coo_and_keep_diag_one(L, i)
                        L = delete_row_in_coo_and_keep_diag_one(L, i + self.FEM2D.n_dof_tot)
                        L = delete_row_in_coo_and_keep_diag_one(L, i + 2*self.FEM2D.n_dof_tot)
    

        return A, B, L 

    def implicitDeC_one_step(self, dt, L, E, S, q_prev, vect_q, L2, vect_L2, q_now,\
                           vect_source, vect_sources_all, sub_sources, \
                            get_residual, get_stabilization, curl_stabilization,\
                           dirichlet_BC = None, curl_stab_flag=False):
        """Perform one DeC correction sweep over one sub-node.
    
        This routine assembles sources, residuals, and stabilization terms for each
        node and applies one implicit update using `inv_lump`.
        """
    
        # Compute L2 high order space time discretization of the residual
    
        c   = self.problem.c
        op  = self.FEM2D.operator
        #stab_curl_coeff = self.stab_curl_coeff

        #self.build_whole_q_vector(q_prev, vect_q)
        #all_sources   = dict()
        gal_residuals = dict()
        all_stabs     = dict()
        for var in self.problem.vars:
            #all_sources[var]   = np.empty(q_prev[var][0,:].shape)
            gal_residuals[var] = np.empty(q_prev[var][0,:].shape)
            all_stabs[var]     = np.empty(q_prev[var][0,:].shape)

        self.build_whole_q_vector(q_prev, vect_q)

        for m in range(1,self.DeC.n_subNodes):
            # Carefull with the signs! source_u,_v,_p are meant on the LHS, while the other source was on the RHS
            get_residual(gal_residuals, q_prev,m,op,c,self.FEM2D.geom.dx_min, self.stab_coeff, self.DeC.theta[m,:], dt)
            get_stabilization(all_stabs, q_prev,m,op,c,self.FEM2D.geom.dx_min, self.stab_coeff, self.DeC.theta[m,:], dt)

            for var in self.problem.vars:
                L2[var][:] = gal_residuals[var]+ all_stabs[var]

            #Can not include curl stabilization: how do we include source terms?
            #if curl_stab_flag:
            #    curl_stabilization(all_stabs,q_prev,all_sources,m,op,c,self.FEM2D.geom.dx_min, stab_curl_coeff, self.DeC.theta[m,:], dt)
            #    for var in ["u","v"]:
            #        L2[var][:] += all_stabs[var]
            
            if dirichlet_BC is not None:
                for bc_item in dirichlet_BC.keys():
                    for var in dirichlet_BC[bc_item].vars:
                        L2[var][0,dirichlet_BC[bc_item].indexes] =\
                            dirichlet_BC[bc_item].dirichlet_vector[var]

            #Define RHS
            self.build_whole_q_vector(L2, vect_L2, 0)
            self.build_whole_q_vector(sub_sources, vect_sources_all, m)
            vect_source = self.define_vector_sources_implicit(vect_sources_all, self.DeC.theta[m, :])
            vect_L2 += vect_source.squeeze()

            beta = self.DeC.beta[m]
            vect_q = sp.linalg.spsolve(-L/dt-beta*(E+S), vect_L2[0,:]) 

            self.split_whole_q_vector(q_now, vect_q, m)
            for var in self.problem.vars:
                q_now[var][m,:] += q_prev[var][m,:]

            if dirichlet_BC is not None:
                for bc_item in dirichlet_BC.keys():
                    for var in dirichlet_BC[bc_item].vars:
                        q_now[var][m,dirichlet_BC[bc_item].indexes] =\
                            dirichlet_BC[bc_item].dirichlet_vector[var]

    def solve_MOR(self, ROM, stab_coeff = None, with_error = False, \
                  with_error_vertex = False, GF = None, CFL = None, \
                  save_sol = False, stab = None, curl_stab_flag = False):
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
            If changed, uses other second-derivative-related operators. (not used for now)
        curl_stab_flag:
            Adds optional curl-based OSS stabilization for velocity components.

        Returns
        -------
        q_save, tt_save, comp_time, error, error_vertex
            Saved solution snapshots, saved times, wall time, and optional
            error arrays.
        """

        error, error_vertex, method_name, error_name, get_residual, get_stabilization, curl_stabilization = \
                self.solver_set_parameters_MOR(stab_coeff, with_error, with_error_vertex, \
                                           GF, CFL, stab)

        self.set_second_derivative_operators()

        dt_save = self.problem.T_fin/(self.Nt_save-2)

        q_save = dict() 

        it = 0
        t = 0.
        t_save  = 0.
        it_save = 0
        L2 = dict()
        for var in ROM.vars: # ("u", "v", "p")
            L2[var] = np.zeros(self.FEM2D.n_dof_rb[var])
            q_save[var] = np.zeros((self.Nt_save, self.FEM2D.n_dof_rb[var]))
        tt_save = np.zeros(self.Nt_save)

        q_prev = dict()
        q_now  = dict()
        q      = dict()
        ic_ROM = ROM.project_onto_ROM(self.ic_vect)
        for var in ROM.vars:
            q_prev[var] = np.zeros((self.DeC.n_subNodes, self.FEM2D.n_dof_rb[var]))
            q_now[var]  = np.zeros((self.DeC.n_subNodes, self.FEM2D.n_dof_rb[var]))
            for i in range(self.DeC.n_subNodes):
                q_now[var][i,:] = ic_ROM[var]

        for var in ROM.vars:
            q_save[var][it_save,:] = q_now[var][-1,:]
        tt_save[it_save] = t

        source_FOM = dict()
        if self.problem.source is not None:
            for var in self.problem.vars:
                source_FOM[var] = self.FEM2D.evaluate_function(lambda x,y: self.problem.source[var](x,y,0.))
        else:
            for var in self.problem.vars:
                source_FOM[var] = self.FEM2D.evaluate_function(lambda x,y: 0.)

        source = ROM.project_onto_ROM(source_FOM)

        if self.problem.coriolis_non_uniform is not None:
            cor_nu = self.FEM2D.evaluate_function(self.problem.coriolis_non_uniform)
        else:
            cor_nu = self.FEM2D.evaluate_function(lambda x,y: 0.)

        if self.problem.dirichlet is not None:
            raise NotImplementedError("Not implemented boundary condition for MOR solver")

        sub_sources = dict()
        for ivar, var in enumerate(ROM.vars):
            sub_sources[var] = np.zeros_like(q_now[var])


        tic = time.time()
        while (t<self.problem.T_fin and it<self.Nt_max):
            # Set dt
            dt =  self.CFL* self.geom.dx_min#self.CFL * self.problem.max_dt(q, self.geom.dx)
            c = self.problem.c

            # Initialize variables
            for ivar, var in enumerate(ROM.vars):
                q[var] = q_now[var][-1,:]
                for i in range(self.DeC.M_sub):
                    q_now[var][i,:] = q_now[var][-1,:] # previous timestep last update
            if self.problem.source is not None:
                for i in range(self.DeC.M_sub+1):
                    for var in self.problem.vars:
                        source_FOM[var] = self.FEM2D.evaluate_function(lambda x,y: self.problem.source[var](x,y,t+dt*self.DeC.beta[i]))

                    source = ROM.project_onto_ROM(source_FOM)
                    for var in ROM.vars:
                        sub_sources[var][i,:] = source[var]
            
            # Print everything 
            max_str = "  ".join(f"{np.max(q[k]):1.3f}" for k in q)
            min_str = "  ".join(f"{np.min(q[k]):1.3f}" for k in q)
            print(f"Iteration {it:07d}, time {t:1.5f}, max vars {max_str} ,  min vars {min_str}", end="\r")

            for k in range(self.DeC.n_iter):
                # Update variables
                for var in ROM.vars:
                    q_prev[var][:,:] = q_now[var][:,:]

                # Compute L2 high order space time discretization of the residual
                # And update of q_now
                # STILL EXPLICIT ROM!
                DeC_one_step_MOR(ROM,self.problem, self.DeC, self.FEM2D, dt, al,\
                                 self.stab_curl_coeff, q_prev, L2, q_now, sub_sources = sub_sources,\
                                 coriolis_not_uni = cor_nu,\
                                 get_residual=get_residual,\
                                 get_stabilization=get_stabilization,\
                                 curl_stabilization=None,\
                                 dirichlet_BC=None,
                                 curl_stab_flag = curl_stab_flag)
            
            for var in ROM.vars:
                q[var] = q_now[var][-1,:]

            it+=1
            t=t+dt
            t_save+=dt
            if t_save > dt_save:
                it_save+=1
                t_save = 0.
                for var in ROM.vars:
                    q_save[var][it_save,:] = q_now[var][-1,:]
                tt_save[it_save] = t

        #print("Iteration %07d, time %1.5f, max vars %1.3f  %1.3f  %1.3f ,  min vars %1.3f  %1.3f  %1.3f "%(it,t,\
        #           np.max(q["u"]),np.max(q["v"]),np.max(q["p"]),\
        #            np.min(q["u"]),np.min(q["v"]),np.min(q["p"])  ))
            

        # Final step to save
        it_save+=1
        Nt_save = it_save
        for var in ROM.vars:
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
                if with_error:
                    reconstruct = ROM.reconstruct_from_ROM(q_now)
                    for ivar, var in enumerate(self.problem.vars):
                        ex = self.FEM2D.evaluate_function(lambda x,y: self. problem.exact[var](x,y,t))
                        error[ivar] += np.linalg.norm(reconstruct[var][-1,:]-ex)/(np.linalg.norm(ex) + 1e-10)*dt_tmp


        if save_sol is not None:
            q_final = dict()
            for var in ROM.vars:
                q_final[var] = q_save[var][-1]

            sol_to_save = [q_final, tt_save[-1], comp_time, error, error_vertex ]
            # Open a file and use dump()
            savefile_name = self.problem.folderName+"/final_sol_MOR_"+method_name+"_ord_%d_N_%04d.pkl"%(self.FEM2D.FEM1Dx.degree+1,self.FEM2D.geom.N_elem_dir[0])
            with open(savefile_name, 'wb') as file:
                # A new file will be created
                pickle.dump(sol_to_save, file)
        
        
        print("")

        return q_save, tt_save, comp_time, error, error_vertex

