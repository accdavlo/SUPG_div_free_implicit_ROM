"""Core finite-element operators and DeC time integration for 2D acoustics.

This module provides the main DeC space-time SUPG/OSS solver
(`DeCSpaceTimeSUPGSolver`). The building blocks it relies on are in
their own modules, and are re-exported here.

- `geometry.py`          -> CartesianGeometry
- `fem1d.py`             -> FiniteElement1D
- `fem2d.py`             -> Scipy2DFEM
- `boundary.py`          -> boundary_index_dir, assemble_1D_sparse_matrix, Dirichlet_BC_set
- `dec.py`               -> DeC
- `sources_residuals.py` -> define_sources*, define_residuals*, define_GF_residuals*
- `stabilization.py`     -> SUPG_*_stabilization*, OSS_*_stabilization*
- `dec_step.py`          -> DeC_one_step, DeC_one_step_MOR
- `sparse_utils.py`      -> get_stencil_indexes, invert_lumped_matrix,
                             put_zero_row_in_csr, delete_row_in_coo
"""

import numpy as np
import time
import pickle, os

from .quadr import lagrange_basis, lagrange_basis_deriv, nodes_weights

from .fem import CartesianGeometry
from .fem import FiniteElement1D
from .fem import Scipy2DFEM
from .fem import boundary_index_dir, assemble_1D_sparse_matrix, Dirichlet_BC_set
from .dec import DeC
from .dec import (
    define_sources, define_sources_MOR, define_sources_MOR_uv_p,
    define_residuals, define_residuals_MOR, define_residuals_MOR_uv_p,
    define_GF_residuals, define_GF_residuals_MOR, define_GF_residuals_MOR_uv_p,
)
from .dec import (
    SUPG_stabilization, SUPG_stabilization_MOR, SUPG_stabilization_MOR_uv_p,
    SUPG_GF_stabilization, SUPG_GF_stabilization_MOR, SUPG_GF_stabilization_MOR_uv_p,
    OSS_stabilization, OSS_GF_stabilization,
    OSS_curl_stabilization, OSS_GF_curl_stabilization,
)
from .dec import DeC_one_step, DeC_one_step_MOR
from .sparse_utils import (
    get_stencil_indexes, invert_lumped_matrix,
    put_zero_row_in_csr, delete_row_in_coo,
)


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
        self.Nt_save = 10
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

    def set_second_derivative_operators_MOR(self):
        """If trick second der is used, then instead of the second derivative operator
        we set the composition of the first derivative operators with
        the inverse of the lumped mass matrix to match the kernels of the central part"""
        op = self.FEM2D.operator_MOR
        for var in self.problem.vars:
            for var_bis in self.problem.vars:
                if self.trick_second_der:
                    op[var][var_bis]["DxDx2"] = op[var][var_bis]["DxI"]@op[var_bis][var]["inv_lump"]@op[var][var_bis]["IDx"]
                    op[var][var_bis]["DyDy2"] = op[var][var_bis]["DyI"]@op[var_bis][var]["inv_lump"]@op[var][var_bis]["IDy"]
                    op[var][var_bis]["DxDx2_tilde"] = \
                        op[var][var_bis]["DxI"]@\
                        op[var_bis][var]["inv_lump"]@\
                        op[var][var_bis]["mass_tilde_y"]@\
                        op[var_bis][var]["inv_lump"]@\
                        op[var][var_bis]["IDx"]
                    op[var][var_bis]["DyDy2_tilde"] = \
                        op[var][var_bis]["DyI"]@\
                        op[var_bis][var]["inv_lump"]@\
                        op[var][var_bis]["mass_tilde_x"]@\
                        op[var_bis][var]["inv_lump"]@\
                        op[var][var_bis]["IDy"]
                else:
                    op[var][var_bis]["DxDx2"] = op[var][var_bis]["DxDx"]
                    op[var][var_bis]["DyDy2"] = op[var][var_bis]["DyDy"]
                    op[var][var_bis]["DxDx2_tilde"] = op[var][var_bis]["DxDx_tilde"]
                    op[var][var_bis]["DyDy2_tilde"] = op[var][var_bis]["DyDy_tilde"]

                op[var][var_bis]["DxDx3"] = op[var][var_bis]["DxI"]@op[var_bis][var]["inv_lump"]@op[var][var_bis]["IDx"]
                op[var][var_bis]["DyDy3"] = op[var][var_bis]["DyI"]@op[var_bis][var]["inv_lump"]@op[var][var_bis]["IDy"]


                op[var][var_bis]["DxDx3_tilde"] = \
                    op[var][var_bis]["DxI"]@\
                    op[var_bis][var]["inv_lump"]@\
                    op[var][var_bis]["mass_tilde_y"]@\
                    op[var_bis][var]["inv_lump"]@\
                    op[var][var_bis]["IDx"]
                op[var][var_bis]["DyDy3_tilde"] = \
                    op[var][var_bis]["DyI"]@\
                    op[var_bis][var]["inv_lump"]@\
                    op[var][var_bis]["mass_tilde_x"]@\
                    op[var_bis][var]["inv_lump"]@\
                    op[var][var_bis]["IDy"]

                op[var][var_bis]["ZxMy"] =   op[var][var_bis]["DxDx"]-op[var][var_bis]["DxDx3"]
                op[var][var_bis]["ZyMx"] =   op[var][var_bis]["DyDy"]-op[var][var_bis]["DyDy3"]

                op[var][var_bis]["ZxMy_tilde"] =   op[var][var_bis]["DxDx_tilde"]-op[var][var_bis]["DxDx3_tilde"]
                op[var][var_bis]["ZyMx_tilde"] =   op[var][var_bis]["DyDy_tilde"]-op[var][var_bis]["DyDy3_tilde"]

                op[var][var_bis]["DxZy_int"] =   op[var][var_bis]["DyDx_tilde"]-\
                                                    op[var][var_bis]["DyI"]@\
                                                    op[var_bis][var]["inv_lump"]@\
                                                    op[var][var_bis]["IDx_tilde"]

                op[var][var_bis]["DyZx_int"] =   op[var][var_bis]["DxDy_tilde"]-\
                                                    op[var][var_bis]["DxI"]@\
                                                    op[var_bis][var]["inv_lump"]@\
                                                    op[var][var_bis]["IDy_tilde"]

                op[var][var_bis]["My_tilde_Zx_int"] =  op[var][var_bis]["DxM_tilde"] - \
                            op[var][var_bis]["DxI"]@op[var_bis][var]["inv_lump"]@op[var][var_bis]["mass_tilde_tilde"]

                op[var][var_bis]["Mx_tilde_Zy_int"] =  op[var][var_bis]["DyM_tilde"]- \
                            op[var][var_bis]["DyI"]@op[var_bis][var]["inv_lump"]@op[var][var_bis]["mass_tilde_tilde"]

                op[var][var_bis]["Zx_int_My"] = op[var][var_bis]["DxI_tilde"]\
                            -op[var][var_bis]["DxI"]@op[var_bis][var]["inv_lump"]@op[var][var_bis]["mass_tilde_x"]
                op[var][var_bis]["Zy_int_Mx"] = op[var][var_bis]["DyI_tilde"]\
                            -op[var][var_bis]["DyI"]@op[var_bis][var]["inv_lump"]@op[var][var_bis]["mass_tilde_y"]
                
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
            If changed, uses other second-derivative-related operators.
        curl_stab_flag:
            Adds optional curl-based OSS stabilization for velocity components.

        Returns
        -------
        q_save, tt_save, comp_time, error, error_vertex
            Saved solution snapshots, saved times, wall time, and optional
            error arrays.
        """

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
                if ROM.variable_split == "u,v,p":
                    get_residual = define_GF_residuals_MOR
                elif ROM.variable_split =="uv,p":
                    get_residual = define_GF_residuals_MOR_uv_p
                if self.stab == "SUPG":
                    if ROM.variable_split == "u,v,p":
                        get_stabilization = SUPG_GF_stabilization_MOR
                    elif ROM.variable_split == "uv,p":
                        get_stabilization = SUPG_GF_stabilization_MOR_uv_p
            else:
                if ROM.variable_split == "u,v,p":
                    get_residual = define_residuals_MOR
                elif ROM.variable_split =="uv,p":
                    get_residual = define_residuals_MOR_uv_p
                if self.stab == "SUPG":
                    if ROM.variable_split == "u,v,p":
                        get_stabilization = SUPG_stabilization_MOR
                    elif ROM.variable_split =="uv,p":
                        get_stabilization = SUPG_stabilization_MOR_uv_p
        else:
            raise NotImplementedError("Equations %s not implemented in solve in DeCSpaceTimeSolver"%self.problem.equations)

        if stab_coeff is None:
            if self.stab == "SUPG":
                if self.FEM2D.FEM1Dx.degree <= 5:
                    al = 0.05  # 5*10**-self.FEM2D.FEM1Dx.degree
                else:
                    al = 0.02
            elif self.stab == "OSS":
                raise NotImplementedError("OSS not implemented for MOR solver")
        else:
            al = stab_coeff
        self.stab_coeff = al
        self.stab_curl_coeff = 1e-4

        # self.set_second_derivative_operators_MOR()

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
                for i in range(self.DeC.M_sub):
                    for var in self.problem.vars:
                        source_FOM[var] = self.FEM2D.evaluate_function(lambda x,y: self.problem.source[var](x,y,t+dt*self.DeC.beta[i]))

                    source = ROM.project_onto_ROM(source_FOM)
                    for var in ROM.vars:
                        sub_sources[var][i,:] = source[var]

            # Create space-separated strings of the max and min values for all keys in q
            max_str = "  ".join(f"{np.max(q[k]):1.3f}" for k in q)
            min_str = "  ".join(f"{np.min(q[k]):1.3f}" for k in q)

            # Print everything using a modern f-string
            print(f"Iteration {it:07d}, time {t:1.5f}, max vars {max_str} ,  min vars {min_str}", end="\r")

            for k in range(self.DeC.n_iter):
                # Update variables
                for var in ROM.vars:
                    q_prev[var][:,:] = q_now[var][:,:]

                # Compute L2 high order space time discretization of the residual
                # And update of q_now
                DeC_one_step_MOR(ROM, self.problem, self.DeC, self.FEM2D, dt, al,\
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
