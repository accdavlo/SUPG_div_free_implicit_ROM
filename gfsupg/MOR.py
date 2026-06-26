import numpy as np
from gfsupg.solver import CartesianGeometry, FiniteElement1D, Scipy2DFEM
from gfsupg.solver import DeC, DeCSpaceTimeSUPGSolver, ImplicitEuler, ImplicitDec
from gfsupg.problem import *
from gfsupg.plotting import *

import matplotlib.pyplot as plt

class MOR:
    """MOR solver with SUPG stabilization.

    The solver advances the state `(u, v, p)` on a fixed structured mesh using MOR 
    and DeC iterations in time.
    """

    def __init__(self, problem, FEM2D, DeC, tol=1e-5, GF=True, stab="SUPG"):
        self.FEM2D   = FEM2D
        self.DeC     = DeC
        self.problem = problem
        
        self.solver = ImplicitEuler(self.problem, self.FEM2D, self.DeC, GF, stab, trick_second_der=False)
        self.solver.set_CFL(100.0)

        self.tol = tol
        if self.solver.GF:
            self.GF_string = "GF"
        else:
            self.GF_string = "noGF"    
    def run_offline(self, mu_offline, n_rb=None, load_sol=False):
        self.mu_offline = mu_offline
        # Generate (or read) the snapshots
        if not load_sol:
            us = np.zeros([self.FEM2D.n_dof_tot,len(self.mu_offline)*self.solver.Nt_save], dtype=np.float64)
            vs = np.zeros([self.FEM2D.n_dof_tot,len(self.mu_offline)*self.solver.Nt_save], dtype=np.float64)
            ps = np.zeros([self.FEM2D.n_dof_tot,len(self.mu_offline)*self.solver.Nt_save], dtype=np.float64)
            for idx_mu, mu_ in enumerate(mu_offline):
                self.problem.set_parameters(mu_)
                self.solver.set_ic()

                print("Computing GF-SUPG")
                qGF, _, _, _ , _  = self.solver.solve(save_sol=False, with_error=False)

                for i in np.arange(0,qGF["u"].shape[0],1):
                    us[:,i + idx_mu*self.solver.Nt_save] = qGF["u"][i,:]
                    vs[:,i + idx_mu*self.solver.Nt_save] = qGF["v"][i,:]
                    ps[:,i + idx_mu*self.solver.Nt_save] = qGF["p"][i,:]
        

            np.savez(f"snapshots_offline_{self.problem.name}_{self.GF_string}.npz", us=us, vs=vs, ps=ps, mu_offline=mu_offline, n_dof_x=self.FEM2D.n_dof_dir[0], n_dof_y=self.FEM2D.n_dof_dir[1])
        else:
            inputfile = np.load(f"snapshots_offline_{self.problem.name}_{self.GF_string}.npz")
            us = inputfile['us']
            vs = inputfile['vs']
            ps = inputfile['ps']

        # Compute SVD
        print("Computing the SVD")
        self.basis_all = {}
        self.Sigma_svd = dict()
        self.basis_all["u"], self.Sigma_svd["u"], _ = np.linalg.svd(us, full_matrices=False)
        self.basis_all["v"], self.Sigma_svd["v"], _ = np.linalg.svd(vs, full_matrices=False)
        self.basis_all["p"], self.Sigma_svd["p"], _ = np.linalg.svd(ps, full_matrices=False)
        #plt.semilogy(self.Sigma_svd["u"],'o')
        #plt.show()

        self.truncate_basis(n_rb=n_rb, tol=self.tol)

        # Check tolerance to assemble reduced basis
        print("Computing the reduced basis")

    def truncate_basis(self, n_rb=None, tol=None):
        self.n_rb = dict()
        self.basis = dict()
        if tol is not None:
            self.tol = tol
        if n_rb is None:
            for var in self.problem.vars:
                sum_SVD_curr = np.sum(self.Sigma_svd[var])
                partial_sum_SVD_curr = 0.0
                self.n_rb[var] = 1
                while partial_sum_SVD_curr <= (1.0 - self.tol)*sum_SVD_curr:
                    partial_sum_SVD_curr += self.Sigma_svd[var][self.n_rb[var] - 1]
                    self.n_rb[var] += 1
                self.basis[var] = self.basis_all[var][:,0:self.n_rb[var]]
        else:
            for var in self.problem.vars:
                self.n_rb[var] = n_rb[var]
                self.basis[var] = self.basis_all[var][:,0:self.n_rb[var]]
        print("Number of reduced basis for u:", self.n_rb["u"])
        print("Number of reduced basis for v:", self.n_rb["v"])
        print("Number of reduced basis for p:", self.n_rb["p"])
        self.FEM2D.build_matrices_MOR(self.basis, self.n_rb)


    def run_online(self, params, compute_residuals=False):
        self.problem.set_parameters(params)
        self.solver.set_ic()

        print("")
        print("Computing GF-SUPG MOR")
        self.qGF_MOR, self.ttGF_MOR, self.comp_timeGF_MOR, self.error_MOR, _  = self.solver.solve_MOR(basis=self.basis, save_sol = True, with_error = True)
        
        # Check residuals
        if compute_residuals:
            sources = self.FEM2D.compute_sources(self.solver.ic_vect, self.problem)
            self.res = dict()
            self.res_rb = dict()
            for var in self.problem.vars:
                self.res[var] = np.zeros_like(self.ttGF_MOR)
                self.res_rb[var] = np.zeros_like(self.ttGF_MOR)
            for idx, t in enumerate(self.ttGF_MOR):
                qGF_mor_curr = dict()
                for var in self.problem.vars:
                    qGF_mor_curr[var] = self.qGF_MOR[var][idx,:]
                _, _, res_tmp, res_rb_tmp = self.FEM2D.compute_GF_residual_MOR(qGF_mor_curr, sources, self.basis)
                for var in self.problem.vars:
                    self.res[var][idx] = res_tmp[var]
                    self.res_rb[var][idx] = res_rb_tmp[var]
            return self.qGF_MOR, self.ttGF_MOR, self.comp_timeGF_MOR, self.error_MOR, self.res, self.res_rb
        return self.qGF_MOR, self.ttGF_MOR, self.comp_timeGF_MOR, self.error_MOR