import numpy as np
from gfsupg.solver import CartesianGeometry, FiniteElement1D, Scipy2DFEM
from gfsupg.solver import DeC, DeCSpaceTimeSUPGSolver
from gfsupg.problem import *
from gfsupg.plotting import *

import matplotlib.pyplot as plt

class MOR:
    """MOR solver with SUPG stabilization.

    The solver advances the state `(u, v, p)` on a fixed structured mesh using MOR 
    and DeC iterations in time.
    """

    def __init__(self, problem, FEM2D, DeC, solver, tol=1e-5, GF=True, stab="SUPG", variable_split="u,v,p", only_final_time=False):
        """Initialize the MOR solver.
        
        Args:
            problem: The problem instance.
            FEM2D: The 2D finite element method instance.
            DeC: The DeC instance.
            solver: The time integrator instance.
            tol: The tolerance for the reduced basis truncation.
            GF: Whether to use the Galerkin formulation.
            stab: The stabilization method to use.
            variable_split: The variable split to use for the reduced basis. #"u,v,p" or "uv,p"
            only_final_time: Whether to only save the final time step.
        """
        self.FEM2D   = FEM2D
        self.DeC     = DeC
        self.problem = problem
        self.variable_split = variable_split
        self.only_final_time = only_final_time
        if self.variable_split == "u,v,p":
            self.vars = ("u", "v", "p")
        elif self.variable_split == "uv,p":
            self.vars = ("uv", "p")

        self.solver = solver

        self.tol = tol

        if self.solver.GF:
            self.GF_string = "GF"
        else:
            self.GF_string = "noGF"    

        # Dirichlet data are imposed through q = q_D + V q_rb.  The lifting
        # q_D contains the prescribed boundary values and V is made zero on
        # the constrained degrees of freedom in ``truncate_basis``.
        self.dirichlet_lifting = None
        self.lifting_residual = None

    @property
    def has_dirichlet(self):
        return self.problem.dirichlet is not None

    def set_dirichlet_lifting(self, state):
        """Build the static FOM lifting from the current Dirichlet data.

        The FOM solver currently prescribes the initial boundary values for
        the whole simulation.  We use the same convention in the ROM.  This
        method must therefore be called after ``solver.set_ic()`` whenever
        the online parameters change.
        """
        if not self.has_dirichlet:
            self.dirichlet_lifting = None
            return

        lifting = {var: np.zeros(self.FEM2D.n_dof_tot) for var in self.problem.vars}
        # for boundary, variables in self.problem.dirichlet.items():
        #     indexes = self.FEM2D.dirichlet_indexes[boundary]
        #     for var in variables:
        #         lifting[var][indexes] = state[var][indexes]

        for var in self.problem.vars:
            lifting[var][:] = state[var][:]

        self.dirichlet_lifting = lifting

    def homogenize(self, q_FOM):
        """Return a copy of a FOM state with its Dirichlet lifting removed."""
        if not self.has_dirichlet:
            return {var: values.copy() for var, values in q_FOM.items()}
        if self.dirichlet_lifting is None:
            raise RuntimeError("Dirichlet lifting has not been initialized")

        homogeneous = {}
        for var, values in q_FOM.items():
            if var not in self.problem.vars:
                continue
            homogeneous[var] = values.copy()
            if values.ndim == 1:
                homogeneous[var] -= self.dirichlet_lifting[var]
            else:
                homogeneous[var] -= self.dirichlet_lifting[var][None, :]
        return homogeneous

    # def _enforce_homogeneous_basis(self):
    #     """Make the reduced basis exactly zero at constrained FOM dofs."""
    #     if not self.has_dirichlet:
    #         return

    #     # Constrained rows of each basis ("uv" stacks u and v)
    #     rows = {var: [] for var in self.vars}
    #     for boundary, variables in self.problem.dirichlet.items():
    #         indexes = self.FEM2D.dirichlet_indexes[boundary]
    #         for var in variables:
    #             if var in rows:
    #                 rows[var].extend(indexes)
    #             elif self.variable_split == "uv,p" and var in ("u", "v"):
    #                 rows["uv"].extend(indexes if var == "u" else indexes + self.FEM2D.n_dof_tot)
    #     rows = {var: np.unique(np.asarray(idx, dtype=np.int64)) for var, idx in rows.items()}

    #     for var in self.vars:
    #         self.basis[var][rows[var], :] = 0.0

    #     # Removing the boundary rows can introduce round-off non-orthogonality.
    #     # QR restores the Euclidean orthonormality assumed by the projections.
    #     # If the basis contains (numerical) null-space modes of the snapshots, these are
    #     # concentrated on the constrained rows and the zeroed basis is ill-conditioned:
    #     # QR then re-introduces non-zero values there, which are removed again.
    #     for var in self.vars:
    #         self.basis[var], _ = np.linalg.qr(self.basis[var], mode="reduced")
    #         self.basis[var][rows[var], :] = 0.0

    def run_offline(self, mu_offline, n_rb=None, load_sol=False):
        self.mu_offline = mu_offline
        # Generate (or read) the snapshots
        if self.only_final_time:
            inputfile_name = os.path.join(self.problem.folderName, \
                                          f"snapshots_final_time_offline_{self.GF_string}_ord_\
                                          {self.solver.FEM2D.FEM1Dx.degree+1}_N_{self.FEM2D.geom.N_elem_dir[0]}.npz")
        else:
            inputfile_name = os.path.join(self.problem.folderName, \
                                          f"snapshots_offline_{self.GF_string}_ord_\
                                          {self.solver.FEM2D.FEM1Dx.degree+1}_N_{self.FEM2D.geom.N_elem_dir[0]}.npz")

        # Saved snapshot files from older runs contain non-homogeneous data and
        # cannot be reused safely for a constrained ROM.
        if self.has_dirichlet:
            load_sol = False

        # if file exists allow to load
        if load_sol and not os.path.exists(inputfile_name):
            load_sol = False

        if not load_sol:
            self.snapshots = dict()
            for var in self.problem.vars:
                if self.only_final_time:
                    self.snapshots[var] = np.zeros([self.FEM2D.n_dof_tot,len(self.mu_offline)], dtype=np.float64)
                else:
                    self.snapshots[var] = np.zeros([self.FEM2D.n_dof_tot,len(self.mu_offline)*self.solver.Nt_save], dtype=np.float64)
            for idx_mu, mu_ in enumerate(mu_offline):
                self.problem.set_parameters(mu_)
                self.solver.set_ic()
                self.set_dirichlet_lifting(self.solver.ic_vect)

                print("Computing GF-SUPG")
                qGF, _, _, _ , _  = self.solver.solve(save_sol=False, with_error=False)

                if self.only_final_time:
                    for var in self.problem.vars:
                        self.snapshots[var][:,idx_mu] = self.homogenize(qGF)[var][-1,:]
                else:
                    for i in np.arange(0,qGF["u"].shape[0],1):
                        for var in self.problem.vars:
                            self.snapshots[var][:,i + idx_mu*self.solver.Nt_save] = self.homogenize(qGF)[var][i,:]
        
            np.savez(inputfile_name, snapshots=self.snapshots, mu_offline=self.mu_offline, \
                     n_dof_x=self.FEM2D.n_dof_dir[0], n_dof_y=self.FEM2D.n_dof_dir[1])
        else:
            inputfile = np.load(inputfile_name, allow_pickle=True)
            self.snapshots = inputfile['snapshots'].item()

        self.compute_SVD(n_rb=n_rb)

    def compute_SVD(self, n_rb=None):
        """Compute the SVD of the snapshots and truncate the basis according 
           to the tolerance or the number of reduced basis specified by the user."""

        # Compute SVD
        print("Computing the SVD")
        self.basis_all = {}
        self.Sigma_svd = dict()

        if self.variable_split == "uv,p":
            self.snapshots["uv"] = np.vstack((self.snapshots["u"], self.snapshots["v"]))

        for var in self.vars:
            self.basis_all[var], self.Sigma_svd[var], _ = np.linalg.svd(self.snapshots[var], full_matrices=False)
            #plt.semilogy(self.Sigma_svd[var],'o')
            #plt.show()

        self.truncate_basis(n_rb=n_rb, tol=self.tol)

        # Check tolerance to assemble reduced basis
        print("Computed the reduced basis")

    def truncate_basis(self, n_rb=None, tol=None):
        self.n_rb = dict()
        self.basis = dict()
        if tol is not None:
            self.tol = tol
        if n_rb is None:
            for var in self.vars:
                sum_SVD_curr = np.sum(self.Sigma_svd[var])
                partial_sum_SVD_curr = 0.0
                self.n_rb[var] = 1
                while partial_sum_SVD_curr <= (1.0 - self.tol)*sum_SVD_curr:
                    partial_sum_SVD_curr += self.Sigma_svd[var][self.n_rb[var] - 1]
                    self.n_rb[var] += 1
                self.basis[var] = self.basis_all[var][:,0:self.n_rb[var]]
                print(f"Number of reduced basis for {var}:", self.n_rb[var])
        else:
            for var in self.vars:
                self.n_rb[var] = n_rb[var]
                self.basis[var] = self.basis_all[var][:,0:self.n_rb[var]]
                print(f"Number of reduced basis for {var}:", self.n_rb[var])

        # self._enforce_homogeneous_basis()
        self.FEM2D.build_matrices_MOR(self)

    def project_onto_ROM(self, q_FOM, apply_lifting=False):
        """Project a FOM state or forcing onto the reduced basis.

        ``apply_lifting`` is true for solution states and false for physical
        source terms.  A source is not a state and must never have q_D removed.
        """
        if apply_lifting:
            q_FOM = self.homogenize(q_FOM)
        q_ROM = dict()
        if q_FOM["p"].ndim==1:
            if self.variable_split == "uv,p":
                # New dict: the input (e.g. a source evaluated in place) is not modified
                q_FOM = {**q_FOM, "uv": np.concatenate((q_FOM["u"], q_FOM["v"]))}
            for var in self.vars:
                q_ROM[var] = self.basis[var].T@ q_FOM[var]
        else:
            for var in self.vars:
                n_times = q_FOM["p"].shape[0]
                q_ROM[var] = np.zeros([n_times, self.n_rb[var]], dtype=np.float64)

                if var == "uv":
                    for i in range(n_times):
                        q_ROM[var][i,:] = self.basis[var].T @ np.concatenate((q_FOM["u"][i,:], q_FOM["v"][i,:]))
                else:
                    for i in range(q_FOM[var].shape[0]):
                        q_ROM[var][i,:] = self.basis[var].T@ q_FOM[var][i,:]
        
        return q_ROM
    
    def reconstruct_from_ROM(self, q_ROM, apply_lifting=True):
        q_FOM = dict()
        if q_ROM["p"].ndim==1:
            for var in self.vars:
                q_FOM[var] = self.basis[var]@ q_ROM[var]
            if self.variable_split == "uv,p":
                q_FOM["u"] = q_FOM["uv"][:self.FEM2D.n_dof_tot]
                q_FOM["v"] = q_FOM["uv"][self.FEM2D.n_dof_tot:]
        else:
            for var in self.vars:
                if var == "uv":
                    q_FOM[var] = np.zeros([q_ROM[var].shape[0], 2*self.FEM2D.n_dof_tot], dtype=np.float64)
                else:
                    q_FOM[var] = np.zeros([q_ROM[var].shape[0], self.FEM2D.n_dof_tot], dtype=np.float64)
                for i in range(q_ROM[var].shape[0]):
                    q_FOM[var][i,:] = self.basis[var]@ q_ROM[var][i,:]
            if self.variable_split == "uv,p":
                q_FOM["u"] = q_FOM["uv"][:,:self.FEM2D.n_dof_tot]
                q_FOM["v"] = q_FOM["uv"][:,self.FEM2D.n_dof_tot:]

        if apply_lifting and self.has_dirichlet and (self.dirichlet_lifting is not None):
            for var in self.problem.vars:
                if q_FOM[var].ndim == 1:
                    q_FOM[var] += self.dirichlet_lifting[var]
                else:
                    q_FOM[var] += self.dirichlet_lifting[var][None, :]
        
        return q_FOM

    def run_online(self, params, compute_residuals=False):
        self.problem.set_parameters(params)
        self.solver.set_ic()
        self.set_dirichlet_lifting(self.solver.ic_vect)

        print("")
        print("Computing GF-SUPG MOR")
        self.qGF_MOR, self.ttGF_MOR, self.comp_timeGF_MOR, self.error_MOR, _  = \
            self.solver.solve_MOR(self, save_sol = True, with_error = True)
        
        # Check residuals
        if compute_residuals:
            sources = self.FEM2D.compute_sources(self.solver.ic_vect, self.problem)
            source_MOR = self.project_onto_ROM(sources, apply_lifting=False)
            self.res = dict()
            self.res_rb = dict()
            for var in self.problem.vars:
                self.res[var] = np.zeros_like(self.ttGF_MOR)
            for var in self.vars:
                self.res_rb[var] = np.zeros_like(self.ttGF_MOR)
            for idx, t in enumerate(self.ttGF_MOR):
                qGF_mor_curr = dict()
                for var in self.vars:
                    qGF_mor_curr[var] = self.qGF_MOR[var][idx,:]
                
                reconstructed = self.reconstruct_from_ROM(qGF_mor_curr, apply_lifting=True)
                _, res_tmp = self.FEM2D.compute_GF_residual(reconstructed, sources)
                _,res_rb_tmp = self.FEM2D.compute_GF_residual_MOR(self, qGF_mor_curr, source_MOR, self.basis) 
                for var in self.problem.vars:
                    self.res[var][idx] = res_tmp[var]
                for var in self.vars:
                    self.res_rb[var][idx] = res_rb_tmp[var]
            
            return self.qGF_MOR, self.ttGF_MOR, self.comp_timeGF_MOR, self.error_MOR, self.res, self.res_rb
        
        return self.qGF_MOR, self.ttGF_MOR, self.comp_timeGF_MOR, self.error_MOR