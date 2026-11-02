"""Regression test and tolerance study for Dirichlet conditions in the GF--SUPG MOR solver
with a moving solution.

The test uses ``MovingSourceTestCase``: a Gaussian structure (velocity given by the gradient of
a Gaussian, pressure by its derivative along the background speed) translating with the background
speed, driven by a time-dependent source on the pressure equation, with Dirichlet conditions on the
whole boundary (boundary values fixed to the initial ones, as in the FOM).  The parameter is the
background speed.  Unlike the steady vortex of test_SUPG_MOR_dirichlet.py, the solution is transported
across the domain: the reduced space must represent a moving structure, and the exact solution is
time dependent.  The Dirichlet data are imposed through the static lifting q = q_D + V q_rb, with q_D
the initial state (MOR.py, solver.py).

The time-dependent source is represented in the reduced space as in test_SUPG_MOR_sources.py: the
source at the saved times is appended to the snapshots.  Its values on the constrained nodes, which
are negligible since the Gaussian is far from the boundary, are set to zero, so that the reduced
basis still vanishes there.

Open point: the explicit reduced solver (solve_MOR) currently sets the source to zero, so that the ROM
misses the forcing that sustains the transport.  With the present code the error w.r.t. the FOM is
therefore O(1) for every number of modes, while the Dirichlet checks below still pass: only the
comparison with the FOM reveals the problem.  With the source restored the error decreases with the
POD tolerance (see the report, section "A moving structure").

1. Complete basis (regression): the reduced basis vanishes on the boundary and the reconstructed
   solution satisfies the Dirichlet data at all saved times.
2. Tolerance study: for each POD tolerance, number of reduced modes, Dirichlet error, error w.r.t.
   the FOM and the exact solution, and speedup.  The Dirichlet data must be satisfied for every tolerance.

Only the explicit DeC solver is available for Dirichlet conditions in the MOR solvers.
The settings are in the "Flags" section below.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from gfsupg.MOR import MOR

from gfsupg.problem import MovingSourceTestCase

from gfsupg.solver import (
    CartesianGeometry, FiniteElement1D, Scipy2DFEM, \
    DeC, DeCSpaceTimeSUPGSolver
)

from gfsupg.plotting import plot_sol, plot_all_sols

################### Flags and parameters ###############################
GF = True # Global-flux formulation
variable_split = "uv,p" #"u,v,p"
CFL = 0.1
Nt_save = 100 # Saved times per trajectory: snapshots must resolve the transport
T_fin = 1.0 # The structure moves by T_fin*|background speed|
n_cells = 30
degree = 2

# Parameters (background speed) used to compute the snapshots and online
mu_offline = [[-0.1*s, 0.1*s] for s in np.linspace(0.5, 1.5, 6)]
online_params = [-0.10, 0.10] # s = 1.0, between two offline values (s = 1.1 is an offline value)
# Tolerances for the truncation of the reduced basis (convergence study of the MOR solver)
tols = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])

append_source = True # Append the (homogenized) source at the saved times to the snapshots
load_sol = False # Load the snapshots from file, if available (the file name contains all the settings they depend on)
dirichlet_tol = 1e-8 # Maximum admissible error on the Dirichlet data
rank_tol = 1e-8 # Complete basis: modes with singular value below rank_tol*s_0 are numerical null space (their round-off on the boundary grows like eps/rank_tol)
fom_tol = None # If not None, fail when the complete-basis ROM vs FOM final-time error exceeds it
plot = True

################### Test-specific routines (Dirichlet data and checks) ###############################
# FOM rows constrained by the Dirichlet conditions for a FOM ("u", "v", "p") or ROM ("uv") variable
def constrained_rows(problem, fem, var):
    """Indexes of the constrained degrees of freedom of var ("uv" stacks u and v as in the basis)."""
    rows = []
    for boundary, variables in problem.dirichlet.items():
        indexes = fem.dirichlet_indexes[boundary]
        if var == "uv":
            if "u" in variables:
                rows.append(indexes)
            if "v" in variables:
                rows.append(indexes + fem.n_dof_tot)
        elif var in variables:
            rows.append(indexes)
    return np.unique(np.concatenate(rows)) if rows else np.array([], dtype=int)

# Source at the saved times, with the constrained rows set to zero (snapshots stored as n_dof x n_times)
def homogeneous_source_snapshots(problem, fem, times, rows):
    """Physical source at the given times, vanishing on the constrained degrees of freedom."""
    snapshots = dict()
    for var in problem.vars:
        snapshots[var] = np.column_stack([fem.evaluate_function(lambda x, y, var=var, t=t: problem.source[var](x, y, t))
                                          for t in times])
        snapshots[var][rows[var], :] = 0.0
    return snapshots

# Check: the reconstructed solution must satisfy the prescribed boundary values
def dirichlet_errors(rom, q_rom, boundary_values, rows):
    """Maximum error on the Dirichlet data of the reconstructed solution at all saved times."""
    q_rec = rom.reconstruct_from_ROM(q_rom, apply_lifting=True)
    return {var: np.max(np.abs(q_rec[var][:, rows[var]] - boundary_values[var][None, :]))
            for var in ("u", "v", "p") if rows[var].size > 0}

# Check: the homogeneous reduced basis must vanish on the constrained degrees of freedom
def basis_on_boundary(rom, problem, fem):
    """Maximum value of the reduced basis functions on the constrained degrees of freedom."""
    errors = dict()
    for var in rom.vars:
        rows = constrained_rows(problem, fem, var)
        if rows.size > 0:
            errors[var] = np.max(np.abs(rom.basis[var][rows, :]))
    return errors

################### Generic utilities ###############################
# Relative error used by all the checks and ROM vs FOM comparisons
def relative_norm(error, reference):
    """Relative Euclidean norm of error w.r.t. reference (safe for a vanishing reference)."""
    return np.linalg.norm(error) / max(np.linalg.norm(reference), 1e-14)

# ROM vs FOM comparison on the reconstructed solution
def compare_with_fom(rom, q_rom, q_fom):
    """Relative ROM vs FOM errors at the final time and maximum over the saved times."""
    q_rec = rom.reconstruct_from_ROM(q_rom, apply_lifting=True)
    n_times = min(q_rec["p"].shape[0], q_fom["p"].shape[0])
    final = {var: relative_norm(q_rec[var][-1] - q_fom[var][-1], q_fom[var][-1]) for var in ("u", "v", "p")}
    max_in_time = {
        var: max(relative_norm(q_rec[var][i] - q_fom[var][i], q_fom[var][i]) for i in range(n_times))
        for var in ("u", "v", "p")
    }
    return final, max_in_time

################### Set the problem data ###############################
problem = MovingSourceTestCase()
problem.T_fin = T_fin

FEM1Dx = FiniteElement1D(degree, "gaussLobatto", "gaussLobatto")
FEM1Dy = FiniteElement1D(degree, "gaussLobatto", "gaussLobatto")
geom = CartesianGeometry(problem.xL, problem.xR, np.array([n_cells, n_cells]), problem.geometry_folder, BC=problem.BC)
FEM2D = Scipy2DFEM(geom, FEM1Dx, FEM1Dy, folder=problem.folderName)
dec = DeC(2, 2, "gaussLobatto")
rows = {var: constrained_rows(problem, FEM2D, var) for var in problem.vars}

################### Set the time integrator ###############################
solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec, GF=GF, stab="SUPG", trick_second_der=False)
solver.set_CFL(CFL)
solver.set_save_slabs(Nt_save)

####################### Perform the offline phase #########################
# Snapshots are homogenized with the lifting of their own parameter, so that
# they vanish on the Dirichlet boundary (snapshots stored as n_dof x n_snapshots).
snapshots_file = os.path.join(problem.folderName,
                              f"snapshots_dirichlet_moving_DeC_{'GF' if GF else 'noGF'}_deg{degree}"
                              f"_N{n_cells:04d}_CFL{CFL:g}_Nt{Nt_save}_T{T_fin:g}{'_src' if append_source else ''}"
                              f"_mu{'_'.join('-'.join(f'{x:g}' for x in mu) for mu in mu_offline)}.npz")
MOR_instance = MOR(problem, FEM2D, dec, solver, GF=GF, stab="SUPG", variable_split=variable_split)
if load_sol and os.path.exists(snapshots_file):
    print("Loading the snapshots from", snapshots_file)
    with np.load(snapshots_file) as data:
        MOR_instance.snapshots = {var: data[var] for var in problem.vars}
else:
    snapshots = {var: [] for var in problem.vars}
    for mu in mu_offline:
        problem.set_parameters(mu)
        solver.set_ic()
        MOR_instance.set_dirichlet_lifting(solver.ic_vect)
        q_fom, tt_fom, _, _, _ = solver.solve(save_sol=None, with_error=False)
        q_hom = MOR_instance.homogenize(q_fom)
        for var in problem.vars:
            snapshots[var].append(q_hom[var].T)
        if append_source:
            source = homogeneous_source_snapshots(problem, FEM2D, tt_fom, rows)
            for var in problem.vars:
                snapshots[var].append(source[var])
    MOR_instance.snapshots = {var: np.hstack(columns) for var, columns in snapshots.items()}
    np.savez(snapshots_file, **MOR_instance.snapshots)

############## Compute the reference FOM solution for the online parameters #####################
problem.set_parameters(online_params)
solver.set_ic()
qGF, ttGF, comp_timeGF_FOM, error_FOM, _ = solver.solve(save_sol=None, with_error=True)
boundary_values = {var: solver.ic_vect[var][rows[var]].copy() for var in problem.vars}

####################### Complete basis: regression checks #########################
# Complete basis = numerical rank of the snapshots: the null-space modes carry no
# information and, the snapshots being homogeneous, are concentrated on the boundary.
MOR_instance.compute_SVD()
MOR_instance.truncate_basis(n_rb={var: int(np.sum(MOR_instance.Sigma_svd[var] > rank_tol*MOR_instance.Sigma_svd[var][0])) for var in MOR_instance.vars})

qGF_MOR, _, _, _, _ = solver.solve_MOR(MOR_instance, save_sol=None, with_error=False)
bc_errors = dirichlet_errors(MOR_instance, qGF_MOR, boundary_values, rows)
basis_errors = basis_on_boundary(MOR_instance, problem, FEM2D)
full_final, full_max = compare_with_fom(MOR_instance, qGF_MOR, qGF)

# Transient w.r.t. the lifting: the ROM represents q - q_D, much smaller than q for the pressure
transient = {var: relative_norm(qGF[var][-1] - qGF[var][0], qGF[var][-1]) for var in problem.vars}

print("\nTime integrator: DeC | GF:", GF, "| variable split:", variable_split)
print("Complete basis, reduced dimensions:", {var: MOR_instance.n_rb[var] for var in MOR_instance.vars})
print("Relative size of the transient ||q(T) - q(0)||/||q(T)||:")
for var in problem.vars:
    print(f"  {var:28s} {transient[var]:.3e}")
print("Maximum value of the basis on the Dirichlet boundary:")
for var, value in basis_errors.items():
    print(f"  {var:28s} {value:.3e}")
print("Maximum Dirichlet error at saved times:")
for var, value in bc_errors.items():
    print(f"  {var:28s} {value:.3e}")
print("Relative Error MOR w.r.t. FOM (final time / max over saved times):")
for var in problem.vars:
    print(f"  {var:28s} {full_final[var]:.3e} / {full_max[var]:.3e}")

worst_bc_error = max(max(bc_errors.values()), max(basis_errors.values()))
if worst_bc_error > dirichlet_tol:
    raise AssertionError(f"Dirichlet condition not enforced (max error {worst_bc_error:.3e})")
if fom_tol is not None and max(full_final.values()) > fom_tol:
    raise AssertionError(f"Complete-basis ROM differs from the FOM (max relative error {max(full_final.values()):.3e})")

####################### Perform the online phase for each tolerance #########################
n_rb_tols = {var: np.zeros(len(tols), dtype=int) for var in MOR_instance.vars}
err_vs_FOM_tols = {var: np.zeros(len(tols)) for var in problem.vars}
err_tols = {var: np.zeros(len(tols)) for var in problem.vars}
bc_err_tols = np.zeros(len(tols))
speedup_tols = np.zeros(len(tols))

for i, tol in enumerate(tols):
    MOR_instance.truncate_basis(tol=tol)
    for var in MOR_instance.vars:
        n_rb_tols[var][i] = MOR_instance.n_rb[var]

    qGF_MOR, _, comp_timeGF_MOR, error_MOR, _ = solver.solve_MOR(MOR_instance, save_sol=None, with_error=True)
    final, _ = compare_with_fom(MOR_instance, qGF_MOR, qGF)
    for ivar, var in enumerate(problem.vars):
        err_vs_FOM_tols[var][i] = final[var]
        err_tols[var][i] = error_MOR[ivar]
    bc_err_tols[i] = max(dirichlet_errors(MOR_instance, qGF_MOR, boundary_values, rows).values())
    speedup_tols[i] = comp_timeGF_FOM/comp_timeGF_MOR

print("\nTolerance study (relative errors; w.r.t. exact: time average as in the solvers)")
print(f"  Relative Error FOM (u, v, p): {error_FOM[0]:.3e}  {error_FOM[1]:.3e}  {error_FOM[2]:.3e}")
print("  tol       n_rb(" + ",".join(MOR_instance.vars) + ")" + " "*max(1, 12 - 4*len(MOR_instance.vars)) + "bc_err     "
      + "  ".join(f"vsFOM_{var}   " for var in problem.vars)
      + "  " + "  ".join(f"vsEx_{var}    " for var in problem.vars) + "  speedup")
for i, tol in enumerate(tols):
    row = f"  {tol:.1e}   " + " ".join(f"{n_rb_tols[var][i]:4d}" for var in MOR_instance.vars) + f"   {bc_err_tols[i]:.2e}   "
    row += "  ".join(f"{err_vs_FOM_tols[var][i]:.2e}  " for var in problem.vars) + "  "
    row += "  ".join(f"{err_tols[var][i]:.2e}  " for var in problem.vars)
    row += f"  {speedup_tols[i]:6.2f}"
    print(row)

# The lifting and the homogeneous basis must enforce the data for any truncation
if np.max(bc_err_tols) > dirichlet_tol:
    raise AssertionError(f"Dirichlet condition not enforced with a truncated basis (max error {np.max(bc_err_tols):.3e})")

####################### Post-processing #########################
if plot:
    # Figures are saved in <problem folder>/figures, tagged with integrator, GF and split
    fig_folder = os.path.join(problem.folderName, "figures")
    os.makedirs(fig_folder, exist_ok=True)
    fig_tag = f"DeC_{'GF' if GF else 'noGF'}_{variable_split.replace(',', '')}"
    def save_figure(fig, name):
        fig.savefig(os.path.join(fig_folder, f"{name}_{fig_tag}.pdf"), bbox_inches="tight")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for var in problem.vars:
        ax[0].loglog(tols, err_vs_FOM_tols[var], "o-", label=var)
        ax[1].loglog(tols, err_tols[var], "o-", label=var)
    for var in MOR_instance.vars:
        ax[2].semilogx(tols, n_rb_tols[var], "o-", label=var)
    ax[0].set_title("Error w.r.t. FOM (final time)")
    ax[1].set_title("Error w.r.t. exact solution")
    ax[2].set_title("Number of reduced modes")
    for a in ax:
        a.set_xlabel("Tolerance")
        a.invert_xaxis()
        a.legend()
    fig.suptitle(f"Moving structure with Dirichlet BCs, DeC, GF = {GF}, split {variable_split}")
    plt.tight_layout()
    save_figure(plt.gcf(), "convergence_tolerance")

    # Plot the final FOM solution
    fig = plot_all_sols(problem, FEM2D, qGF, -1, ttGF[-1], levels=21)
    save_figure(fig, "FOM_final_time")

    # Compare MOR (smallest tolerance), FOM and exact solution at the final time
    it = -1
    qGF_MOR_recon = MOR_instance.reconstruct_from_ROM(qGF_MOR, apply_lifting=True)
    q_exact = {var: FEM2D.evaluate_function(lambda x, y, var=var: problem.exact[var](x, y, ttGF[it]))
               for var in problem.vars}
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for col, (lab, q) in enumerate([("MOR", {var: qGF_MOR_recon[var][it,:] for var in problem.vars}),
                                    ("FOM", {var: qGF[var][it,:] for var in problem.vars}),
                                    ("Exact", q_exact)]):
        plot_sol(FEM2D, q["u"]**2 + q["v"]**2, axs[0,col], fig, levels=21)
        plot_sol(FEM2D, q["p"], axs[1,col], fig, levels=21)
        axs[0,col].set_title(f"{lab}: |u|^2")
        axs[1,col].set_title(f"{lab}: p")
    fig.suptitle(f"Final time, tol = {tols[-1]:g}")
    plt.tight_layout()
    save_figure(plt.gcf(), "comparison_final_time")

    # Pointwise MOR vs FOM difference at the final time: it must vanish on the Dirichlet boundary
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for k, var in enumerate(problem.vars):
        plot_sol(FEM2D, qGF_MOR_recon[var][it,:] - qGF[var][it,:], axs[k], fig, levels=21)
        axs[k].set_title(f"MOR - FOM: {var}")
    fig.suptitle(f"Final time, tol = {tols[-1]:g}")
    plt.tight_layout()
    save_figure(plt.gcf(), "difference_final_time")

    # Relative error MOR w.r.t. FOM in time (smallest tolerance)
    n_times = min(qGF_MOR_recon["p"].shape[0], qGF["p"].shape[0])
    plt.figure()
    for var in problem.vars:
        err_time = [relative_norm(qGF_MOR_recon[var][i] - qGF[var][i], qGF[var][i]) for i in range(n_times)]
        plt.semilogy(ttGF[:n_times], err_time, "o-", label=var)
    plt.xlabel("Time")
    plt.ylabel("Relative Error MOR w.r.t. FOM")
    plt.title(f"tol = {tols[-1]:g}")
    plt.legend()
    plt.tight_layout()
    save_figure(plt.gcf(), "error_in_time")
    plt.show()
