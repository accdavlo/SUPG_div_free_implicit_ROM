"""Regression test and tolerance study for source terms in the GF--SUPG MOR solver.

The test uses ``SourceVortexTestCase`` on the periodic square (is_smaller=True) with the
source amplitude as parameter.  The POD is built from FOM solutions obtained
with three source amplitudes; the source itself is appended to the snapshot set,
so that with the complete basis it is representable exactly.

1. Complete basis (regression): the reduced source contributions must be the
   projection of the FOM ones (explicit residual and, for the implicit
   integrators, the implicit source vector) up to rounding.
2. Tolerance study (as in test_SUPG_MOR.py): for each POD tolerance, number of
   reduced modes, error w.r.t. the FOM and the exact solution, and speedup.

The settings are in the "Flags" section below.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from gfsupg.MOR import MOR

from gfsupg.problem import SourceVortexTestCase

from gfsupg.solver import (
    CartesianGeometry, FiniteElement1D, Scipy2DFEM, \
    DeC, DeCSpaceTimeSUPGSolver, \
    define_sources, \
    define_sources_MOR, define_sources_MOR_uv_p, \
    define_residuals, define_GF_residuals, \
    define_residuals_MOR, define_GF_residuals_MOR, \
    define_residuals_MOR_uv_p, define_GF_residuals_MOR_uv_p, \
    SUPG_stabilization, SUPG_GF_stabilization, \
    SUPG_stabilization_MOR, SUPG_GF_stabilization_MOR, \
    SUPG_stabilization_MOR_uv_p, SUPG_GF_stabilization_MOR_uv_p
)
from gfsupg.ImplicitSolvers import ImplicitEuler, ImplicitDec

from gfsupg.plotting import plot_sol, plot_all_sols

################### Flags and parameters ###############################
time_integrator = ImplicitEuler #DeCSpaceTimeSUPGSolver #ImplicitDec
GF = True
variable_split = "uv,p" #"u,v,p"
CFL = 100.0
n_cells = 30
degree = 2

offline_amplitudes = (0.0, 0.01, 0.02)
online_amplitude = 0.015
# Tolerances for the truncation of the reduced basis (convergence study of the MOR solver)
tols = np.array([1e-1, 1e-2, 1e-3, 1e-4])

load_sol = False
fom_tol = None # If not None, fail when the complete-basis ROM vs FOM final-time error exceeds it
plot = True

################### Test-specific routines (source representation and checks) ###############################
# Physical source only (FEM2D.compute_sources also adds Coriolis and friction terms)
def source_vector(problem, fem, time):
    """Evaluate the physical source at one time node in FOM variables."""
    return {
        var: fem.evaluate_function(lambda x, y, var=var: problem.source[var](x, y, time))
        for var in problem.vars
    }

# The source column makes the source exactly representable with the complete basis
def append_snapshots(storage, solution, source):
    """Append all saved FOM states and one source snapshot to a POD database."""
    for var in ("u", "v", "p"):
        storage[var].append(solution[var].T)
        storage[var].append(source[var][:, None])

# Error of the source representation in the reduced basis (also used in the tolerance study)
def source_representation(rom, source):
    """Relative error of the projection V V^T s of the source on the reduced basis."""
    source_rb = rom.project_onto_ROM(source,apply_lifting=False)
    source_rec = rom.reconstruct_from_ROM(source_rb,apply_lifting=False)
    return {f"source_representation_{var}": relative_norm(source_rec[var] - source[var], source[var])
            for var in ("u", "v", "p") if np.linalg.norm(source[var]) > 0.0}

# Explicit DeC: same routines as DeC_one_step and DeC_one_step_MOR, with a zero state
def check_explicit_source_residual(solver, rom, fem, source):
    """Compare the projected FOM source contribution to the explicit residual with the MOR one.

    With a zero state the Galerkin residual and the SUPG stabilization reduce to
    their source terms. The routines are chosen according to the GF flag and to
    the variable splitting, as in the solvers.
    """
    problem = solver.problem
    n_sub = solver.DeC.n_subNodes
    m = n_sub - 1 # last sub-node, i.e. integration over the whole time step
    theta_m = solver.DeC.theta[m, :]
    cor_nu = np.zeros(fem.n_dof_tot)
    args = (problem.c, fem.geom.dx_min, solver.stab_coeff, theta_m, 1.0) # dt is irrelevant with a zero state

    # FOM contribution, projected onto the reduced basis
    fom_residual, fom_stab = (define_GF_residuals, SUPG_GF_stabilization) if solver.GF \
                             else (define_residuals, SUPG_stabilization)
    q_zero = {var: np.zeros((n_sub, fem.n_dof_tot)) for var in problem.vars}
    sub_sources = {var: np.tile(source[var], (n_sub, 1)) for var in problem.vars}
    all_sources, gal, stab = ({var: np.empty(fem.n_dof_tot) for var in problem.vars} for _ in range(3))
    define_sources(all_sources, q_zero, sub_sources, theta_m, problem.coriolis, cor_nu, problem.friction)
    fom_residual(gal, q_zero, all_sources, m, fem.operator, *args)
    fom_stab(stab, q_zero, all_sources, m, fem.operator, *args)
    projected = rom.project_onto_ROM({var: gal[var] + stab[var] for var in problem.vars})

    # MOR contribution
    if rom.variable_split == "uv,p":
        rom_sources = define_sources_MOR_uv_p
        rom_residual, rom_stab = (define_GF_residuals_MOR_uv_p, SUPG_GF_stabilization_MOR_uv_p) if solver.GF \
                                 else (define_residuals_MOR_uv_p, SUPG_stabilization_MOR_uv_p)
    else:
        rom_sources = define_sources_MOR
        rom_residual, rom_stab = (define_GF_residuals_MOR, SUPG_GF_stabilization_MOR) if solver.GF \
                                 else (define_residuals_MOR, SUPG_stabilization_MOR)
    source_rb = rom.project_onto_ROM(source)
    q_zero_rb = {var: np.zeros((n_sub, rom.n_rb[var])) for var in rom.vars}
    sub_sources_rb = {var: np.tile(source_rb[var], (n_sub, 1)) for var in rom.vars}
    all_sources_rb, gal_rb, stab_rb = ({var: np.empty(rom.n_rb[var]) for var in rom.vars} for _ in range(3))
    rom_sources(fem.operator_MOR, all_sources_rb, q_zero_rb, sub_sources_rb, theta_m,
                problem.coriolis, cor_nu, problem.friction)
    rom_residual(gal_rb, q_zero_rb, all_sources_rb, m, fem.operator_MOR, *args)
    rom_stab(stab_rb, q_zero_rb, all_sources_rb, m, fem.operator_MOR, *args)

    return {f"explicit_residual_{var}": relative_norm(gal_rb[var] + stab_rb[var] - projected[var], projected[var])
            for var in rom.vars}

# Implicit integrators: same routines as implicitEuler_one_step for the source vector
def check_implicit_source_vector(solver, rom, fem, source):
    """Compare the projected FOM implicit source vector with the MOR one."""
    n_dof = fem.n_dof_tot
    weights = np.array([0.0, 1.0]) # implicit Euler on the sources, as in implicitEuler_one_step

    sub_sources_fom = {var: np.vstack((np.zeros(n_dof), source[var])) for var in ("u", "v", "p")}
    vect_fom = np.empty((1, 3*n_dof))
    solver.define_vector_sources_implicit(sub_sources_fom, vect_fom, weights,
                                          fem.operator, solver.stab_coeff, fem.geom.dx_min)
    # FOM vector is ordered (u, v, p)
    projected = rom.project_onto_ROM({var: vect_fom[0, i*n_dof:(i + 1)*n_dof] for i, var in enumerate(("u", "v", "p"))})

    source_rb = rom.project_onto_ROM(source)
    sub_sources_rom = {var: np.vstack((np.zeros_like(source_rb[var]), source_rb[var])) for var in rom.vars}
    vect_rom = np.empty((1, sum(rom.n_rb[var] for var in rom.vars)))
    solver.define_vector_sources_implicit_MOR(rom, sub_sources_rom, vect_rom, weights,
                                              fem.operator_MOR, solver.stab_coeff, fem.geom.dx_min)

    # MOR vector is ordered as rom.vars
    errors, start = dict(), 0
    for var in rom.vars:
        block = vect_rom[0, start:start + rom.n_rb[var]]
        errors[f"implicit_vector_{var}"] = relative_norm(block - projected[var], projected[var])
        start += rom.n_rb[var]
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
problem = SourceVortexTestCase(is_long=True, is_smaller=True)

FEM1Dx = FiniteElement1D(degree, "gaussLobatto", "gaussLobatto")
FEM1Dy = FiniteElement1D(degree, "gaussLobatto", "gaussLobatto")
geom = CartesianGeometry(problem.xL, problem.xR, np.array([n_cells, n_cells]), problem.geometry_folder, BC=problem.BC)
FEM2D = Scipy2DFEM(geom, FEM1Dx, FEM1Dy, folder=problem.folderName)
dec = DeC(2, 2, "gaussLobatto")

################### Set the time integrator ###############################
solver = time_integrator(problem, FEM2D, dec, GF=GF, stab="SUPG", trick_second_der=False)
solver.set_CFL(CFL)
is_implicit = isinstance(solver, ImplicitEuler) # ImplicitDec derives from ImplicitEuler

####################### Perform the offline phase #########################
# Linear dependence on the amplitude means that the offline amplitudes also span
# the online one, while the source snapshots guarantee that the source itself
# is representable by the complete reduced basis.
snapshots_file = os.path.join(problem.folderName,
                              f"snapshots_sources_{time_integrator.__name__}_{'GF' if GF else 'noGF'}_deg{degree}"
                              f"_N{n_cells:04d}_CFL{CFL:g}"
                              f"_amp{'_'.join(f'{a:g}' for a in offline_amplitudes)}.npz")
MOR_instance = MOR(problem, FEM2D, dec, solver, GF=GF, stab="SUPG", variable_split=variable_split)
if load_sol and os.path.exists(snapshots_file):
    print("Loading the snapshots from", snapshots_file)
    with np.load(snapshots_file) as data:
        MOR_instance.snapshots = {var: data[var] for var in problem.vars}
else:
    snapshots = {var: [] for var in problem.vars}
    for amplitude in offline_amplitudes:
        problem.set_parameters([amplitude])
        solver.set_ic()
        q_fom, _, _, _, _ = solver.solve(save_sol=None, with_error=False)
        append_snapshots(snapshots, q_fom, source_vector(problem, FEM2D, 0.0))
    MOR_instance.snapshots = {var: np.hstack(columns) for var, columns in snapshots.items()}
    np.savez(snapshots_file, **MOR_instance.snapshots)

############## Compute the reference FOM solution for the online parameters #####################
problem.set_parameters([online_amplitude])
solver.set_ic()
qGF, ttGF, comp_timeGF_FOM, error_FOM, _ = solver.solve(save_sol=None, with_error=True)
source = source_vector(problem, FEM2D, 0.0)

####################### Complete basis: regression checks #########################
# Retaining all modes turns this into a direct consistency test of the
# projected source assembly rather than a test of POD compression.
n_snapshot_columns = MOR_instance.snapshots["p"].shape[1]
# All the snapshot columns are retained ("uv" has 2*n_dof rows). Unlike in test_SUPG_MOR_dirichlet.py,
# where the basis is restricted to the numerical rank of the snapshots, the numerical null-space modes
# are harmless here: no constrained rows are removed from the basis, which stays orthonormal, and the
# source column remains exactly representable.
MOR_instance.compute_SVD(n_rb={var: min((2 if var == "uv" else 1)*FEM2D.n_dof_tot, n_snapshot_columns)
                               for var in MOR_instance.vars})

qGF_MOR, _, _, _, _ = solver.solve_MOR(MOR_instance, save_sol=None, with_error=False)
source_errors = source_representation(MOR_instance, source)
source_errors.update(check_explicit_source_residual(solver, MOR_instance, FEM2D, source))
if is_implicit:
    source_errors.update(check_implicit_source_vector(solver, MOR_instance, FEM2D, source))
full_final, full_max = compare_with_fom(MOR_instance, qGF_MOR, qGF)

print("\nTime integrator:", time_integrator.__name__, "| GF:", GF, "| variable split:", variable_split)
print("Complete basis, reduced dimensions:", {var: MOR_instance.n_rb[var] for var in MOR_instance.vars})
print("Source representation/projection errors:")
for name, value in source_errors.items():
    print(f"  {name:28s} {value:.3e}")
print("Relative Error MOR w.r.t. FOM (final time / max over saved times):")
for var in problem.vars:
    print(f"  {var:28s} {full_final[var]:.3e} / {full_max[var]:.3e}")

worst_source_error = max(source_errors.values())
if worst_source_error > 1e-10:
    raise AssertionError(f"MOR source assembly is inconsistent (max relative error {worst_source_error:.3e})")
if fom_tol is not None and max(full_final.values()) > fom_tol:
    raise AssertionError(f"Complete-basis ROM differs from the FOM (max relative error {max(full_final.values()):.3e})")

####################### Perform the online phase for each tolerance #########################
n_rb_tols = {var: np.zeros(len(tols), dtype=int) for var in MOR_instance.vars}
err_vs_FOM_tols = {var: np.zeros(len(tols)) for var in problem.vars}
err_tols = {var: np.zeros(len(tols)) for var in problem.vars}
source_repr_tols = np.zeros(len(tols))
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
    source_repr_tols[i] = max(source_representation(MOR_instance, source).values())
    speedup_tols[i] = comp_timeGF_FOM/comp_timeGF_MOR

print("\nTolerance study (relative errors w.r.t. exact: time average (as in the solvers))")
print(f"  Relative Error FOM (u, v, p): {error_FOM[0]:.3e}  {error_FOM[1]:.3e}  {error_FOM[2]:.3e}")
print("  tol       n_rb(" + ",".join(MOR_instance.vars) + ")" + \
      " "*max(1, 12 - 4*len(MOR_instance.vars)) + "src_repr   " + "  ".join(f"vsFOM_{var}   " for var in problem.vars) \
      + "  " + "  ".join(f"vsEx_{var}    " for var in problem.vars) + "  speedup")
for i, tol in enumerate(tols):
    row = f"  {tol:.1e}   " + " ".join(f"{n_rb_tols[var][i]:4d}" for var in MOR_instance.vars) + f"   {source_repr_tols[i]:.2e}   "
    row += "  ".join(f"{err_vs_FOM_tols[var][i]:.2e}  " for var in problem.vars) + "  "
    row += "  ".join(f"{err_tols[var][i]:.2e}  " for var in problem.vars)
    row += f"  {speedup_tols[i]:6.2f}"
    print(row)

####################### Post-processing #########################
if plot:
    # Figures are saved in <problem folder>/figures, tagged with integrator, GF and split
    fig_folder = os.path.join(problem.folderName, "figures")
    os.makedirs(fig_folder, exist_ok=True)
    fig_tag = f"{time_integrator.__name__}_{'GF' if GF else 'noGF'}_{variable_split.replace(',', '')}"
    def save_figure(fig, name):
        fig.savefig(os.path.join(fig_folder, f"{name}_{fig_tag}.pdf"), bbox_inches="tight")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for var in problem.vars:
        ax[0].loglog(tols, err_vs_FOM_tols[var], "o-", label=var)
        ax[1].loglog(tols, err_tols[var], "o-", label=var)
    ax[0].loglog(tols, source_repr_tols, "k--", label="source repr.")
    for var in MOR_instance.vars:
        ax[2].semilogx(tols, n_rb_tols[var], "o-", label=var)
    ax[0].set_title("Error w.r.t. FOM (final time)")
    ax[1].set_title("Error w.r.t. exact solution")
    ax[2].set_title("Number of reduced modes")
    for a in ax:
        a.set_xlabel("Tolerance")
        a.invert_xaxis()
        a.legend()
    fig.suptitle(f"Source vortex, {time_integrator.__name__}, GF = {GF}, split {variable_split}")
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

    # Source (only p is non-zero) and its representation in the reduced basis
    source_rb = MOR_instance.project_onto_ROM(source)
    source_p_rb = MOR_instance.basis["p"] @ source_rb["p"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    plot_sol(FEM2D, source["p"], axs[0], fig, levels=21)
    plot_sol(FEM2D, source_p_rb, axs[1], fig, levels=21)
    plot_sol(FEM2D, source_p_rb - source["p"], axs[2], fig, levels=21)
    axs[0].set_title("Source p")
    axs[1].set_title("Source p in the reduced basis")
    axs[2].set_title("Difference")
    fig.suptitle(f"Source representation, tol = {tols[-1]:g}")
    plt.tight_layout()
    save_figure(plt.gcf(), "source_representation")

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