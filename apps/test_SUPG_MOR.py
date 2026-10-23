import time

import numpy as np
from gfsupg.solver import CartesianGeometry, FiniteElement1D, Scipy2DFEM
from gfsupg.solver import DeC, DeCSpaceTimeSUPGSolver
from gfsupg.ImplicitSolvers import ImplicitEuler, ImplicitDec
from gfsupg.problem import *
from gfsupg.plotting import *
from gfsupg.MOR import *

import matplotlib.pyplot as plt

################### Set the problem data ###############################
order = 3
order_space = order
order_time = 2

FEM1Dx = FiniteElement1D(order_space-1,"gaussLobatto","gaussLobatto")
FEM1Dy = FiniteElement1D(order_space-1,"gaussLobatto","gaussLobatto")
dec = DeC((order_time+1)//2,order_time,"gaussLobatto")

Ny = 40
Nx = 40
Ns = np.array([Nx,Ny], dtype=np.int32)

problem = SmoothVortexTestCaseParam(is_long=True)
# problem = ObliqueTestCase()
# problem = ShuVortexTestCaseParam()
# problem.T_fin = 100.0

geom = CartesianGeometry(problem.xL,problem.xR, Ns, problem.geometry_folder, BC=problem.BC)
FEM2D = Scipy2DFEM(geom, FEM1Dx, FEM1Dy, folder=problem.folderName)

################### Set the time integrator ###############################
time_integrator = ImplicitEuler # DeCSpaceTimeSUPGSolver # ImplicitDec
solver = time_integrator(problem, FEM2D, dec, GF=True, stab="SUPG", trick_second_der=False)
solver.set_CFL(500.0)

################### Set the MOR data ###############################
# Parameters used to compute the snapshots
coeff_exp = np.arange(1,11,1)
mu_offline = [[9.81, 0.45, coeff_exp_] for coeff_exp_ in coeff_exp]
# mu_offline = [[]]
# mu_offline = [[x0, y0, r0] for x0 in np.linspace(0.35,0.65,6) for y0 in np.linspace(0.35,0.65,6) for r0 in np.linspace(0.15,0.25,6)]
# x0_offline = np.array([0.3,0.4,0.5,0.6,0.7])
# y0_offline = np.array([0.3,0.4,0.5,0.6,0.7])
# r_offline = np.array([0.05,0.1,0.2,0.25])
# mu_offline = [[x0, y0, r0] for x0 in x0_offline for y0 in y0_offline for r0 in r_offline]

# Parameters of the online phase
online_params = [9.81, 0.45, 2.5]
# online_params = []
# problem.set_final_time(1.0)
# online_params = [0.47, 0.48, 0.21]

load_sol = True # Load the snapshots from file, if available
compute_residuals = True
# Tolerances for the truncation of the reduced basis (convergence study of the MOR solver)
tols = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])

####################### Perform the offline phase #########################
MOR_instance = MOR(problem, FEM2D, dec, solver, tol=tols[0], GF=True, stab="SUPG", variable_split="uv,p")
MOR_instance.run_offline(mu_offline, load_sol=load_sol)

############## Compute the reference FOM solution for the online parameters #####################
problem.set_parameters(online_params)
solver.set_ic()

tic = time.time()
qGF, ttGF, comp_timeGF, error, _  = solver.solve(save_sol=True, with_error=True)
comp_timeGF_FOM = time.time() - tic
print("Relative Error FOM u:", error[0])
print("Relative Error FOM v:", error[1])
print("Relative Error FOM p:", error[2])
print("")

# Plot the final FOM solution
plot_all_sols(problem, FEM2D, qGF, -1, ttGF[-1], levels=21)
plt.show()

####################### Perform the online phase for each tolerance #########################
n_rb_tols = dict()
for var in MOR_instance.vars:
    n_rb_tols[var] = np.zeros_like(tols)

err_tols = dict()
err_vs_FOM_tols = dict()
for var in problem.vars:
    err_tols[var] = np.zeros_like(tols)
    err_vs_FOM_tols[var] = np.zeros_like(tols)

it = -1 # Time index used for the comparison with the FOM
for i, tol in enumerate(tols):
    MOR_instance.truncate_basis(tol=tol)
    for var in MOR_instance.vars:
        n_rb_tols[var][i] = MOR_instance.n_rb[var]

    # run_online sets the parameters and the initial condition
    tic = time.time()
    MOR_instance.run_online(online_params, compute_residuals=compute_residuals)
    comp_timeGF_MOR = time.time() - tic

    print("Computation time FOM:", comp_timeGF_FOM)
    print("Computation time MOR:", comp_timeGF_MOR)
    print("Speedup:", comp_timeGF_FOM/comp_timeGF_MOR)
    if compute_residuals:
        # Left column: residuals of the reconstructed solution, right column: reduced residuals
        n_rows = max(len(problem.vars), len(MOR_instance.vars))
        fig, axs = plt.subplots(n_rows, 2, figsize=(12, 3*n_rows), squeeze=False)
        for ivar, var in enumerate(problem.vars):
            axs[ivar,0].plot(MOR_instance.ttGF_MOR, MOR_instance.res[var], label="Residuals FOM %s" % var)
            axs[ivar,0].legend()
        for ivar, var in enumerate(MOR_instance.vars):
            axs[ivar,1].plot(MOR_instance.ttGF_MOR, MOR_instance.res_rb[var], label="Residuals MOR %s" % var)
            axs[ivar,1].legend()
        fig.suptitle("Residuals, tol = %g" % tol)

    for ivar, var in enumerate(problem.vars):
        err_tols[var][i] = MOR_instance.error_MOR[ivar]
        print("Relative Error MOR %s:" % var, MOR_instance.error_MOR[ivar])
    print("")

    qGF_MOR_recon = MOR_instance.reconstruct_from_ROM(MOR_instance.qGF_MOR, apply_lifting=True)

    #plot_all_sols(problem, FEM2D, qGF_MOR_recon, it, MOR_instance.ttGF_MOR[it], levels=21)
    error_MOR_vs_FOM = np.zeros(len(problem.vars))
    for ivar, var in enumerate(problem.vars):
        error_MOR_vs_FOM[ivar] = np.linalg.norm(qGF_MOR_recon[var][it,:] - qGF[var][it,:])/np.linalg.norm(qGF[var][it,:])

        err_vs_FOM_tols[var][i] = error_MOR_vs_FOM[ivar]
        print("Relative Error MOR w.r.t. FOM %s:" % var, error_MOR_vs_FOM[ivar])

####################### Post-processing #########################
# Convergence w.r.t. the tolerance: rows = quantity, columns = variable
fig, ax = plt.subplots(3, len(problem.vars), figsize=(15, 20), squeeze=False)
for ivar, var in enumerate(problem.vars):
    ax[0,ivar].loglog(tols, err_tols[var])
    ax[0,ivar].set_title("Error %s" % var)
    ax[0,ivar].set_xlabel("Tolerance")
    ax[0,ivar].set_ylabel("Error w.r.t analytical")
    ax[2,ivar].loglog(tols, err_vs_FOM_tols[var])
    ax[2,ivar].set_title("Error w.r.t FOM %s"%var)
    ax[2,ivar].set_xlabel("Tolerance")
    ax[2,ivar].set_ylabel("Error w.r.t FOM")
for ivar, var in enumerate(MOR_instance.vars):
    ax[1,ivar].loglog(tols, n_rb_tols[var])
    ax[1,ivar].set_title("Number RB %s" % var)
    ax[1,ivar].set_xlabel("Tolerance")
    ax[1,ivar].set_ylabel("n_rb")

fig.suptitle("Convergence w.r.t tolerance", fontsize=16)
#plt.tight_layout()
#plt.show()

# Compare MOR (smallest tolerance) and FOM at the final time
fig, axs = plt.subplots(1,3, figsize=(15,4))
plot_sol(FEM2D, (qGF_MOR_recon["u"][it,:])**2 + (qGF_MOR_recon["v"][it,:])**2 , axs[0],fig, levels=21)
plot_sol(FEM2D, qGF["u"][it,:]**2 + qGF["v"][it,:]**2, axs[1], fig, levels=21)
plot_sol(FEM2D, qGF["u"][0,:]**2 + qGF["v"][0,:]**2, axs[2], fig, levels=21)
if solver.GF:
    axs[0].set_title("Global Flux: MOR")
    axs[1].set_title("Global Flux: FOM")
else:
    axs[0].set_title("Non Global Flux: MOR")
    axs[1].set_title("Non Global Flux: FOM")
axs[2].set_title("Exact Solution (initial condition)")
plt.tight_layout()
plt.show()

# Check if the reconstructed MOR solution is discretely divergence free in time
div_ROM = np.zeros(len(MOR_instance.ttGF_MOR))
for it_t in range(len(MOR_instance.ttGF_MOR)):
    q_loc = {"u": qGF_MOR_recon["u"][it_t,:],
             "v": qGF_MOR_recon["v"][it_t,:],
             "p": qGF_MOR_recon["p"][it_t,:]}
    disc_div, disc_div_norm = FEM2D.compute_GF_residual(q_loc, FEM2D.compute_sources(q_loc, problem))
    div_ROM[it_t] = disc_div_norm["p"]

plt.figure()
plt.semilogy(MOR_instance.ttGF_MOR, div_ROM)
plt.xlabel("Time")
plt.ylabel("Discrete divergence of the reconstructed MOR solution")
plt.tight_layout()
plt.savefig(os.path.join(problem.folderName,"discrete_divergence_MOR_in_time.pdf"))
plt.show()

# TODO: the discrete divergence is not fully below 10^-5?

# Start a FOM simulation from the final reconstructed MOR solution and check if it changes in time
for var in problem.vars:
    solver.ic_vect[var] = qGF_MOR_recon[var][-1,:].copy()

qGF_recon_evolved, ttGF_recon, comp_timeGF_recon, error_recon, _  = solver.solve(save_sol=False, with_error=True)
variation_from_RB_eq = dict()
for var in problem.vars:
    variation_from_RB_eq[var] = np.linalg.norm(qGF_recon_evolved[var][-1,:] - qGF_MOR_recon[var][-1,:])/ \
                                (np.linalg.norm(qGF_MOR_recon[var][-1,:]) + 1e-12)

print("Variation from RB equilibrium for u:", variation_from_RB_eq["u"])
print("Variation from RB equilibrium for v:", variation_from_RB_eq["v"])
print("Variation from RB equilibrium for p:", variation_from_RB_eq["p"])