import numpy as np
from gfsupg.solver import CartesianGeometry, FiniteElement1D, Scipy2DFEM
from gfsupg.solver import DeC, DeCSpaceTimeSUPGSolver, ImplicitDec, ImplicitEuler
from gfsupg.problem import *
from gfsupg.plotting import *
from gfsupg .MOR import *

import matplotlib.pyplot as plt

# Set the problem data
order = 2

FEM1Dx = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
FEM1Dy = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
dec = DeC((order+1)//2,order,"gaussLobatto")

Nx = 20
Ny = 20
Ns = np.array([Nx,Ny], dtype=np.int32)
problem = SmoothVortexTestCaseParam(is_long=True)
problem.T_fin = 100.
#problem = ObliqueTestCase()
# problem = ShuVortexTestCaseParam()
geom = CartesianGeometry(problem.xL,problem.xR, Ns, problem.geometry_folder, BC=problem.BC)
FEM2D = Scipy2DFEM(geom, FEM1Dx, FEM1Dy, folder=problem.folderName)

# Compute the FOM solution for the current 'online parameters'
online_params = [9.81, 0.45, 2.5]
#online_params = []
# problem.set_final_time(1.0)
# online_params = [0.47, 0.48, 0.21]
problem.set_parameters(online_params)
solver = ImplicitEuler(problem, FEM2D, dec, GF=True, stab="SUPG", trick_second_der=False)
solver.set_CFL(100.0)
qGF, ttGF, comp_timeGF, error, _  = solver.solve(save_sol=True, with_error=True)
print("Relative Error FOM u:", error[0])
print("Relative Error FOM v:", error[1])
print("Relative Error FOM p:", error[2])
print("")

# Plot one solution

plot_all_sols(problem, FEM2D, qGF, -1, ttGF[-1], levels=21)
plt.show()

# Perform a convergence of the MOR solver (i.e. test with different tolerance values)
load_sol = True
compute_residuals = False
tols = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
err_tols = dict()
n_rb_tols = dict()
err_vs_FOM_tols = dict()
for var in problem.vars:
    err_tols[var] = np.zeros_like(tols)
    n_rb_tols[var] = np.zeros_like(tols)
    err_vs_FOM_tols[var] = np.zeros_like(tols)


# Define the parameters to compute our snapshots
coeff_exp = np.arange(1,11,1)
mu_offline = [[9.81, 0.45, coeff_exp_] for coeff_exp_ in coeff_exp]
#mu_offline = [[]]
# mu_offline = [[x0, y0, r0] for x0 in np.linspace(0.35,0.65,6) for y0 in np.linspace(0.35,0.65,6) for r0 in np.linspace(0.15,0.25,6)]
#x0_offline = np.array([0.3,0.4,0.5,0.6,0.7])
#y0_offline = np.array([0.3,0.4,0.5,0.6,0.7])
#r_offline = np.array([0.05,0.1,0.2,0.25])
#mu_offline = [[x0, y0, r0] for x0 in x0_offline for y0 in y0_offline for r0 in r_offline]

MOR_instance = MOR(problem, FEM2D, dec, tol=tols[0], GF=True, stab="SUPG")
# Perform the offline phase 
MOR_instance.run_offline(mu_offline, load_sol=load_sol)

for i, tol in enumerate(tols):
    MOR_instance.truncate_basis(tol=tol)

    n_rb_tols["u"][i] = MOR_instance.n_rb["u"]
    n_rb_tols["v"][i] = MOR_instance.n_rb["v"]
    n_rb_tols["p"][i] = MOR_instance.n_rb["p"]

    # Perform the online phase
    MOR_instance.run_online(online_params, compute_residuals=compute_residuals)

    if compute_residuals:
        plt.subplot(3, 1, 1)
        plt.plot(MOR_instance.ttGF_MOR, MOR_instance.res["u"])
        plt.plot(MOR_instance.ttGF_MOR, MOR_instance.res_rb["u"])
        plt.subplot(3, 1, 2)
        plt.plot(MOR_instance.ttGF_MOR, MOR_instance.res["v"])
        plt.plot(MOR_instance.ttGF_MOR, MOR_instance.res_rb["v"])
        plt.subplot(3, 1, 3)
        plt.plot(MOR_instance.ttGF_MOR, MOR_instance.res["p"])
        plt.plot(MOR_instance.ttGF_MOR, MOR_instance.res_rb["p"])

    err_tols["u"][i] = MOR_instance.error_MOR[0]
    print("Relative Error MOR u:", MOR_instance.error_MOR[0])
    err_tols["v"][i] = MOR_instance.error_MOR[1]
    print("Relative Error MOR v:", MOR_instance.error_MOR[1])
    err_tols["p"][i] = MOR_instance.error_MOR[2]
    print("Relative Error MOR p:", MOR_instance.error_MOR[2])
    print("")

    qGF_MOR_recon = MOR_instance.qGF_MOR.copy()
    for var in tuple(MOR_instance.basis.keys()):
        qGF_MOR_recon[var] = (MOR_instance.basis[var] @ MOR_instance.qGF_MOR[var].T).T
    it = -1
    #plot_all_sols(problem, FEM2D, qGF_MOR_recon, it, MOR_instance.ttGF_MOR[it], levels=21)
    error_MOR_vs_FOM = np.zeros(len(problem.vars))
    for ivar, var in enumerate(problem.vars):
        error_MOR_vs_FOM[ivar] = np.linalg.norm(qGF_MOR_recon[var][it,:] - qGF[var][it,:])/np.linalg.norm(qGF[var][it,:])
    err_vs_FOM_tols["u"][i] = error_MOR_vs_FOM[0]
    print("Relative Error MOR w.r.t. FOM u:", error_MOR_vs_FOM[0])
    err_vs_FOM_tols["v"][i] = error_MOR_vs_FOM[1]
    print("Relative Error MOR w.r.t. FOM v:", error_MOR_vs_FOM[1])
    err_vs_FOM_tols["p"][i] = error_MOR_vs_FOM[2]
    print("Relative Error MOR w.r.t. FOM p:", error_MOR_vs_FOM[2])

fig, ax = plt.subplots(3, 3, figsize=(15, 4))
ax[0,0].semilogx(tols, err_tols["u"])
ax[0,0].set_title("Error u")
ax[0,0].set_xlabel("Tolerance")
ax[0,0].set_ylabel("Error w.r.t analytical")
ax[1,0].semilogx(tols, n_rb_tols["u"])
ax[1,0].set_title("Number RB u")
ax[1,0].set_xlabel("Tolerance")
ax[1,0].set_ylabel("n_rb")
ax[2,0].semilogx(tols, err_vs_FOM_tols["u"])
ax[2,0].set_title("Error w.r.t FOM u")
ax[2,0].set_xlabel("Tolerance")
ax[2,0].set_ylabel("Error w.r.t FOM")
ax[0,1].semilogx(tols, err_tols["v"])
ax[0,1].set_title("Error v")
ax[0,1].set_xlabel("Tolerance")
ax[0,1].set_ylabel("Error w.r.t analytical")
ax[1,1].semilogx(tols, n_rb_tols["v"])
ax[1,1].set_title("Number RB v")
ax[1,1].set_xlabel("Tolerance")
ax[1,1].set_ylabel("n_rb")
ax[2,1].semilogx(tols, err_vs_FOM_tols["v"])
ax[2,1].set_title("Error w.r.t FOM v")
ax[2,1].set_xlabel("Tolerance")
ax[2,1].set_ylabel("Error w.r.t FOM")
ax[0,2].semilogx(tols, err_tols["p"])
ax[0,2].set_title("Error p")
ax[0,2].set_xlabel("Tolerance")
ax[0,2].set_ylabel("Error w.r.t analytical")
ax[1,2].semilogx(tols, n_rb_tols["p"])
ax[1,2].set_title("Number RB p")
ax[1,2].set_xlabel("Tolerance")
ax[1,2].set_ylabel("n_rb")
ax[2,2].semilogx(tols, err_vs_FOM_tols["p"])
ax[2,2].set_title("Error w.r.t FOM p")
ax[2,2].set_xlabel("Tolerance")
ax[2,2].set_ylabel("Error w.r.t FOM")
fig.suptitle("Convergence w.r.t tolerance", fontsize=16)
#plt.tight_layout()
#plt.show()

# Post-processing analysis
fig, axs = plt.subplots(1,3, figsize=(15,4))
plot_sol(FEM2D, (MOR_instance.basis["u"] @ MOR_instance.qGF_MOR["u"][it,:])**2 + (MOR_instance.basis["v"] @ MOR_instance.qGF_MOR["v"][it,:])**2 , axs[0],fig, levels=21)
plot_sol(FEM2D, qGF["u"][it,:]**2 + qGF["v"][it,:]**2, axs[1], fig, levels=21)
plot_sol(FEM2D, qGF["u"][0,:]**2 + qGF["v"][0,:]**2, axs[2], fig, levels=21)
if solver.GF:
    axs[0].set_title("Global Flux: MOR") 
    axs[1].set_title("Global Flux: FOM")
else:
    axs[0].set_title("Non Global Flux: MOR") 
    axs[1].set_title("Non Global Flux: FOM")
axs[2].set_title("Exact Solution")
plt.tight_layout()
plt.show()