import numpy as np
from gfsupg.solver import CartesianGeometry, FiniteElement1D, Scipy2DFEM
from gfsupg.solver import DeC, DeCSpaceTimeSUPGSolver
from gfsupg.problem import *
from gfsupg.plotting import *
from gfsupg .MOR import *

import matplotlib.pyplot as plt

order = 2

FEM1Dx = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
FEM1Dy = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
dec = DeC((order+1)//2,order,"gaussLobatto")

Ns = np.array([10,10], dtype=np.int32)
problem = SmoothVortexTestCaseParam(is_long=True)
geom = CartesianGeometry(problem.xL,problem.xR, Ns, problem.geometry_folder, BC=problem.BC)
FEM2D = Scipy2DFEM(geom, FEM1Dx, FEM1Dy, folder=problem.folderName)

MOR_instance = MOR(problem, FEM2D, dec, tol=1e-5, GF=True, stab="SUPG")
load_sol = True

# Define the parameters to compute our snapshots
coeff_exp = np.arange(1,11,1)
mu_offline = [[9.81, 0.45, coeff_exp_] for coeff_exp_ in coeff_exp]

# Perform the offline phase  
MOR_instance.run_offline(mu_offline, load_sol=load_sol)

# Perform the online phase
compute_residuals = True
online_params = [9.81, 0.45, 2.5]
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

print("Relative Error MOR u:", MOR_instance.error_MOR[0])
print("Relative Error MOR v:", MOR_instance.error_MOR[1])
print("Relative Error MOR p:", MOR_instance.error_MOR[2])
print("")

# Compute the FOM solution for the current 'online parameters'
problem.set_parameters(online_params)
solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec, GF = True, stab = "SUPG", trick_second_der = False)
qGF, ttGF, comp_timeGF, error, _  = solver.solve(save_sol = True, with_error = True)
print("Relative Error FOM u:", error[0])
print("Relative Error FOM v:", error[1])
print("Relative Error FOM p:", error[2])
print("")

# Post-processing analysis
it = -1
fig, axs = plt.subplots(1,3, figsize=(15,4))
plot_sol(FEM2D, (MOR_instance.basis["u"] @ MOR_instance.qGF_MOR["u"][it,:])**2 + (MOR_instance.basis["v"] @ MOR_instance.qGF_MOR["v"][it,:])**2 , axs[0],fig, levels=21)
plot_sol(FEM2D, qGF["u"][it,:]**2 + qGF["v"][it,:]**2 , axs[1],fig, levels=21)
plot_sol(FEM2D, qGF["u"][0,:]**2 + qGF["v"][0,:]**2 , axs[2],fig, levels=21)
axs[0].set_title("Global Flux: MOR") 
axs[1].set_title("Global Flux: FOM")
axs[2].set_title("Exact Solution")

qGF_MOR_recon = MOR_instance.qGF_MOR
for var in tuple(MOR_instance.basis.keys()):
    qGF_MOR_recon[var] = (MOR_instance.basis[var] @ MOR_instance.qGF_MOR[var].T).T
plot_all_sols(problem, FEM2D, qGF_MOR_recon, it, MOR_instance.ttGF_MOR[-1], levels=21)

error_MOR_vs_FOM = np.zeros(len(problem.vars))
for ivar, var in enumerate(problem.vars):
    error_MOR_vs_FOM[ivar] = np.linalg.norm(qGF_MOR_recon[var][it,:] - qGF[var][it,:])/np.linalg.norm(qGF[var][it,:])
print("Relative Error MOR w.r.t. FOM u:", error_MOR_vs_FOM[0])
print("Relative Error MOR w.r.t. FOM v:", error_MOR_vs_FOM[1])
print("Relative Error MOR w.r.t. FOM p:", error_MOR_vs_FOM[2])
plt.show()