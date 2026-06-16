import numpy as np
from gfsupg.solver import CartesianGeometry, FiniteElement1D, Scipy2DFEM
from gfsupg.solver import DeC, DeCSpaceTimeSUPGSolver
from gfsupg.problem import *
from gfsupg.plotting import *

import matplotlib.pyplot as plt


order=2

FEM1Dx = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
FEM1Dy = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
dec = DeC((order+1)//2,order,"gaussLobatto")


# problem = SmoothVortexTestCase(is_long=True)#, pert_coeff=1e-3, pert_type="an") #LinearAdvection("smooth_vortex_long")
problem = CoriolisVortexTestCase(is_long=True)#, pert_coeff=1e-3, pert_type="an") #LinearAdvection("smooth_vortex_long")


Ns = np.array([10,10], dtype=np.int32)

geom = CartesianGeometry(problem.xL,problem.xR, Ns, problem.geometry_folder, BC=problem.BC)

FEM2D = Scipy2DFEM(geom,FEM1Dx, FEM1Dy, folder=problem.folderName)


print("Computing the classical SUPG solution")
solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec, GF = False, stab = "SUPG", trick_second_der=False)

q, tt, comp_time, _ , _  = solver.solve( save_sol = True)
for it in [0,10,20,29, len(tt)-1]:
    t = tt[it]
    plot_all_sols(problem, FEM2D,q,it, t , 200)
    



print("Computing the GF-SUPG")

solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec, GF = True, stab = "SUPG", trick_second_der=False)
qGF, ttGF, comp_timeGF, _ , _  = solver.solve(save_sol = True, with_error = True)
for it in [0,10,20,29, len(tt)-1]:
    t = ttGF[it]
    plot_all_sols(problem, FEM2D,qGF,it, t , 200)
    
# plt.close("all")


it = -1
fig, axs = plt.subplots(1,3, figsize=(15,4))
plot_sol(FEM2D, q["u"][it,:]**2+q["v"][it,:]**2 , axs[0],fig, levels=21)
plot_sol(FEM2D, qGF["u"][it,:]**2+qGF["v"][it,:]**2 , axs[1],fig, levels=21)
plot_sol(FEM2D, qGF["u"][0,:]**2+qGF["v"][0,:]**2 , axs[2],fig, levels=21)
axs[0].set_title("Non Global Flux")
axs[1].set_title("Global Flux")
axs[2].set_title("Exact Solution")
plt.show()