import numpy as np
from gfsupg.solver import CartesianGeometry, FiniteElement1D, Scipy2DFEM
from gfsupg.solver import DeC, DeCSpaceTimeSUPGSolver
# from gfsupg.problem import LinearAcoustic2D
from gfsupg.problem2 import *
from gfsupg.plotting import *

import matplotlib.pyplot as plt


#problem.T_fin = 1.
order=3

FEM1Dx = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
FEM1Dy = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
dec = DeC((order+1)//2,order,"gaussLobatto")
# dec = DeC(4,5,"gaussLobatto")


problem = SmoothVortexTestCase(is_long=False, pert_coeff=1e-3, pert_type="an") #LinearAdvection("smooth_vortex_long")


Ns = np.array([20,20], dtype=np.int32)

geom = CartesianGeometry(problem.xL,problem.xR, Ns, problem.geometry_folder, BC=problem.BC)

FEM2D = Scipy2DFEM(geom,FEM1Dx, FEM1Dy, folder=problem.folderName)


# print("Computing the classical SUPG solution")
# solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec, GF = False, stab = "SUPG", trick_second_der=False)

# q, tt, comp_time, _ , _  = solver.solve(CFL=0.5, save_sol = True)
# for it in [0,10,20,29, len(tt)-1]:
#     t = tt[it]
#     plot_all_sols(problem, FEM2D,q,it, t , 200)


# print("Computing the GF-SUPG")

solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec, GF = True, stab = "SUPG", trick_second_der=False)
q, tt, comp_time, _ , _  = solver.solve(CFL=0.5, save_sol = True, with_error = True)
for it in [0,10,20,29, len(tt)-1]:
    t = tt[it]
    plot_all_sols(problem, FEM2D,q,it, t , 200)