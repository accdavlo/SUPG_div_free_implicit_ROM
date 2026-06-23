import numpy as np
from gfsupg.solver import CartesianGeometry, FiniteElement1D, Scipy2DFEM
from gfsupg.solver import DeC, DeCSpaceTimeSUPGSolver
from gfsupg.problem import *
from gfsupg.plotting import *

import matplotlib.pyplot as plt

mode = "online"

if mode == "offline":
    order = 2

    FEM1Dx = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
    FEM1Dy = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
    dec = DeC((order+1)//2,order,"gaussLobatto")

    Ns = np.array([10,10], dtype=np.int32)

    coeff_exp = np.arange(1,11,1)
    us = np.zeros([(Ns[0] + 1)**2 * (Ns[1] + 1)**2,len(coeff_exp)], dtype=np.float64)
    vs = np.zeros([(Ns[0] + 1)**2 * (Ns[1] + 1)**2,len(coeff_exp)], dtype=np.float64)
    ps = np.zeros([(Ns[0] + 1)**2 * (Ns[1] + 1)**2,len(coeff_exp)], dtype=np.float64)
    for i, coeff_exp_ in enumerate(coeff_exp):
        problem = SmoothVortexTestCaseParam(is_long=True, coeff_exp=coeff_exp_)

        geom = CartesianGeometry(problem.xL,problem.xR, Ns, problem.geometry_folder, BC=problem.BC)

        FEM2D = Scipy2DFEM(geom, FEM1Dx, FEM1Dy, folder=problem.folderName)

        print("Computing the GF-SUPG")
        solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec, GF = True, stab = "SUPG", trick_second_der=False)
        qGF, ttGF, comp_timeGF, _ , _  = solver.solve(save_sol = True, with_error = True)

        us[:,i] = qGF["u"][-1,:]
        vs[:,i] = qGF["v"][-1,:]
        ps[:,i] = qGF["p"][-1,:]

        np.savez("snapshots_coeff_exp.npz", us=us, vs=vs, ps=ps)
elif mode == "online":  
    inputfile = np.load("snapshots_coeff_exp.npz")
    us = inputfile['us']
    vs = inputfile['vs']
    ps = inputfile['ps']

    basis = {}
    basis["u"],Sigma_svd_u,V_svd_u = np.linalg.svd(us, full_matrices=False)
    basis["v"],Sigma_svd_v,V_svd_v = np.linalg.svd(vs, full_matrices=False)
    basis["p"],Sigma_svd_p,V_svd_p = np.linalg.svd(ps, full_matrices=False)
    #plt.semilogy(Sigma_svd_u,'o')
    #plt.show()

    problem = SmoothVortexTestCaseParam(is_long=True)
    order = 2
    FEM1Dx = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
    FEM1Dy = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
    dec = DeC((order+1)//2,order,"gaussLobatto")
    Ns = np.array([10,10], dtype=np.int32)
    geom = CartesianGeometry(problem.xL,problem.xR, Ns, problem.geometry_folder, BC=problem.BC)
    FEM2D = Scipy2DFEM(geom, FEM1Dx, FEM1Dy, folder=problem.folderName)
    FEM2D.build_matrices_MOR(basis)

    print("Computing the GF-SUPG")
    solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec, GF = True, stab = "SUPG", trick_second_der=False)
    qGF_MOR, ttGF_MOR, comp_timeGF_MOR, _ , _  = solver.solve_MOR(basis=basis,save_sol = True, with_error = True)

#it = -1
#fig, axs = plt.subplots(1,3, figsize=(15,4))
#plot_sol(FEM2D, q["u"][it,:]**2+q["v"][it,:]**2 , axs[0],fig, levels=21)
#plot_sol(FEM2D, qGF["u"][it,:]**2+qGF["v"][it,:]**2 , axs[1],fig, levels=21)
#plot_sol(FEM2D, qGF["u"][0,:]**2+qGF["v"][0,:]**2 , axs[2],fig, levels=21)
#axs[0].set_title("Non Global Flux")
#axs[1].set_title("Global Flux")
#axs[2].set_title("Exact Solution")
#plt.show()