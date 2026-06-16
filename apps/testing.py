import numpy as np
from gfsupg.solver import CartesianGeometry, FiniteElement1D
from gfsupg.solver import Scipy2DFEM, DeC, DeCSpaceTimeSUPGSolver
from gfsupg.plotting import *
from gfsupg.problem import *
import csv

import matplotlib.pyplot as plt

problem = SourceVortexTestCase()

order = 3
FEM1Dx = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
FEM1Dy = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
dec = DeC((order+1)//2,order,"gaussLobatto")

N =40

Ns = np.array([ N, N], dtype=np.int32)

geom = CartesianGeometry(problem.xL,problem.xR, Ns, problem.geometry_folder, BC=problem.BC)


FEM2D = Scipy2DFEM(geom,FEM1Dx, FEM1Dy, folder=problem.folderName)

solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec)

q_save, tt_save, comp_time, err, err_vertex  = solver.solve(with_error = True, with_error_vertex = True, GF=False)
q_save_GF, tt_save_GF, comp_time_GF, err_GF, err_vertex_GF  \
    = solver.solve(with_error = True, with_error_vertex = True, GF=True)



stop()

# GFs = [True, False]
# GF_names = ["GF", "noGF"]

# mesh_sizes_one = [20, 40, 80] #, 160, 320]

# for order in range(5,8):

#     FEM1Dx = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
#     FEM1Dy = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
#     dec = DeC((order+1)//2,order,"gaussLobatto")

#     mesh_sizes = np.array([np.int32(N//(order-1)) for N in mesh_sizes_one])

#     N_sizes = len(mesh_sizes)
#     errors     = np.zeros((len(GFs),N_sizes, problem.n_eq))
#     comp_times = np.zeros((len(GFs),N_sizes))
#     orders     = np.zeros((len(GFs),N_sizes, problem.n_eq))

#     for iN, N in enumerate(mesh_sizes):
#         print("N = %d"%N)
#         Ns = np.array([ N, N], dtype=np.int32)

#         geom = CartesianGeometry(problem.xL,problem.xR, Ns, problem.geometry_folder, BC=problem.BC)


#         FEM2D = Scipy2DFEM(geom,FEM1Dx, FEM1Dy)


#         for iGF, GF in enumerate(GFs):
#             print(GF_names[iGF])
#             if GF:
#                 method_name = "SUPG_GF"
#                 error_name = "errors_SUPG_GF"
#             else:
#                 method_name = "SUPG"
#                 error_name = "errors_SUPG"

#             solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec, GF=GF)
#             q_save, tt_save,  comp_times[iGF, iN], err, err_vertex = solver.solve(with_error = True)

#             errors[iGF, iN, :] = err
#             print("Errors ", err)


#             if iN>0:
#                 for k in range(problem.n_eq):
#                     orders[iGF, iN, k] = np.log(errors[iGF, iN,k]/errors[iGF, iN-1,k])/np.log(mesh_sizes[iN-1]/mesh_sizes[iN])

#             # Plot final simulation
#             fig = plot_all_sols(problem, FEM2D, q_save, -1, tt_save[-1])
#             fig.tight_layout()
#             fig.savefig(problem.folderName+"/simul_"+method_name+"_ord_%d_N_%04d.pdf"%(order,N))

#     for iGF, GF in enumerate(GFs):
#         print(GF_names[iGF])
#         if GF:
#             error_name = "errors_SUPG_GF"
#         else:
#             error_name = "errors_SUPG"

#         with open (problem.folderName+"/"+error_name+"_ord%d.csv"%order,"w") as file:
#             writer = csv.writer(file)

#             header = ["N", "comp time", "err u", "err v", "err p", "ord u", "ord v", "ord p"]
#             writer.writerow(header)

#             for iN, N in enumerate(mesh_sizes):
#                 line = [N, comp_times[iGF, iN], errors[iGF, iN,0], errors[iGF, iN,1], errors[iGF, iN,2],  orders[iGF, iN,0], orders[iGF, iN,1], orders[iGF, iN,2] ]
#                 writer.writerow(line)

#     plt.figure()
#     for iGF, GF in enumerate(GFs):
#         if GF:
#             plt.loglog(mesh_sizes, errors[iGF,:,:], linewidth=3, label = [var + " GF"*GF for var in problem.vars])
#         else:
#             plt.loglog(mesh_sizes, errors[iGF,:,:], "--", label = problem.vars)

#     plt.loglog(mesh_sizes, 1./mesh_sizes**order * errors[iGF, -1,0]*mesh_sizes[-1]**order , ":", label="order %d"%order)
#     plt.legend()
#     plt.ylabel("Error")
#     plt.xlabel("Elements in one direction")
#     plt.tight_layout()
#     plt.savefig(problem.folderName+"/"+error_name+"_ord%d.pdf"%order)
#     plt.show(block=False)
#     plt.close('all')
