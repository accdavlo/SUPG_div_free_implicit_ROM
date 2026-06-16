import numpy as np
from gfsupg.solver import CartesianGeometry, FiniteElement1D
from gfsupg.solver import Scipy2DFEM, DeC, DeCSpaceTimeSUPGSolver
from gfsupg.problem_old import LinearAcoustic2D
from gfsupg.plotting import *
import pickle
import csv

import matplotlib.pyplot as plt

problem = LinearAcoustic2D("oblique")

GF = False
stab = "SUPG" # "OSS" #"SUPG" #


mesh_sizes_one = [20, 30, 40, 80]#, 160, 320]

for order in range(2,8):

    print("---------------")
    print("--- Order %d ---"%order)
    print("---------------")

    FEM1Dx = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
    FEM1Dy = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
    dec = DeC((order+1)//2,order,"gaussLobatto")

    mesh_sizes = np.array([np.int32(N//(order-1)) for N in mesh_sizes_one])

    N_sizes = len(mesh_sizes)

    for iN, N in enumerate(mesh_sizes):
        print("N = %d"%N)
        Ns = np.array([ N, N], dtype=np.int32)

        geom = CartesianGeometry(problem.xL,problem.xR, Ns, problem.geometry_folder, BC=problem.BC)



        FEM2D = Scipy2DFEM(geom,FEM1Dx, FEM1Dy, folder=problem.folderName)

        solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec, GF=GF, stab=stab)

        fig = plot_one_sol(problem, FEM2D, solver.ic_vect)
        fig.tight_layout()
        fig.savefig(problem.folderName+"/IC_ord_%d_N_%04d.pdf"%(order,N))
        plt.close(fig)
