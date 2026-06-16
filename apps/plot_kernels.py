import numpy as np
from gfsupg.solver import CartesianGeometry, FiniteElement1D, Scipy2DFEM
from gfsupg.solver import DeC, DeCSpaceTimeSUPGSolver
from gfsupg.plotting import *
from gfsupg.problem import *
import sympy as sym
import os

import matplotlib.pyplot as plt

def two_cells_matrix(ZZ):
    AA = np.zeros((2*(order-1)-1,2*(order-1)+1))
    AA[:order-1,:order] = ZZ[1:,:]
    AA[order-2:,order-1:] = AA[order-2:,order-1:] + ZZ[:-1,:]
    AA[abs(AA)<1e-14]=0.
    return AA

def more_cells_matrix(ZZ,N):
    AA = np.zeros((N*(order-1)-1,N*(order-1)+1))
    #i=0
    AA[:order-1,:order] = ZZ[1:,:]
    for i in range(1,N-1):
        AA[(order-1)*i-1:(order-1)*(i+1),(order-1)*i:(order-1)*(i+1)+1] +=  ZZ
    
    #i=N-1
    AA[(order-1)*(N-1)-1:(order-1)*(N)-1,(order-1)*(N-1):(order-1)*(N)+1]+=ZZ[:-1,:]

    AA[abs(AA)<1e-14]=0.
    return AA


def more_cells_matrix_full(ZZ,N):
    AA = np.zeros((N*(order-1)+1,N*(order-1)+1))
    #i=0
    AA[:order,:order] = ZZ[:,:]
    for i in range(1,N-1):
        AA[(order-1)*i:(order-1)*(i+1)+1,(order-1)*i:(order-1)*(i+1)+1] +=  ZZ
    
    #i=N-1
    AA[(order-1)*(N-1):(order-1)*(N)+1,(order-1)*(N-1):(order-1)*(N)+1]+=ZZ[:,:]

    AA[abs(AA)<1e-14]=0.
    return AA
def get_kernel_vectors(null_space):
    a = np.array(null_space[0]+null_space[1]).flatten()
    b = np.array(null_space[0]-null_space[1]).flatten()
    return a,b

def plot_vector(u, N_cell, FEM1D, null_space_name):
    uu = np.zeros((FEM1D.degree+1,N_cell))
    xx = np.zeros((FEM1D.degree+1,N_cell))
    x_edge = np.linspace(0,1,N_cell+1)
    dx = x_edge[1]-x_edge[0]
    for i in range(N_cell):
        uu[:,i] = u[i*FEM1D.degree:(i+1)*FEM1D.degree+1]
        xx[:,i] = x_edge[i]+dx*FEM1D.nodes
        poly=np.polyfit(xx[:,i],uu[:,i],FEM1D.degree)
        xx_local = np.linspace(x_edge[i],x_edge[i+1],100)
        plt.plot(xx_local,np.polyval(poly,xx_local))
    plt.plot(xx,uu,'o')
    plt.tight_layout()
    plt.savefig(os.path.join(kernel_folder,f"Kernel_{null_space_name}_deg_{FEM1D.degree}_N_{N_cell}.pdf"))

def plot_kernel(null_space, null_space_name):
    plt.figure()
    a,b = get_kernel_vectors(null_space)
    plot_vector(a,N_cell,FEM1Dx,null_space_name)
    plot_vector(b,N_cell,FEM1Dx,null_space_name)

kernel_folder = "kernels"
os.system(f"mkdir {kernel_folder}")


#problem.T_fin = 1.
for order in range(2,8):
    for N_cell in [5,6]:

        FEM1Dx = FiniteElement1D(order-1,"gaussLobatto","gaussLegendre")
        FEM1Dy = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
        dec = DeC((order+1)//2,order,"gaussLobatto")


        D_mult_j = more_cells_matrix(FEM1Dx.matrix["deriv_j"],N_cell)
        D_mult_ij = more_cells_matrix(FEM1Dx.matrix["deriv_ij"],N_cell)
        D_mult_i  = more_cells_matrix(FEM1Dx.matrix["deriv_i"],N_cell)
        Mass_mult_full = more_cells_matrix_full(FEM1Dx.matrix["lump_mass"],N_cell)
        D_mult_full_j = more_cells_matrix_full(FEM1Dx.matrix["deriv_j"],N_cell)
        Z= D_mult_ij - D_mult_j@np.linalg.inv(Mass_mult_full)@D_mult_full_j


        Dij_sym = sym.Matrix(D_mult_ij)
        VV_Dij = Dij_sym.nullspace()

        Dj_sym = sym.Matrix(D_mult_j)
        VV_Dj = Dj_sym.nullspace()

        Z_sym = sym.Matrix(Z)
        VV_Z = Z_sym.nullspace()



        print("Kernel Dj")
        plot_kernel(VV_Dj,"Dj")
        plot_kernel(VV_Dij,"Dij")
        plot_kernel(VV_Z,"Z")

