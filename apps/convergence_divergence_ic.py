import numpy as np
from gfsupg.solver import CartesianGeometry, FiniteElement1D
from gfsupg.solver import Scipy2DFEM, DeC, DeCSpaceTimeSUPGSolver
from gfsupg.problem import LinearAcoustic2D
from gfsupg.plotting import *
import csv

import matplotlib.pyplot as plt

problem = LinearAcoustic2D("SG")
GFs = [True]#, False]
GF_names = ["GF"]#, "noGF"]

stabilizations = ["SUPG"]#, "OSS"]

for GF_divergence in [False,True]:

    if GF_divergence:
        div_name = "divergenceGF"
    else:
        div_name = "divergenceSimple"

    mesh_sizes_one = [20, 40, 80, 160, 320]

    for order in range(2,8):

        print("---------------")
        print("--- Order %d ---"%order)
        print("---------------")

        FEM1Dx = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
        FEM1Dy = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
        dec = DeC((order+1)//2,order,"gaussLobatto")

        mesh_sizes = np.array([np.int32(N//(order-1)) for N in mesh_sizes_one])

        N_sizes = len(mesh_sizes)
        errors            = np.zeros((len(stabilizations),len(GFs),N_sizes, problem.n_eq))
        errors_vertex     = np.zeros((len(stabilizations),len(GFs),N_sizes, problem.n_eq))
        errors_div_ic     = np.zeros((len(stabilizations),len(GFs),N_sizes))
        errors_div_ic_nbr = np.zeros((len(stabilizations),len(GFs),N_sizes))
        comp_times        = np.zeros((len(stabilizations),len(GFs),N_sizes))
        orders            = np.zeros((len(stabilizations),len(GFs),N_sizes, problem.n_eq))
        orders_vertex     = np.zeros((len(stabilizations),len(GFs),N_sizes, problem.n_eq))
        orders_div_ic     = np.zeros((len(stabilizations),len(GFs),N_sizes))
        orders_div_ic_nbr = np.zeros((len(stabilizations),len(GFs),N_sizes))

        for iN, N in enumerate(mesh_sizes):
            print("N = %d"%N)
            Ns = np.array([ N, N], dtype=np.int32)

            geom = CartesianGeometry(problem.xL,problem.xR, Ns,problem.geometry_folder, BC=problem.BC)


            FEM2D = Scipy2DFEM(geom,FEM1Dx, FEM1Dy, problem.folderName)

            for istab, stab in enumerate(stabilizations):
                for iGF, GF in enumerate(GFs):
                    print(GF_names[iGF])
                    if GF:
                        method_name = stab+"_GF"
                        error_name = "errors_"+stab+"_GF"
                    else:
                        method_name = stab
                        error_name = "errors_"+stab

                    print(method_name)

                    solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec, GF=GF, stab=stab)

                    sources = FEM2D.compute_sources(solver.ic_vect, problem)

                    if GF_divergence:
                        # disc_div = FEM2D.compute_discrete_divergence(solver.ic_vect)
                        disc_res = FEM2D.compute_GF_residual(solver.ic_vect,sources)
                    else:
                        # disc_div = FEM2D.compute_wrong_discrete_divergence(solver.ic_vect)
                        disc_res = FEM2D.compute_noGF_residual(solver.ic_vect,sources)
                    disc_div = disc_res["p"]
                    errors_div_ic[istab, iGF, iN] = np.linalg.norm(disc_div)/np.sqrt(FEM2D.n_dof_tot)
                    disc_div[FEM2D.dirichlet_indexes["all"]]=0.
                    errors_div_ic_nbr[istab, iGF, iN] = np.linalg.norm(disc_div)/np.sqrt(FEM2D.n_dof_tot)

                    print(errors_div_ic[istab, iGF, iN])
                    print(errors_div_ic_nbr[istab, iGF, iN])

                    if iN>0:
                        orders_div_ic[istab, iGF, iN] = np.log(errors_div_ic[istab, iGF, iN]\
                                                /errors_div_ic[istab, iGF, iN-1])/np.log(mesh_sizes[iN-1]/mesh_sizes[iN])
                        orders_div_ic_nbr[istab, iGF, iN] = np.log(errors_div_ic_nbr[istab, iGF, iN]\
                                                /errors_div_ic_nbr[istab, iGF, iN-1])/np.log(mesh_sizes[iN-1]/mesh_sizes[iN])
                        

        for istab, stab in enumerate(stabilizations):

            with open (problem.folderName+"/%s_discretization_%s_error_ord%d.csv"%(div_name,stab,order),"w") as file:
                writer = csv.writer(file)

                header = ["N", "err", "ord","err_nbr","ord_nbr"]
                writer.writerow(header)

                for iN, N in enumerate(mesh_sizes):
                    line = [N, errors_div_ic[istab,iGF, iN],orders_div_ic[istab,iGF, iN],\
                            errors_div_ic_nbr[istab,iGF, iN],orders_div_ic_nbr[istab,iGF, iN]]
                    writer.writerow(line)
                

            experimental_order = np.mean(orders_div_ic[istab,0,1:])
            guess_order = np.ceil(np.mean(orders_div_ic[istab,0,1:]))

            plt.figure()
            iGF = 0
            GF = True
            plt.loglog(mesh_sizes, errors_div_ic[istab,iGF,:], linewidth=4, label ="Discrete Divergence p="+str(order-1))

            for (shift, linestyle) in zip(range(-1,2),["--","-.",":"]):
                plt.loglog(mesh_sizes, 1./np.float64(mesh_sizes)**(guess_order+shift) * errors_div_ic[istab,iGF, 1]*mesh_sizes[1]**(guess_order+shift) , linestyle=linestyle, label="order %d"%(guess_order+shift))
            plt.legend()
            plt.ylabel("Discrete %s of analytical IC"%div_name)
            plt.xlabel("Elements in one direction")
            plt.title( "Convergence discrete %s (experimental order %1.2f)"%(div_name,experimental_order)) 
            plt.tight_layout()
            plt.savefig(problem.folderName+"/discrete_%s_%s_IC_ord%d.pdf"%(stab,div_name,order))
        #    plt.show(block=False)
            plt.close('all')


            experimental_order = np.mean(orders_div_ic_nbr[istab,0,1:])
            guess_order = np.ceil(np.mean(orders_div_ic_nbr[istab,0,1:]))

            plt.figure()
            iGF = 0
            GF = True
            plt.loglog(mesh_sizes, errors_div_ic_nbr[istab,iGF,:], linewidth=4, label ="Discrete Divergence p="+str(order-1))

            for (shift, linestyle) in zip(range(-1,2),["--","-.",":"]):
                plt.loglog(mesh_sizes, 1./np.float64(mesh_sizes)**(guess_order+shift) * errors_div_ic_nbr[istab,iGF, 1]*mesh_sizes[1]**(guess_order+shift) , linestyle=linestyle, label="order %d"%(guess_order+shift))
            plt.legend()
            plt.ylabel("Discrete %s of analytical IC"%div_name)
            plt.xlabel("Elements in one direction")
            plt.title( "Convergence discrete %s (experimental order %1.2f)"%(div_name,experimental_order)) 
            plt.tight_layout()
            plt.savefig(problem.folderName+"/discrete_%s_nbr_%s_IC_ord%d.pdf"%(div_name,stab,order))
        #    plt.show(block=False)
            plt.close('all')
