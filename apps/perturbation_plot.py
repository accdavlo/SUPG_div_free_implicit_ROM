import numpy as np
from gfsupg.solver import CartesianGeometry, FiniteElement1D, Scipy2DFEM
from gfsupg.solver import DeC, DeCSpaceTimeSUPGSolver
from gfsupg.plotting import *
from gfsupg.problem import *

import matplotlib.pyplot as plt

stab = "SUPG"

for pert_val in [1e-3, 1e-2]: #0. #1e-6, 1e-1, 
    problem = StommelGyreTestCase(pert_coeff = pert_val, pert_type="opt")


    mesh_sizes_one = [20, 30, 40, 80]

    for order in range(2,6):

        print("=============== ORDER %d ================="%order)

        FEM1Dx = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
        FEM1Dy = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
        dec = DeC((order+1)//2,order,"gaussLobatto")

        mesh_sizes = np.array([np.int32(N//(order-1)) for N in mesh_sizes_one])



        FEM1Dx = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
        FEM1Dy = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
        dec = DeC((order+1)//2,order,"gaussLobatto")


        #%%  


        for Nx in  mesh_sizes:

            Ns = np.array([Nx,Nx], dtype=np.int32)

            geom = CartesianGeometry(problem.xL,problem.xR, Ns, problem.geometry_folder, BC=problem.BC)

            FEM2D = Scipy2DFEM(geom,FEM1Dx, FEM1Dy, folder=problem.folderName)

            solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec)
            #solver.set_save_slabs(200)
            solver.set_Nt_max(1e6)
            q_save, tt_save, comp_time, err, err_vertex  = solver.solve(with_error = True, with_error_vertex = True, GF=False, stab = stab)

            div_errors = FEM2D.compute_discrete_divergence_norms(q_save)



            solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec)
            q_save_GF, tt_save_GF, comp_time_GF, err_GF, err_GF_vertex = solver.solve(with_error = True, with_error_vertex = True, GF=True, stab=stab)
            print(err_GF)
            print(err_GF_vertex)

            div_errors_GF = FEM2D.compute_discrete_divergence_norms(q_save_GF)


            fig,axs = plt.subplots(1,3,figsize=(15,4))
            plot_sol(FEM2D, solver.ic_vect["u"], axs[0], fig)
            plot_sol(FEM2D, solver.ic_vect["v"], axs[1], fig)
            plot_sol(FEM2D, solver.ic_vect["p"], axs[2], fig, levels=np.linspace(np.min(solver.ic_vect["p"]),np.max(solver.ic_vect["p"]),13))

            for iv, var in enumerate(problem.vars):
                axs[iv].set_title(var)

            fig.tight_layout()
            fig.savefig(problem.folderName+"/IC_pert_%s_ord%d_N%04d_pert_%1.3e.pdf"%(stab,order,Ns[0],pert_val))
            plt.close()


            fig = plot_all_sols(problem, FEM2D, q_save, len(q_save)-1, tt_save[len(q_save)-1], levels=None)
            fig.tight_layout()
            fig.savefig(problem.folderName+"/final_sol_%s_NOGF_ord%d_N%04d_pert_%1.3e.pdf"%(stab,order,Ns[0],pert_val))
            plt.close()

            fig = plot_all_sols(problem, FEM2D, q_save, len(q_save_GF)-1, tt_save_GF[len(q_save_GF)-1], levels=None)
            fig.tight_layout()
            fig.savefig(problem.folderName+"/final_sol_%s_GF_ord%d_N%04d_pert_%1.3e.pdf"%(stab,order,Ns[0],pert_val))
            plt.close()


            diff   = dict()
            diffGF = dict()
            for var in problem.vars:
                diff[var] = q_save[var]-solver.ic_no_pert[var]
                diffGF[var]   = q_save_GF[var]-solver.ic_no_pert[var]
            v2   = q_save["u"][-1,:]**2+q_save["v"][-1,:]**2
            v2GF = q_save_GF["u"][-1,:]**2+q_save_GF["v"][-1,:]**2
            v2IC = solver.ic_no_pert["u"]**2+ solver.ic_no_pert["v"]**2

            diffv2   = v2  -v2IC
            diffv2GF = v2GF-v2IC

            fig = plot_all_sols(problem, FEM2D, diff, len(q_save)-1, tt_save[len(q_save)-1], levels=None)
            fig.tight_layout()
            fig.savefig(problem.folderName+"/pert_%s_NOGF_ord%d_N%04d_pert_%1.3e.pdf"%(stab,order,Ns[0],pert_val))
            plt.close()
            fig = plot_all_sols(problem, FEM2D, diffGF, len(q_save_GF)-1, tt_save_GF[len(q_save_GF)-1], levels=None)
            fig.tight_layout()
            fig.savefig(problem.folderName+"/pert_%s_GF_ord%d_N%04d_pert_%1.3e.pdf"%(stab,order,Ns[0],pert_val))
            plt.close()

            fig, axs = plt.subplots(1,2, figsize=(10,4))
            plot_sol(FEM2D, np.abs(diffv2) , axs[0],fig)
            plot_sol(FEM2D, np.abs(diffv2GF),axs[1],fig)
            # plot_sol(FEM2D, np.abs(v2IC),axs[2],fig, levels=21)
            axs[0].set_title("Non Global Flux")
            axs[1].set_title("Global Flux")

            fig.tight_layout()
            fig.savefig(problem.folderName+"/vel_squared_diff_%s_ord%d_N%04d_pert_%1.3e.pdf"%(stab,order,Ns[0],pert_val))
            plt.close()


            it = -1
            fig, axs = plt.subplots(1,2, figsize=(10,4))
            plot_sol(FEM2D, np.sqrt(diff["u"][it,:]**2+diff["v"][it,:]**2) , axs[0],fig)
            plot_sol(FEM2D, np.sqrt(diffGF["u"][it,:]**2+diffGF["v"][it,:]**2),axs[1],fig)
            # plot_sol(FEM2D, np.abs(v2IC),axs[2],fig, levels=21)
            axs[0].set_title("Non Global Flux")
            axs[1].set_title("Global Flux")

            fig.tight_layout()
            fig.savefig(problem.folderName+"/diff_vel_norm_%s_ord%d_N%04d_pert_%1.3e.pdf"%(stab,order,Ns[0],pert_val))
            plt.close()

