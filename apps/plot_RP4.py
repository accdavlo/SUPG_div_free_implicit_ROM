import numpy as np
from gfsupg.solver import CartesianGeometry, FiniteElement1D, Scipy2DFEM
from gfsupg.solver import DeC, DeCSpaceTimeSUPGSolver
from gfsupg.problem_old import LinearAcoustic2D
from gfsupg.problem_old import exact_radial_RP4
from gfsupg.plotting import *
import pickle
import matplotlib.pyplot as plt
import os


problem = LinearAcoustic2D("RP4")

meshes = [100, 200, 50, 25]
orders = [  2,   2,  3,  5]
symbols = ["x", "+","1","."]

len_meth = len(meshes)
rr = 2**np.linspace(-8,-0.5,200)

for GF, GF_tag in zip([True,False], ["GF", "NOGF"]):
    fig_rad, ax_rad = plt.subplots(1,1)

    for i_meth in range(len_meth):
        order = orders[i_meth]
        N     = meshes[i_meth]

        FEM1Dx = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
        FEM1Dy = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
        dec = DeC((order+1)//2,order,"gaussLobatto")

        Ns = np.array([N,N], dtype=np.int32)

        geom = CartesianGeometry(problem.xL,problem.xR, \
                                Ns, problem.geometry_folder, BC=problem.BC)

        FEM2D = Scipy2DFEM(geom,FEM1Dx, FEM1Dy, folder=problem.folderName)
        solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec)
        
        savefile_name = problem.folderName+"/final_sol_"+GF_tag+"_ord_%d_N_%04d.pkl"%(order,N)
        
        if os.path.exists(savefile_name):
            with open(savefile_name, 'rb') as file:
                sol_to_save = pickle.load(file)
                q_final = sol_to_save[0]
                t_final = sol_to_save[1]
                comp_time = sol_to_save[2]
                err       = sol_to_save[3]
                err_vertex = sol_to_save[4]

        else:
            q_save, tt_save, comp_time, err, err_vertex  = \
                solver.solve(with_error = True, \
                            with_error_vertex = True, GF=GF)
            
            t_final = tt_save[-1]
            q_final = dict()
            for var in problem.vars:
                q_final[var] = q_save[var][-1]
            sol_to_save = [q_final, t_final, comp_time, err, err_vertex ]
            # Open a file and use dump()
            with open(savefile_name, 'wb') as file:
                # A new file will be created
                pickle.dump(sol_to_save, file)


        plot_all_sols(problem, FEM2D, q_final, None, t_final, 200)
        plt.savefig(savefile_name[:-3]+"pdf")
        plt.close()

        
        # plt.plot(FEM2D.mat_to_vect(np.sqrt((FEM2D.xx_mat[0]-0.5)**2+\
        #         (FEM2D.xx_mat[1]-0.5)**2)), q_save["v"][-1,:], '.')
        # plt.plot(rr, np.vectorize(exact_radial_RP4)(rr,tt_save[-1]), '--')

        ax_rad.semilogx(FEM2D.mat_to_vect(np.sqrt((FEM2D.xx_mat[0]-0.5)**2+\
                    (FEM2D.xx_mat[1]-0.5)**2)), q_final["v"][:], symbols[i_meth], label="ord%d N%d"%(order,N))
    

        fig_single, ax_single = plt.subplots(1,1)
        ax_single.semilogx(FEM2D.mat_to_vect(np.sqrt((FEM2D.xx_mat[0]-0.5)**2+\
                    (FEM2D.xx_mat[1]-0.5)**2)), q_final["v"][:], symbols[i_meth], label="ord%d N%d"%(order,N))
        ax_single.semilogx(rr, np.vectorize(exact_radial_RP4)(rr,t_final), '--', label="exact")
        ax_single.set_xlabel("radius")
        ax_single.set_ylabel("v")
        fig_single.tight_layout()
        fig_single.legend()
        fig_single.savefig(problem.folderName+"/radial distribution_"+GF_tag+"_ord_%d_N_%04d.pdf"%(order,N))
    
    ax_rad.semilogx(rr, np.vectorize(exact_radial_RP4)(rr,t_final), '--', label="exact")
    ax_rad.set_xlabel("radius")
    ax_rad.set_ylabel("v")
    fig_rad.tight_layout()
    fig_rad.legend()
    fig_rad.savefig(problem.folderName+"/radial distribution_"+GF_tag+".pdf")
