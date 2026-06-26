import numpy as np
from gfsupg.solver import CartesianGeometry, FiniteElement1D, ImplicitDec
from gfsupg.solver import Scipy2DFEM, DeC, DeCSpaceTimeSUPGSolver
from gfsupg.problem import *
from gfsupg.plotting import *
import pickle
import csv

import matplotlib.pyplot as plt

problem = ObliqueTestCase() 


stab ="SUPG"#"OSS" #"SUPG" #"OSS" #

GFs = [True]#, False]
GF_names = ["GF"]# "noGF"]
save_solutions = False
save_final_solution = True
save_IC = True

mesh_sizes_one = [20,40,80,160] #[10,20,40,80]#,160,320]#[10,20]# [20, 30, 40, 80]#, 160, 320]

for order in range(2,8):

    FEM1Dx = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
    FEM1Dy = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
    dec = DeC((order+1)//2,order,"gaussLobatto")

    mesh_sizes = np.array([np.int32(N//(order-1)) for N in mesh_sizes_one])

    N_sizes = len(mesh_sizes)
    errors        = np.zeros((len(GFs),N_sizes, problem.n_eq))
    errors_vertex = np.zeros((len(GFs),N_sizes, problem.n_eq))
    errors_div_ic = np.zeros((len(GFs),N_sizes))
    comp_times    = np.zeros((len(GFs),N_sizes))
    orders        = np.zeros((len(GFs),N_sizes, problem.n_eq))
    orders_vertex = np.zeros((len(GFs),N_sizes, problem.n_eq))
    orders_div_ic = np.zeros((len(GFs),N_sizes))

    for iN, N in enumerate(mesh_sizes):
        print("N = %d"%N)
        Ns = np.array([ N, N], dtype=np.int32)

        geom = CartesianGeometry(problem.xL,problem.xR, Ns, problem.geometry_folder, BC=problem.BC)


        FEM2D = Scipy2DFEM(geom,FEM1Dx, FEM1Dy, folder=problem.folderName)

        fig_div, ax_div = plt.subplots(1,1)
        for iGF, GF in enumerate(GFs):
            print(GF_names[iGF])
            if GF:
                method_name = stab+"_GF"
                error_name = "errors_"+stab+"_GF"
            else:
                method_name = stab
                error_name = "errors_"+stab

            # solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec, GF=GF, stab=stab)
            solver = ImplicitDec(problem, FEM2D, dec, GF = GF, stab = stab, trick_second_der=False)

            if GF:
                disc_div = FEM2D.compute_discrete_divergence(solver.ic_vect)
            else:
                disc_div = FEM2D.compute_wrong_discrete_divergence(solver.ic_vect)
                
            errors_div_ic[iGF, iN] = np.linalg.norm(disc_div)/np.sqrt(FEM2D.n_dof_tot)
            
            q_save, tt_save, comp_times[iGF, iN], err, err_vertex = solver.solve(with_error = True, with_error_vertex= True)



            if save_solutions:
                sol_to_save = [q_save, tt_save, comp_times[iGF, iN], err, err_vertex ]
                # Open a file and use dump()
                savefile_name = problem.folderName+"/simul_"+method_name+"_ord_%d_N_%04d.pkl"%(order,N)
                with open(savefile_name, 'wb') as file:
                    # A new file will be created
                    pickle.dump(sol_to_save, file)

            if save_final_solution:
                q_final = dict()
                for var in problem.vars:
                    q_final[var] = q_save[var][-1]
                sol_to_save = [q_final, tt_save[-1], comp_times[iGF, iN], err, err_vertex ]
                # Open a file and use dump()
                savefile_name = problem.folderName+"/final_sol_"+method_name+"_ord_%d_N_%04d.pkl"%(order,N)
                with open(savefile_name, 'wb') as file:
                    # A new file will be created
                    pickle.dump(sol_to_save, file)


            errors[iGF, iN, :] = err
            print("Errors ", err)

            errors_vertex[iGF, iN, :] = err_vertex
            print("Errors Vertex ", err_vertex)

            if GF:
                div_error = FEM2D.compute_discrete_divergence_norms(q_save)
            else:
                div_error = FEM2D.compute_wrong_discrete_divergence_norms(q_save)
            ax_div.semilogy(tt_save, div_error,label=method_name)
            with open (problem.folderName+"/div_err_time_"+method_name+"_ord_%d_N_%04d.csv"%(order,N),"w") as file:
                writer = csv.writer(file)

                header = ["t", "div"]
                writer.writerow(header)

                for it, t in enumerate(tt_save):
                    line = [t, div_error[it]]
                    writer.writerow(line)



            if iN>0:
                for k in range(problem.n_eq):
                    orders[iGF, iN, k] = np.log(errors[iGF, iN,k]/(errors[iGF, iN-1,k] + 1e-15))/np.log(mesh_sizes[iN-1]/mesh_sizes[iN])
                    orders_vertex[iGF, iN, k] = np.log(errors_vertex[iGF, iN,k]/errors_vertex[iGF, iN-1,k])/np.log(mesh_sizes[iN-1]/mesh_sizes[iN])
                orders_div_ic[iGF, iN] = np.log(errors_div_ic[iGF, iN]\
                                        /errors_div_ic[iGF, iN-1])/np.log(mesh_sizes[iN-1]/mesh_sizes[iN])

            # Plot final simulation
            fig = plot_all_sols(problem, FEM2D, q_save, -1, tt_save[-1])
            fig.tight_layout()
            fig.savefig(problem.folderName+"/simul_"+method_name+"_ord_%d_N_%04d.pdf"%(order,N))
            plt.close(fig)

        if save_IC:      
            fig = plot_one_sol(problem, FEM2D, solver.ic_vect)
            fig.tight_layout()
            fig.savefig(problem.folderName+"/IC_ord_%d_N_%04d.pdf"%(order,N))
            plt.close(fig)


        ax_div.set_xlabel("Time")
        ax_div.set_ylabel("Divergence error")
        fig_div.tight_layout()
        fig_div.legend(loc="upper right")
        fig_div.savefig(problem.folderName+"/div_error_%s_ord_%d_N_%04d.pdf"%(stab,order,N))
        plt.close(fig_div)
        
    for iGF, GF in enumerate(GFs):
        print(GF_names[iGF])
        if GF:
            error_name = "errors_"+stab+"_GF"
        else:
            error_name = "errors_"+stab

        with open (problem.folderName+"/"+error_name+"_ord%d.csv"%order,"w") as file:
            writer = csv.writer(file)

            header = ["N", "comp time", \
                      "err u", "err v", "err p", "ord u", "ord v", "ord p",\
                      "err vert u", "err vert v", "err vert p", "ord vert u", "ord vert v", "ord vert p"]
            writer.writerow(header)

            for iN, N in enumerate(mesh_sizes):
                line = [N, comp_times[iGF, iN],\
                        errors[iGF, iN,0], errors[iGF, iN,1], errors[iGF, iN,2],\
                        orders[iGF, iN,0], orders[iGF, iN,1], orders[iGF, iN,2],\
                        errors_vertex[iGF, iN,0], errors_vertex[iGF, iN,1], errors_vertex[iGF, iN,2],\
                        orders_vertex[iGF, iN,0], orders_vertex[iGF, iN,1], orders_vertex[iGF, iN,2] ]
                writer.writerow(line)

    plt.figure()
    for iGF, GF in enumerate(GFs):
        if GF:
            plt.loglog(mesh_sizes, errors[iGF,:,:], linewidth=3, label = [var + " GF"*GF for var in problem.vars])
        else:
            plt.loglog(mesh_sizes, errors[iGF,:,:], "--", label = problem.vars)

    plt.loglog(mesh_sizes, 1./np.float64(mesh_sizes)**order * errors[iGF, -1,0]*mesh_sizes[-1]**order , ":", label="order %d"%order)
    plt.legend()
    plt.ylabel("Error")
    plt.xlabel("Elements in one direction")
    plt.tight_layout()
    plt.savefig(problem.folderName+"/"+error_name+"_ord%d.pdf"%order)
    plt.show(block=False)
    plt.close('all')

    plt.figure()
    for iGF, GF in enumerate(GFs):
        if GF:
            plt.loglog(mesh_sizes, errors_vertex[iGF,:,:], linewidth=3, label = [var + " GF"*GF for var in problem.vars])
        else:
            plt.loglog(mesh_sizes, errors_vertex[iGF,:,:], "--", label = problem.vars)

    plt.loglog(mesh_sizes, 1./np.float64(mesh_sizes)**order * errors_vertex[iGF, -1,0]*mesh_sizes[-1]**order , ":", label="order %d"%order)
    plt.legend()
    plt.ylabel("Error in vertexes")
    plt.xlabel("Elements in one direction")
    plt.tight_layout()
    plt.savefig(problem.folderName+"/"+error_name+"_vertex_ord%d.pdf"%order)
    plt.show(block=False)
    plt.close('all')


    experimental_order = np.mean(orders_div_ic[0,1:])
    guess_order = np.ceil(np.mean(orders_div_ic[0,1:]))
    if  np.isnan(guess_order):
        guess_order = 1

    plt.figure()
    iGF = 0
    GF = True
    plt.loglog(mesh_sizes, errors_div_ic[iGF,:], linewidth=4, label ="Discrete Divergence p="+str(order-1))

    for (shift, linestyle) in zip(range(-1,2),["--","-.",":"]):
        plt.loglog(mesh_sizes, 1./np.float64(mesh_sizes)**(guess_order+shift) * errors_div_ic[iGF, 1]*mesh_sizes[1]**(guess_order+shift) , linestyle=linestyle, label="order %d"%(guess_order+shift))
    plt.legend()
    plt.ylabel("Discrete divergence of analytical IC")
    plt.xlabel("Elements in one direction")
    plt.title( "Convergence discrete divergence (experimental order %1.2f)"%experimental_order) 
    plt.tight_layout()
    plt.savefig(problem.folderName+"/discrete_divergence_IC_%s_ord%d.pdf"%(stab,order))
#    plt.show(block=False)
    plt.close('all')
