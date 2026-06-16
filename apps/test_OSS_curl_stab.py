import numpy as np
from gfsupg.solver import CartesianGeometry, FiniteElement1D, Scipy2DFEM
from gfsupg.solver import DeC, DeCSpaceTimeSUPGSolver
from gfsupg.problem_old import LinearAcoustic2D
from gfsupg.plotting import *

import matplotlib.pyplot as plt

#problem.T_fin = 1.
order=3

FEM1Dx = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
FEM1Dy = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
dec = DeC((order+1)//2,order,"gaussLobatto")
# dec = DeC(4,5,"gaussLobatto")

problem = LinearAcoustic2D("smooth_vortex_long")
problem.T_fin = 10
Ns = np.array([20,20], dtype=np.int32)

geom = CartesianGeometry(problem.xL,problem.xR, Ns, problem.geometry_folder, BC=problem.BC)

FEM2D = Scipy2DFEM(geom,FEM1Dx, FEM1Dy, folder=problem.folderName)

solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec, GF = True, stab = "SUPG", trick_second_der=False)

labels = []
q_save = dict()
tt_save = dict()
comp_time= dict()
err = dict()
err_vertex = dict()
i=-1
for stab in ["OSS"]:#,"SUPG"]:
    for trick in [False]:#True, False]:
        for GF in [True, False]:
            for curl_stab_flag in [False,True]:
                i+=1
                labels.append( stab + "_GF"*GF + "_trick"*trick +"_curl_stab"*curl_stab_flag)
                print(labels[-1])

                q_save[labels[i]], tt_save[labels[i]], comp_time[labels[i]], err[labels[i]], err_vertex[labels[i]]  = \
            solver.solve(with_error = True, with_error_vertex = True, GF=GF,\
                        stab=stab, trick_second_der=trick, curl_stab_flag=curl_stab_flag)

                plot_all_sols(problem, FEM2D, q_save[labels[i]], -1, problem.T_fin, levels=None)
plt.show()

i+=1
labels.append("Gal")
q_save[labels[i]], tt_save[labels[i]], comp_time[labels[i]], err[labels[i]], err_vertex[labels[i]]  = \
        solver.solve(with_error = True, with_error_vertex = True, GF=False,\
                    stab="SUPG", trick_second_der=False, stab_coeff=0.)

q_tmp = dict()
for var in problem.vars:
    q_tmp[var] = solver.ic_vect[var]

styles = ["-","--", "-", "--","-.",":", "-.", ":",":"]
colors = ["k","k","b","b","g","g","r","r","m"]

curl0 = FEM2D.compute_discrete_curl(q_tmp)
err_curl = dict()

for il, lab in enumerate(labels):
    err_curl[lab] = np.zeros(len(tt_save[lab]))
    for it, t in enumerate(tt_save[lab]):
        q_tmp = dict()
        for var in problem.vars:
            q_tmp[var] = q_save[lab][var][it]
        curl  =  FEM2D.compute_discrete_curl(q_tmp)
        curl_diff = curl-curl0
        err_curl[lab][it] =  curl_diff.T @ (FEM2D.operator["mass"] @ curl_diff)

    plt.semilogy(tt_save[lab][1:-1], err_curl[lab][1:-1], label=lab, linestyle=styles[il], color=colors[il])
plt.legend()
plt.ylabel(r"$|| \nabla \times u^{n+1} - \nabla \times u^0 ||$")
plt.xlabel("Time")
plt.savefig(problem.folderName+"/curl_error_ord_%d_N_%04d.pdf"%(order,Ns[0]))
plt.show()