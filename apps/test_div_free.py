import numpy as np
from gfsupg.solver import CartesianGeometry, FiniteElement1D, Scipy2DFEM
from gfsupg.solver import DeC, DeCSpaceTimeSUPGSolver
from gfsupg.problem_old import LinearAcoustic2D
from gfsupg.plotting import *

import matplotlib.pyplot as plt


from scipy.optimize import LinearConstraint, minimize
from scipy.sparse import hstack, vstack, identity




#problem.T_fin = 1.
order=7
Nx = 6
Ny = Nx

FEM1Dx = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
FEM1Dy = FiniteElement1D(order-1,"gaussLobatto","gaussLobatto")
dec = DeC((order+1)//2,order,"gaussLobatto")

problem_tags = ["vortex", "smooth_vortex"]
for problem_tag in problem_tags:
    problem = LinearAcoustic2D(problem_tag)

    Ns = np.array([Nx,Ny], dtype=np.int32)

    geom = CartesianGeometry(problem.xL,problem.xR, Ns, problem.geometry_folder, BC=problem.BC)
    FEM2D = Scipy2DFEM(geom,FEM1Dx, FEM1Dy, folder=problem.folderName)
    solver = DeCSpaceTimeSUPGSolver(problem, FEM2D, dec)

    #%% Davide Approach
    div_oper = hstack([FEM2D.operator["IDx_tilde"],FEM2D.operator["IDy_tilde"]] )

    linear_constraint = LinearConstraint(div_oper , np.zeros(div_oper.shape[0]), np.zeros(div_oper.shape[0]))


    u = solver.ic_vect["u"]
    v = solver.ic_vect["v"]

    u_ic = np.concatenate([u,v])
    u_ic.shape

    def target_function(u):
        return np.sum((u-u_ic)**2)
    def target_function_dir(u):
        return 2.*(u-u_ic)
    def target_function_hess(u):
        return 2.*identity(len(u))

    x0 = np.array([0.5, 0])
    res = minimize(target_function, u_ic, method='trust-constr', jac=target_function_dir, hess=target_function_hess,
                constraints=[linear_constraint],
                options={'verbose': 1})


    q_davide_vec = dict()
    q_davide_vec["u"]       = res.x[:len(u)]
    q_davide_vec["v"]       = res.x[len(u):]

    div_davide_vec = FEM2D.compute_discrete_divergence(q_davide_vec)

    div_davide_mat = FEM2D.vect_to_mat(div_davide_vec)

    aaa = np.mean(np.abs(div_davide_mat[:-1,:-1]))
    bbb = np.mean(np.abs(div_davide_mat))
    with open(problem.folderName+"/div_disc_davide_ord%d_N%03d.csv"%(order,Nx),"w") as file:
        file.write("Disc div all, disc div not bord \n")
        file.write(" %1.5g, %1.5g"%(aaa,bbb) )

    plt.contourf(FEM2D.xx_mat[0], FEM2D.xx_mat[1], div_davide_mat)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(problem.folderName+"/div_disc_davide_ord%d_N%03d.pdf"%(order,Nx))
    plt.close()

    div_NGF = (FEM2D.operator["IDx"]@q_davide_vec["u"]+FEM2D.operator["IDy_tilde"]@q_davide_vec["v"])/geom.dx[0]/geom.dx[1]

    div_NGF_mat = FEM2D.vect_to_mat(div_NGF)

    plt.contourf(FEM2D.xx_mat[0], FEM2D.xx_mat[1], div_NGF_mat)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(problem.folderName+"/div_disc_ic_ord%d_N%03d.pdf"%(order,Nx))
    plt.close()


    u_mat = FEM2D.vect_to_mat(u)
    v_mat = FEM2D.vect_to_mat(v)

    plt.contourf(FEM2D.xx_mat[0], FEM2D.xx_mat[1], u_mat)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(problem.folderName+"/u_ic_ord%d_N%03d.pdf"%(order,Nx))
    plt.close()

    plt.contourf(FEM2D.xx_mat[0], FEM2D.xx_mat[1], FEM2D.vect_to_mat(q_davide_vec["u"]))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(problem.folderName+"/u_davide_ord%d_N%03d.pdf"%(order,Nx))
    plt.close()


    plt.contourf(FEM2D.xx_mat[0], FEM2D.xx_mat[1], v_mat)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(problem.folderName+"/v_ic_ord%d_N%03d.pdf"%(order,Nx))
    plt.close()

    plt.contourf(FEM2D.xx_mat[0], FEM2D.xx_mat[1], FEM2D.vect_to_mat(q_davide_vec["v"]))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(problem.folderName+"/v_davide_ord%d_N%03d.pdf"%(order,Nx))
    plt.close()

    plt.contourf(FEM2D.xx_mat[0], FEM2D.xx_mat[1], v_mat-FEM2D.vect_to_mat(q_davide_vec["v"]))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(problem.folderName+"/v_err_davide_ord%d_N%03d.pdf"%(order,Nx))
    plt.close()


    #%% Mario's approach

    u = solver.ic_vect["u"]
    v = solver.ic_vect["v"]

    u_mat = FEM2D.vect_to_mat(u)
    v_mat = FEM2D.vect_to_mat(v)

    u_mario = np.empty_like(u_mat)
    v_mario = np.empty_like(v_mat)


    # Solve I^y u(x_m) l = int y_0^y_l u^ex(x_m,y) dy

    from gfsupg.quadr import nodes_weights
    HO_quad_nodes, HO_quad_weights = nodes_weights(10, "gaussLegendre")

    RHS = np.zeros(FEM1Dy.degree)
    for m, x in enumerate(FEM2D.xx_dofs[0]):
        u_mario[m, 0] = problem.ics["u"](x, geom.xL[1])

        # Solve I^y u(x_m) l = int y_0^y_l u^ex(x_m,y) dy
        for j_cell in range(geom.N_elem_dir[1]):
            j_cell_global = j_cell*FEM1Dy.degree      
            j1_cell_global = (j_cell+1)*FEM1Dy.degree
            for j_dof in range(1,FEM1Dy.degree+1):
                j_dof_global = j_cell_global+j_dof 
                dy = FEM2D.xx_dofs[1][j_dof_global] - geom.xx[1][j_cell]
                quad_nodes = HO_quad_nodes*dy +geom.xx[1][j_cell]
                quad_weights =  HO_quad_weights*dy
                RHS[j_dof-1] = np.sum(np.vectorize(problem.ics["u"])(x, quad_nodes) * quad_weights)
            
            dy = geom.xx[1][j_cell+1]-geom.xx[1][j_cell]
            RHS = RHS/dy- FEM1Dy.matrix["int_mat"][1:,0]*u_mario[m, j_cell_global]
            u_mario[ m, j_cell_global+1:j1_cell_global+1 ] = np.linalg.solve(FEM1Dy.matrix["int_mat"][1:,1:],RHS)




    # Solve I^x v(y_l) m = int x_0^x_m v^ex(x,y_l) dx

    RHS = np.zeros(FEM1Dx.degree)
    for l, y in enumerate(FEM2D.xx_dofs[1]):
        v_mario[0,l] = problem.ics["v"](geom.xL[0], y )

        # Solve I^x v(y_l) m = int x_0^x_m v^ex(x,y_l) dx
        for j_cell in range(geom.N_elem_dir[0]):
            j_cell_global = j_cell*FEM1Dx.degree      
            j1_cell_global = (j_cell+1)*FEM1Dx.degree
            for j_dof in range(1,FEM1Dx.degree+1):
                j_dof_global = j_cell_global+j_dof 
                dx = FEM2D.xx_dofs[0][j_dof_global] - geom.xx[0][j_cell]
                quad_nodes = HO_quad_nodes*dx +geom.xx[0][j_cell]
                quad_weights =  HO_quad_weights*dx
                RHS[j_dof-1] = np.sum(np.vectorize(problem.ics["v"])(quad_nodes, y) * quad_weights)
            
            dx = geom.xx[0][j_cell+1]-geom.xx[0][j_cell]
            RHS = RHS/dx- FEM1Dx.matrix["int_mat"][1:,0]*v_mario[ j_cell_global, l]
            v_mario[ j_cell_global+1:j1_cell_global+1, l ] = np.linalg.solve(\
                FEM1Dx.matrix["int_mat"][1:,1:],RHS)




    q_mario_vec = dict()
    q_mario_vec["u"]       = FEM2D.mat_to_vect(u_mario)
    q_mario_vec["v"]       = FEM2D.mat_to_vect(v_mario)

    div_mario_vec = FEM2D.compute_discrete_divergence(q_mario_vec)

    div_mario_mat = FEM2D.vect_to_mat(div_mario_vec)

    aaa = np.mean(np.abs(div_mario_mat[:-1,:-1]))
    bbb = np.mean(np.abs(div_mario_mat))
    with open(problem.folderName+"/div_disc_mario_ord%d_N%03d.csv"%(order,Nx),"w") as file:
        file.write("Disc div all, disc div not bord \n")
        file.write(" %1.5g, %1.5g"%(aaa,bbb) )


    plt.contourf(FEM2D.xx_mat[0], FEM2D.xx_mat[1], div_mario_mat)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(problem.folderName+"/div_disc_mario_ord%d_N%03d.pdf"%(order,Nx))
    plt.close()

    div_NGF = (FEM2D.operator["IDx"]@q_mario_vec["u"]+FEM2D.operator["IDy_tilde"]@q_mario_vec["v"])/geom.dx[0]/geom.dx[1]

    div_NGF_mat = FEM2D.vect_to_mat(div_NGF)






    plt.contourf(FEM2D.xx_mat[0], FEM2D.xx_mat[1], u_mario)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(problem.folderName+"/u_mario_ord%d_N%03d.pdf"%(order,Nx))
    plt.close()


    plt.contourf(FEM2D.xx_mat[0], FEM2D.xx_mat[1], v_mario)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(problem.folderName+"/v_mario_ord%d_N%03d.pdf"%(order,Nx))
    plt.close()

    plt.contourf(FEM2D.xx_mat[0], FEM2D.xx_mat[1], v_mat-v_mario)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(problem.folderName+"/v_err_mario_ord%d_N%03d.pdf"%(order,Nx))
    plt.close()
