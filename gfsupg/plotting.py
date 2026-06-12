from .problem import LinearAcoustic2D
from .solver import Scipy2DFEM
import matplotlib.pyplot as plt
import numpy as np

def plot_sol(FEM2D, uu, ax, fig, levels = None, no_boundaries = False):
    uu_mat = FEM2D.vect_to_mat(uu)
    if uu_mat.min()==uu_mat.max() or levels is None:
        if no_boundaries:
            cnt = ax.contourf(FEM2D.xx_mat[0][1:-1,1:-1],FEM2D.xx_mat[1][1:-1,1:-1],uu_mat[1:-1,1:-1])
        else:
            cnt = ax.contourf(FEM2D.xx_mat[0],FEM2D.xx_mat[1],uu_mat)
    elif isinstance(levels,int):
        if no_boundaries:
            lvls = np.linspace(uu_mat[1:-1,1:-1].min(),uu_mat[1:-1,1:-1].max(), levels)
            cnt = ax.contourf(FEM2D.xx_mat[0][1:-1,1:-1],FEM2D.xx_mat[1][1:-1,1:-1],uu_mat[1:-1,1:-1], lvls)
        else:
            lvls = np.linspace(uu_mat.min(),uu_mat.max(), levels)
            cnt = ax.contourf(FEM2D.xx_mat[0],FEM2D.xx_mat[1],uu_mat, lvls)

    else:
        if no_boundaries:
            cnt = ax.contourf(FEM2D.xx_mat[0][1:-1,1:-1],FEM2D.xx_mat[1][1:-1,1:-1],uu_mat[1:-1,1:-1], levels)
        else:
            cnt = ax.contourf(FEM2D.xx_mat[0],FEM2D.xx_mat[1],uu_mat, levels)
    ax.axis("equal")
    fig.colorbar(cnt, ax=ax)

def plot_all_sols(problem, FEM2D, q_save, it, time, levels=None):
    fig, axs = plt.subplots(1,problem.n_eq, figsize=(15,4))
    for k, var in enumerate(problem.vars):
        axs[k].set_title(var)
        if levels is None:
            if it is None:
                plot_sol(FEM2D,q_save[var][:], axs[k],fig) 
            else:
                plot_sol(FEM2D,q_save[var][it,:], axs[k],fig) 
        else:
            if it is None:
                plot_sol(FEM2D,q_save[var][:], axs[k],fig, levels) 
            else:
                plot_sol(FEM2D,q_save[var][it,:], axs[k],fig, levels) 
    fig.suptitle("Time %1.4f"%time)
    return fig

def plot_one_sol(problem, FEM2D, q_save, levels=None, no_boundaries = False):
    fig, axs = plt.subplots(1,problem.n_eq, figsize=(15,4))
    for k, var in enumerate(problem.vars):
        axs[k].set_title(var)
        if levels is None:
            plot_sol(FEM2D,q_save[var], axs[k],fig, no_boundaries=no_boundaries) 
        else:
            plot_sol(FEM2D,q_save[var], axs[k],fig, levels, no_boundaries=no_boundaries) 
    return fig


def plot_all_sols_errs(problem, FEM2D, q_save, it, time, levels=None):
    fig, axs = plt.subplots(1,problem.n_eq, figsize=(15,4))
    for k, var in enumerate(problem.vars):
        axs[k].set_title(var)
        plot_sol(FEM2D,q_save[var][it,:], axs[k],fig, levels) 
    fig.suptitle("Approx Time %1.4f"%time)

    fig, axs = plt.subplots(1,problem.n_eq, figsize=(15,4))
    for k, var in enumerate(problem.vars):
        ex = FEM2D.evaluate_function(lambda x,y: problem.exact[var](x,y,time))
        axs[k].set_title(var)
        plot_sol(FEM2D,ex, axs[k],fig, levels) 
    fig.suptitle("Exact Time %1.4f"%time)

def compute_errors(problem, FEM2D, q_save, it, time):
    err = np.zeros(len(problem.vars))
    rel_err = np.zeros(len(problem.vars))
    for k, var in enumerate(problem.vars):
        ex = FEM2D.evaluate_function(lambda x,y: problem.exact[var](x,y,time))
        err[k] = np.linalg.norm(q_save[var][it,:]-ex)/np.sqrt(FEM2D.n_dof_tot)
        rel_err[k] = np.linalg.norm(q_save[var][it,:]-ex)/(np.linalg.norm(ex)+1e-10)
    return err, rel_err

def compute_integral_error(problem, FEM2D, q_save, tt_save):
    for it,time in enumerate(tt_save[1:]):
        err = np.zeros(len(problem.vars))
        rel_err = np.zeros(len(problem.vars))
        for k, var in enumerate(problem.vars):
            dt = tt_save[it]-tt_save[it-1]
            ex = FEM2D.evaluate_function(lambda x,y: problem.exact[var](x,y,time))
            err[k] += np.linalg.norm(q_save[var][it,:]-ex)/np.sqrt(FEM2D.n_dof_tot)*dt
    return err
